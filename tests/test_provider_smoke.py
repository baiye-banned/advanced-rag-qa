"""Provider smoke acceptance goal (docs/MAINTENANCE_GOAL_CLOSURE.md, goal 4).

Every provider client the RAG chain talks to -- the OpenAI-compatible embedding API, the
qwen3 reranker, the DeepSeek chat model and its two fallbacks -- is driven here against
in-process stubs, covering the success contract, failures, timeouts and malformed
responses. The point of the failure cases is that none of them may stay silent: a
configured embedding backend raises EmbeddingBackendError instead of quietly switching to
hash vectors (which live in a different vector space), a failed rerank reports a failed
trace entry, and a failed answer stream either reports an error event or re-generates the
whole answer behind a reset event.

The last two tests run scripts/smoke_providers.py, the operator-facing entry point: once
offline (every provider reported, zero exit code) and once in `--live` mode without
credentials, where the summary must report the skips instead of claiming a pass.

No network access is required.
"""

import asyncio
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from rag import llm, milvus_client, rerank

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_BASE_URL = "https://embedding.test/v1"
RERANK_BASE_URL = "https://rerank.test/v1/reranks"
CHUNKS = [
    {"file_id": 1, "chunk_id": "0", "content": "员工迟到一次罚款50元。", "file_name": "考勤制度.txt"},
    {"file_id": 1, "chunk_id": "1", "content": "报销需要在30天内提交。", "file_name": "报销制度.txt"},
]


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

    def json(self):
        return self._payload


class _StubSyncClient:
    """Records every embedding request; ``behaviour`` decides what the call does."""

    calls = []

    def __init__(self, behaviour, *args, **kwargs):
        self._behaviour = behaviour
        self.kwargs = kwargs
        type(self).calls.append({"timeout": kwargs.get("timeout")})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        self.calls[-1].update({"url": url, "json": json, "headers": headers})
        return self._behaviour(len(json.get("input") or []))


class _CapturingAsyncClient:
    calls = []

    def __init__(self, behaviour, *args, **kwargs):
        self._behaviour = behaviour
        type(self).calls.append({"timeout": kwargs.get("timeout")})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        self.calls[-1].update({"url": url, "json": json, "headers": headers})
        return self._behaviour(json)


def _install_embedding_client(monkeypatch, behaviour):
    _StubSyncClient.calls = []
    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-embedding-key")
    monkeypatch.setattr(
        milvus_client.httpx, "Client", lambda *args, **kwargs: _StubSyncClient(behaviour, *args, **kwargs)
    )
    return _StubSyncClient.calls


def _forbidden_hash_fallback(*_args, **_kwargs):  # pragma: no cover - must never run
    """Fail loudly if a configured embedding backend degrades to local hash vectors."""
    raise AssertionError("configured embedding backend fell back to hash vectors")


def _install_rerank_client(monkeypatch, behaviour):
    _CapturingAsyncClient.calls = []
    monkeypatch.setattr(rerank, "RERANK_BASE_URL", RERANK_BASE_URL)
    monkeypatch.setattr(rerank, "RERANK_API_KEY", "test-rerank-key")
    monkeypatch.setattr(
        rerank.httpx, "AsyncClient", lambda *args, **kwargs: _CapturingAsyncClient(behaviour, *args, **kwargs)
    )
    return _CapturingAsyncClient.calls


def _embedding_payload(count):
    return {"data": [{"index": index, "embedding": [0.5] * milvus_client.EMBEDDING_DIM} for index in range(count)]}


def test_embedding_request_contract_and_vectors(monkeypatch, caplog):
    # The embedding timeout is configuration (config.EMBEDDING_TIMEOUT_SECONDS), read at call time
    # through the milvus_client module global. Pinning a value that matches neither the config
    # default nor any hard-coded literal keeps the assertion below falsifiable: a client built
    # from a stale constant goes red here instead of passing by coincidence.
    monkeypatch.setattr(milvus_client, "EMBEDDING_TIMEOUT_SECONDS", 12)
    calls = _install_embedding_client(monkeypatch, lambda count: _Response(_embedding_payload(count)))

    with caplog.at_level(logging.WARNING, logger=milvus_client.__name__):
        vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["迟到处罚", "报销流程"])

    assert len(vectors) == 2
    assert all(len(vector) == milvus_client.EMBEDDING_DIM for vector in vectors)
    assert vectors[0] == [0.5] * milvus_client.EMBEDDING_DIM

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == f"{EMBEDDING_BASE_URL}/embeddings"
    assert call["json"] == {
        "model": milvus_client.EMBEDDING_MODEL,
        "input": ["迟到处罚", "报销流程"],
        "dimensions": milvus_client.EMBEDDING_DIM,
    }
    assert call["headers"]["Authorization"] == "Bearer test-embedding-key"
    # Follow the configured timeout instead of a literal: the client must use whatever
    # EMBEDDING_TIMEOUT_SECONDS currently holds, so the deployed value can be tuned without
    # this contract test going stale (and a hard-coded value elsewhere still fails here).
    assert call["timeout"] == milvus_client.EMBEDDING_TIMEOUT_SECONDS
    assert caplog.text == ""


def test_embedding_without_credentials_never_opens_a_connection(monkeypatch):
    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("HTTP client must not be created without credentials")

    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(milvus_client.httpx, "Client", explode)

    vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["迟到处罚"])

    assert len(vectors) == 1
    assert len(vectors[0]) == milvus_client.EMBEDDING_DIM


def test_embedding_timeout_raises_instead_of_falling_back(monkeypatch, caplog):
    def behaviour(_count):
        raise httpx.ConnectTimeout("embedding endpoint timed out")

    _install_embedding_client(monkeypatch, behaviour)
    monkeypatch.setattr(milvus_client, "_hash_embedding_fn", _forbidden_hash_fallback)

    with caplog.at_level(logging.ERROR, logger=milvus_client.__name__):
        with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
            milvus_client._OpenAICompatibleEmbeddingFunction()(["迟到处罚"])

    message = str(exc_info.value)
    assert "向量化接口调用失败" in message
    assert "embedding endpoint timed out" in message
    # The failure must reach the operator, not just the caller.
    assert "向量化接口调用失败" in caplog.text
    assert "embedding endpoint timed out" in caplog.text


@pytest.mark.parametrize(
    "payload, detail",
    [
        ({"data": []}, "期望 2 条向量，实际得到 0 条"),
        ({"data": [{"index": 0, "embedding": []}]}, "期望 2 条向量，实际得到 1 条"),
        ({"data": [{"index": 0, "embedding": [0.1] * milvus_client.EMBEDDING_DIM}]}, "期望 2 条向量，实际得到 1 条"),
        (
            {
                "data": [
                    {"index": 0, "embedding": [0.1] * milvus_client.EMBEDDING_DIM},
                    {"index": 1, "embedding": [0.1]},
                ]
            },
            f"期望 {milvus_client.EMBEDDING_DIM} 维向量，实际得到 {[milvus_client.EMBEDDING_DIM, 1]}",
        ),
    ],
    ids=["empty-data", "empty-vector", "count-mismatch", "dimension-mismatch"],
)
def test_embedding_malformed_response_raises_instead_of_falling_back(monkeypatch, caplog, payload, detail):
    _install_embedding_client(monkeypatch, lambda _count: _Response(payload))
    monkeypatch.setattr(milvus_client, "_hash_embedding_fn", _forbidden_hash_fallback)

    with caplog.at_level(logging.ERROR, logger=milvus_client.__name__):
        with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
            milvus_client._OpenAICompatibleEmbeddingFunction()(["迟到处罚", "报销流程"])

    message = str(exc_info.value)
    assert "向量化接口响应结构异常" in message
    assert detail in message
    assert "向量化接口响应结构异常" in caplog.text


def test_rerank_request_contract_clamps_and_sorts_scores(monkeypatch):
    def behaviour(request):
        assert len(request["documents"]) == 2
        return _Response(
            {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 1.7},
                        {"index": 0, "relevance_score": -0.3},
                    ]
                }
            }
        )

    calls = _install_rerank_client(monkeypatch, behaviour)

    ranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么处罚", [dict(chunk) for chunk in CHUNKS]))

    assert [chunk["chunk_id"] for chunk in ranked] == ["1", "0"]
    assert [chunk["rerank_score"] for chunk in ranked] == [1.0, 0.0]
    assert trace["status"] == "done"
    assert trace["provider"] == rerank.RERANK_PROVIDER
    assert trace["model"] == rerank.RERANK_MODEL
    assert [item["chunk_id"] for item in trace["items"]] == ["1", "0"]

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == RERANK_BASE_URL
    assert call["headers"]["Authorization"] == "Bearer test-rerank-key"
    assert call["timeout"] == rerank.RERANK_TIMEOUT_SECONDS
    assert call["json"]["model"] == rerank.RERANK_MODEL
    assert call["json"]["query"] == "迟到怎么处罚"
    assert call["json"]["documents"] == [chunk["content"] for chunk in CHUNKS]
    assert call["json"]["top_n"] == min(rerank.RETRIEVAL_RERANK_TOP_N, len(CHUNKS))


def test_rerank_timeout_reports_failed_trace(monkeypatch, caplog):
    def behaviour(_request):
        raise httpx.TimeoutException("rerank endpoint timed out")

    calls = _install_rerank_client(monkeypatch, behaviour)
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", False)

    with caplog.at_level(logging.WARNING, logger=rerank.__name__):
        ranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么处罚", [dict(chunk) for chunk in CHUNKS]))

    assert ranked == []
    assert trace["status"] == "failed"
    assert trace["provider"] == rerank.RERANK_PROVIDER
    assert trace["model"] == rerank.RERANK_MODEL
    # issue #263: the trace is persisted and returned to clients, so it carries only the
    # fixed message -- never the provider's raw exception text.
    assert trace["error"] == rerank.RERANK_FAILED_MESSAGE
    assert "rerank endpoint timed out" not in trace["error"]
    assert len(calls) == 1
    # Diagnosability is not lost: the original exception still reaches the server log.
    assert "rerank endpoint timed out" in caplog.text
    assert "scope=rerank_primary" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {"output": {"results": []}},
        {"output": {"results": [{"index": 99, "relevance_score": 0.5}]}},
        {"output": {"results": [{"index": "bad", "relevance_score": 0.5}]}},
    ],
    ids=["no-results", "index-out-of-range", "non-numeric-index"],
)
def test_rerank_malformed_response_reports_failed_trace(monkeypatch, payload):
    _install_rerank_client(monkeypatch, lambda _request: _Response(payload))
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", False)

    ranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么处罚", [dict(chunk) for chunk in CHUNKS]))

    assert ranked == []
    assert trace["status"] == "failed"
    assert trace["error"]


def test_rerank_without_api_key_reports_failed_trace(monkeypatch, caplog):
    def explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("HTTP client must not be created without credentials")

    monkeypatch.setattr(rerank, "RERANK_API_KEY", "")
    monkeypatch.setattr(rerank.httpx, "AsyncClient", explode)
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", False)

    with caplog.at_level(logging.WARNING, logger=rerank.__name__):
        ranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么处罚", [dict(chunk) for chunk in CHUNKS]))

    assert ranked == []
    assert trace["status"] == "failed"
    # issue #263: a missing credential is configuration detail, not client-facing trace text.
    assert trace["error"] == rerank.RERANK_FAILED_MESSAGE
    assert "RERANK_API_KEY" not in trace["error"]
    # ...but the operator still learns which configuration is missing from the server log.
    assert "RERANK_API_KEY or DASHSCOPE_API_KEY is not configured" in caplog.text
    assert "scope=rerank_primary" in caplog.text


def _load_smoke_module(monkeypatch):
    """Import scripts/smoke_providers.py with the environment it rewrites contained.

    Importing the script decides offline mode from sys.argv and overwrites the provider
    credentials and base URLs in os.environ before the config is imported. Registering each
    of those names with monkeypatch first makes that rewrite revert with the test instead of
    leaking into the rest of the session.
    """
    for name in (
        "DEEPSEEK_API_KEY",
        "TEXT_FALLBACK_API_KEY",
        "EMBEDDING_API_KEY",
        "RERANK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "TEXT_FALLBACK_BASE_URL",
        "EMBEDDING_BASE_URL",
        "RERANK_BASE_URL",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setattr(sys, "argv", ["smoke_providers.py"])

    spec = importlib.util.spec_from_file_location("smoke_providers_under_test", ROOT / "scripts" / "smoke_providers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _UnreachableRerankClient:
    """The client shape when the qwen3-rerank endpoint cannot be reached."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        raise httpx.ConnectError("dashscope rerank unreachable")


def _reachable_fallback(*_args, **_kwargs):
    """The LLM fallback answering: the shape that used to make `rerank` report a pass."""

    async def _call():
        return {"results": [{"id": 2, "score": 0.8, "reason": "fallback ok"}]}

    return _call()


def test_smoke_rerank_check_rejects_a_fallback_served_request(monkeypatch, caplog):
    # issue #148: the LLM fallback returns status="done" too, so `rerank` used to report a
    # pass while the qwen3-rerank endpoint never answered -- an unreachable endpoint, an
    # expired key and a bad model name were all indistinguishable from a verified rerank.
    smoke = _load_smoke_module(monkeypatch)
    monkeypatch.setattr(smoke.config, "RERANK_API_KEY", "test-rerank-key")
    monkeypatch.setattr(rerank, "RERANK_API_KEY", "test-rerank-key")
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(rerank.httpx, "AsyncClient", _UnreachableRerankClient)
    monkeypatch.setattr(rerank, "call_chat_json", _reachable_fallback)

    with caplog.at_level(logging.WARNING, logger=rerank.__name__):
        with pytest.raises(AssertionError) as exc_info:
            smoke.check_rerank(live=True)

    # The failure has to carry enough to tell an operator which provider answered and why
    # the primary one did not.
    message = str(exc_info.value)
    assert "status='done'" in message
    assert "provider='deepseek_fallback'" in message
    # issue #263: the trace's fallback_reason is the fixed message, so the smoke failure
    # message carries that contract rather than the raw exception text.
    assert f"fallback='{rerank.RERANK_FALLBACK_MESSAGE}'" in message
    assert "dashscope rerank unreachable" not in message
    # The original exception is still visible in the server log.
    assert "dashscope rerank unreachable" in caplog.text
    assert "scope=rerank_fallback_success" in caplog.text


def test_smoke_rerank_fallback_check_rejects_a_primary_served_request(monkeypatch):
    # The mirror invariant: `rerank-fallback` drives the LLM fallback directly, so a trace
    # claiming the primary provider means the fallback was never verified.
    smoke = _load_smoke_module(monkeypatch)

    async def _primary_provider_trace(_question, _chunks):
        return [{"file_id": 1, "chunk_id": "1", "rerank_score": 1.0}], {
            "status": "done",
            "provider": rerank.RERANK_PROVIDER,
            "model": rerank.RERANK_MODEL,
            "items": [],
        }

    monkeypatch.setattr(rerank, "_rerank_chunks_with_llm", _primary_provider_trace)

    with pytest.raises(AssertionError) as exc_info:
        smoke.check_rerank_fallback(live=False)

    assert f"provider='{rerank.RERANK_PROVIDER}'" in str(exc_info.value)


class _RecordingChatModel:
    """Stand-in for ChatOpenAI that records its construction kwargs."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)

    async def ainvoke(self, messages):
        raise RuntimeError("offline stub: chat endpoint unreachable")

    async def astream(self, messages):
        # An async generator on purpose: the chain iterates the stream, so raising from
        # a plain coroutine would surface as a TypeError instead of the provider error.
        raise RuntimeError("offline stub: chat stream unreachable")
        yield  # pragma: no cover - unreachable, only marks this as a generator


class _StreamingChatModel(_RecordingChatModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_primary = kwargs.get("model") == llm.normalize_deepseek_model("deepseek-test-primary")

    async def astream(self, messages):
        if self.is_primary:
            raise RuntimeError("offline stub: DeepSeek stream unreachable")
        for part in ("后备", "模型", "回答"):
            yield _Message(part)


class _Message:
    def __init__(self, content):
        self.content = content


def test_deepseek_model_configuration_contract(monkeypatch):
    _RecordingChatModel.instances = []
    monkeypatch.setattr(llm, "ChatOpenAI", _RecordingChatModel)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.test/")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseekv4pro")

    model = llm.get_deepseek_model(streaming=True, temperature=0.2, max_tokens=256)

    assert model.kwargs == {
        "api_key": "test-deepseek-key",
        "base_url": "https://api.deepseek.test/v1",
        "model": "deepseek-v4-pro",
        "temperature": 0.2,
        "streaming": True,
        "timeout": 60,
        "max_retries": 0,
        "max_completion_tokens": 256,
    }


def test_deepseek_model_without_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")

    with pytest.raises(RuntimeError, match="DeepSeek API Key 未配置"):
        llm.get_deepseek_model()


def test_text_fallback_model_without_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "")

    with pytest.raises(RuntimeError, match="文本后备模型 API Key 未配置"):
        llm.get_text_fallback_model()


def _collect_events(**kwargs):
    async def collect():
        return [event async for event in llm.stream_answer_events(**kwargs)]

    return asyncio.run(collect())


def test_answer_stream_reports_error_when_fallback_disabled(monkeypatch):
    trace = _Recorder()
    monkeypatch.setattr(llm, "ChatOpenAI", _RecordingChatModel)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)

    events = _collect_events(question="迟到怎么处罚", context="迟到罚款50元", trace=trace)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "TEXT_FALLBACK_ENABLED" in events[0]["message"]
    assert [event["event"] for event in trace.events] == [
        "langchain_generation_prompt_built",
        "langchain_generation_failed",
    ]


def test_answer_stream_reports_error_when_fallback_unconfigured(monkeypatch):
    monkeypatch.setattr(llm, "ChatOpenAI", _RecordingChatModel)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "")

    events = _collect_events(question="迟到怎么处罚", context="迟到罚款50元")

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "TEXT_FALLBACK_API_KEY" in events[0]["message"]


def test_answer_stream_switches_to_text_fallback_model(monkeypatch):
    trace = _Recorder()
    monkeypatch.setattr(llm, "ChatOpenAI", _StreamingChatModel)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-test-primary")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "test-fallback-key")

    events = _collect_events(question="迟到怎么处罚", context="迟到罚款50元", trace=trace)

    # The fallback model regenerates the whole answer, so the consumer has to be told to
    # discard whatever the failed stream already emitted before the new text arrives.
    assert len(events) >= 2
    reset, chunks = events[0], events[1:]
    assert isinstance(reset, dict)
    assert reset["type"] == "reset"
    assert reset["reason"] == "text_fallback"
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunks) == "后备模型回答"
    assert [event["event"] for event in trace.events] == [
        "langchain_generation_prompt_built",
        "langchain_generation_failed",
        "langchain_stream_reset",
        "langchain_text_fallback_started",
    ]


class _Recorder:
    def __init__(self):
        self.events = []

    def add(self, event, *args, **kwargs):
        self.events.append({"event": event, "args": args, **kwargs})


def test_smoke_script_runs_offline_and_reports_every_provider():
    env = {key: value for key, value in os.environ.items() if not key.endswith("API_KEY")}

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_providers.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LIVE MODE" not in result.stdout
    for name in ("deepseek", "embedding", "rerank", "rerank-fallback", "text-fallback"):
        assert f"[PASS] {name}" in result.stdout
    assert "5 checks: 5 passed, 0 skipped, 0 failed" in result.stdout


def test_smoke_script_live_without_credentials_reports_skips_not_passes():
    # Empty values, not just removed variables: config.py loads .env/.env.development
    # through dotenv, which does not overwrite variables that are already set, so an
    # empty value keeps a developer's real credentials out of the child process. Every
    # live check then skips before it can open a socket.
    env = {key: value for key, value in os.environ.items() if not key.endswith("API_KEY")}
    env.update(
        {
            "DEEPSEEK_API_KEY": "",
            "TEXT_FALLBACK_API_KEY": "",
            "EMBEDDING_API_KEY": "",
            "RERANK_API_KEY": "",
            "DASHSCOPE_API_KEY": "",
        }
    )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_providers.py"), "--live"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LIVE MODE" in result.stdout
    for name in ("deepseek", "embedding", "rerank", "rerank-fallback", "text-fallback"):
        assert f"[SKIP] {name}" in result.stdout
    assert "5 checks: 0 passed, 5 skipped, 0 failed" in result.stdout
    assert "no check executed" in result.stdout
    assert "checks passed" not in result.stdout


@pytest.mark.parametrize("abbreviation", ["--li", "--l"])
def test_smoke_script_rejects_an_abbreviated_live_flag(abbreviation):
    # issue #149: argparse used to accept `--li` as `--live` while the module-level scan that
    # prepares the environment matched the flag literally, so the run printed the LIVE banner
    # on top of the offline stub credentials and reported five provider failures that never
    # touched a provider. The entry point must reject the spelling instead of running.
    env = {key: value for key, value in os.environ.items() if not key.endswith("API_KEY")}

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_providers.py"), abbreviation],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    # argparse exits 2 on an argument it does not know; both readings of the flag agree
    # because neither of them sees a live run.
    assert result.returncode == 2, result.stdout + result.stderr
    assert abbreviation in result.stderr
    assert "LIVE MODE" not in result.stdout
    # Nothing may run at all: a check line would mean the offline stub environment decided
    # the run while the banner said otherwise.
    for marker in ("[PASS]", "[FAIL]", "[SKIP]", "offline-stub.invalid", "checks:"):
        assert marker not in result.stdout
