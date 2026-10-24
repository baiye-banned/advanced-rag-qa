#!/usr/bin/env python
"""Smoke-check the model providers the RAG chain depends on.

The script is offline by default: every provider call is served by an in-process stub,
so it exercises the real client code paths -- URL building, request payload, response
parsing, error handling and fallbacks -- without opening a socket. Offline mode also
overwrites the provider credentials with a dummy value before importing the config, so
a half-stubbed call can never carry a real key to the network.

Pass ``--live`` to talk to the real endpoints. That requires configured credentials,
consumes quota and reaches the internet, so it is never the default and prints a banner.
The flag is matched literally and abbreviations are rejected (``--li`` exits 2): the
module-level scan below reads ``sys.argv`` before argparse runs, so the two have to accept
exactly the same spellings, or a run would announce live requests while its credentials and
base URLs had already been rewritten to the offline stub.

Checks: deepseek, embedding, rerank, rerank-fallback, text-fallback.

The summary always reports passed, skipped and failed separately, and a skipped check is
never counted as a pass: a `--live` run without credentials prints `5 checks: 0 passed,
5 skipped, 0 failed` plus a "no check executed" line instead of claiming that 5/5 checks
passed. Exit code is 0 unless a check failed (a skip is an unverified check, not a
failure), so read the summary rather than the exit code to see what actually ran.

A check only passes on the provider it names: `rerank` counts as verified only when the
qwen3-rerank endpoint answered, so a successful LLM fallback is reported as a failure
instead of hiding an unreachable DashScope endpoint behind a pass.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

OFFLINE_KEY = "offline-smoke-key"
OFFLINE_BASE_URL = "http://offline-stub.invalid/v1"
CHUNKS = [
    {"file_id": 1, "chunk_id": "0", "content": "员工迟到一次罚款50元。", "file_name": "考勤制度.txt"},
    {"file_id": 1, "chunk_id": "1", "content": "报销需要在30天内提交。", "file_name": "报销制度.txt"},
]
QUESTION = "迟到怎么处罚"


class Skip(Exception):
    """A check that cannot run in the current mode, e.g. live mode without credentials."""


def _prepare_offline_environment() -> None:
    for name in ("DEEPSEEK_API_KEY", "TEXT_FALLBACK_API_KEY", "EMBEDDING_API_KEY", "RERANK_API_KEY"):
        os.environ[name] = OFFLINE_KEY
    os.environ["DEEPSEEK_BASE_URL"] = OFFLINE_BASE_URL
    os.environ["TEXT_FALLBACK_BASE_URL"] = OFFLINE_BASE_URL
    os.environ["EMBEDDING_BASE_URL"] = OFFLINE_BASE_URL
    os.environ["RERANK_BASE_URL"] = f"{OFFLINE_BASE_URL}/reranks"


LIVE = "--live" in sys.argv[1:]
if not LIVE:
    _prepare_offline_environment()

import config  # noqa: E402  (import order is deliberate: env must be set first)
from rag import llm, milvus_client, rerank  # noqa: E402


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubSyncClient:
    """httpx.Client stand-in returning a well-formed embedding response."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        texts = list((json or {}).get("input") or [])
        return _Response(
            {
                "data": [
                    {"index": index, "embedding": [0.01] * milvus_client.EMBEDDING_DIM}
                    for index in range(len(texts))
                ]
            }
        )


class _StubAsyncClient:
    """httpx.AsyncClient stand-in returning a well-formed qwen3-rerank response."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):
        documents = list((json or {}).get("documents") or [])
        return _Response(
            {
                "output": {
                    "results": [
                        {"index": index, "relevance_score": 1.0 - index * 0.1}
                        for index in range(len(documents))
                    ]
                }
            }
        )


class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubChatModel:
    """ChatOpenAI stand-in; the DeepSeek slot fails so fallbacks get exercised."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_primary = kwargs.get("model") == llm.normalize_deepseek_model(config.DEEPSEEK_MODEL)

    async def ainvoke(self, messages):
        if self.is_primary:
            raise RuntimeError("offline stub: DeepSeek chat endpoint unreachable")
        return _StubMessage('{"results": [{"id": 1, "score": 0.9, "reason": "stub"}]}')

    async def astream(self, messages):
        if self.is_primary:
            raise RuntimeError("offline stub: DeepSeek stream unreachable")
        for part in ("offline ", "fallback ", "answer"):
            yield _StubMessage(part)


def check_deepseek(live: bool) -> str:
    if live:
        if not config.DEEPSEEK_API_KEY:
            raise Skip("DEEPSEEK_API_KEY is not configured")
        model = llm.get_deepseek_model(streaming=False, temperature=0.1, max_tokens=16)
        reply = asyncio.run(model.ainvoke([("user", "ping")]))
        text = llm._message_text(reply.content).strip()
        if not text:
            raise AssertionError("DeepSeek returned an empty reply")
        return f"model={model.model_name} reply={text[:40]!r}"

    model = llm.get_deepseek_model(streaming=False, temperature=0.1, max_tokens=16)
    expected_url = llm.openai_base_url(config.DEEPSEEK_BASE_URL)
    if model.max_retries != 0:
        raise AssertionError(f"max_retries={model.max_retries}, expected 0 (no silent retries)")
    if model.openai_api_base != expected_url:
        raise AssertionError(f"base_url={model.openai_api_base!r}, expected {expected_url!r}")
    if getattr(model, "request_timeout", None) != 60:
        raise AssertionError(f"timeout={getattr(model, 'request_timeout', None)!r}, expected 60")
    key_configured = bool(llm.DEEPSEEK_API_KEY)
    original = llm.DEEPSEEK_API_KEY
    try:
        llm.DEEPSEEK_API_KEY = ""
        try:
            llm.get_deepseek_model()
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing API key did not raise")
    finally:
        llm.DEEPSEEK_API_KEY = original  # scan-secrets:allow restores the key blanked for the missing-key check, not a credential
    return f"model={model.model_name} base_url={expected_url} key_configured={key_configured}"


def check_embedding(live: bool) -> str:
    if live:
        if not config.EMBEDDING_API_KEY:
            raise Skip("EMBEDDING_API_KEY / DASHSCOPE_API_KEY is not configured")
        vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["离线冒烟", "online smoke"])
    else:
        original = milvus_client.httpx.Client
        milvus_client.httpx.Client = _StubSyncClient
        try:
            vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["离线冒烟", "online smoke"])
        finally:
            milvus_client.httpx.Client = original
    if len(vectors) != 2:
        raise AssertionError(f"expected 2 vectors, got {len(vectors)}")
    if any(len(vector) != milvus_client.EMBEDDING_DIM for vector in vectors):
        raise AssertionError(f"expected dim {milvus_client.EMBEDDING_DIM}")
    mode = "live" if live else "stubbed transport"
    return f"model={config.EMBEDDING_MODEL} dim={milvus_client.EMBEDDING_DIM} ({mode})"


def check_rerank(live: bool) -> str:
    if live and not config.RERANK_API_KEY:
        raise Skip("RERANK_API_KEY / DASHSCOPE_API_KEY is not configured")
    original = rerank.httpx.AsyncClient
    if not live:
        rerank.httpx.AsyncClient = _StubAsyncClient
    try:
        ranked, trace = asyncio.run(rerank.rerank_chunks(QUESTION, [dict(chunk) for chunk in CHUNKS]))
    finally:
        rerank.httpx.AsyncClient = original
    # `status == "done"` alone does not mean the qwen3-rerank endpoint answered: when it
    # fails and RERANK_LLM_FALLBACK_ENABLED is on (the default), rerank_chunks() retries
    # through the LLM and returns a done trace as well, so an unreachable endpoint, an
    # expired key or a bad model name would all be reported as a passing rerank.
    if trace.get("status") != "done" or trace.get("provider") != rerank.RERANK_PROVIDER:
        raise AssertionError(
            f"rerank status={trace.get('status')!r} provider={trace.get('provider')!r} "
            f"error={trace.get('error')!r} fallback={trace.get('fallback_reason')!r}"
        )
    if not ranked:
        raise AssertionError("rerank returned no chunk")
    scores = [chunk.get("rerank_score") for chunk in ranked]
    if scores != sorted(scores, reverse=True):
        raise AssertionError(f"scores not sorted: {scores}")
    return f"model={trace.get('model')} ranked={len(ranked)} scores={scores}"


def check_rerank_fallback(live: bool) -> str:
    if live:
        if not config.DEEPSEEK_API_KEY:
            raise Skip("DEEPSEEK_API_KEY is not configured")
        ranked, trace = asyncio.run(
            rerank._rerank_chunks_with_llm(QUESTION, [dict(chunk) for chunk in CHUNKS])
        )
    else:
        original_chat = rerank.call_chat_json

        async def stub_call_chat_json(system_prompt, user_prompt, **kwargs):
            return {"results": [{"id": 2, "score": 0.8, "reason": "stub"}]}

        rerank.call_chat_json = stub_call_chat_json
        try:
            ranked, trace = asyncio.run(
                rerank._rerank_chunks_with_llm(QUESTION, [dict(chunk) for chunk in CHUNKS])
            )
        finally:
            rerank.call_chat_json = original_chat
    # The mirror of the check above: this one drives the LLM fallback directly, so a trace
    # claiming the primary provider did not verify the fallback either. Without it a
    # `rerank-fallback` served by qwen3-rerank would pass and the two checks would report
    # the same thing.
    if trace.get("status") != "done" or trace.get("provider") == rerank.RERANK_PROVIDER:
        raise AssertionError(
            f"fallback rerank status={trace.get('status')!r} provider={trace.get('provider')!r} "
            f"error={trace.get('error')!r}"
        )
    if not ranked:
        raise AssertionError("fallback rerank returned no chunk")
    return f"provider={trace.get('provider')} ranked={len(ranked)} first={ranked[0].get('chunk_id')}"


def check_text_fallback(live: bool) -> str:
    if live:
        if not config.TEXT_FALLBACK_API_KEY:
            raise Skip("TEXT_FALLBACK_API_KEY / DASHSCOPE_API_KEY is not configured")
        model = llm.get_text_fallback_model(streaming=True, temperature=0.1, max_tokens=16)
    else:
        original = llm.ChatOpenAI
        llm.ChatOpenAI = _StubChatModel
        try:
            model = llm.get_text_fallback_model(streaming=True, temperature=0.1, max_tokens=16)
        finally:
            llm.ChatOpenAI = original

    async def collect() -> str:
        parts = []
        async for chunk in llm._stream_model_chunks(model, [("user", QUESTION)]):
            parts.append(llm._chunk_text(chunk))
        return "".join(parts)

    text = asyncio.run(collect())
    if not text.strip():
        raise AssertionError("text fallback model produced no content")
    return f"model={config.TEXT_FALLBACK_MODEL} chars={len(text)}"


CHECKS = [
    ("deepseek", check_deepseek),
    ("embedding", check_embedding),
    ("rerank", check_rerank),
    ("rerank-fallback", check_rerank_fallback),
    ("text-fallback", check_text_fallback),
]


def main() -> int:
    # allow_abbrev=False keeps this parser and the module-level `LIVE` scan above accepting
    # the same spellings: an abbreviation such as `--li` must fail here (exit 2) rather than
    # be accepted, because the environment was already prepared for the offline stub.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--live", action="store_true", help="send real requests to the providers")
    args = parser.parse_args()

    if args.live:
        print("!! LIVE MODE: real provider requests, network access and quota usage !!\n")
    else:
        print("offline mode (in-process stubs, no network); use --live for real requests\n")

    results = []
    for name, check in CHECKS:
        try:
            detail = check(args.live)
        except Skip as exc:
            results.append((name, "SKIP", str(exc)))
        except Exception as exc:  # noqa: BLE001 - a smoke check reports any failure
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        else:
            results.append((name, "PASS", detail))

    width = max(len(name) for name, _ in CHECKS)
    for name, status, detail in results:
        print(f"[{status}] {name.ljust(width)}  {detail}")

    passed = [name for name, status, _ in results if status == "PASS"]
    skipped = [name for name, status, _ in results if status == "SKIP"]
    failed = [name for name, status, _ in results if status == "FAIL"]

    print(f"\n{len(results)} checks: {len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed")
    if skipped:
        print(f"skipped, nothing verified: {', '.join(skipped)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    if not passed:
        print("no check executed: this run verified nothing against the providers")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
