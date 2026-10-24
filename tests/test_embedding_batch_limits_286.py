"""issue #286：入库批次必须按模型能力（条数 + token 预算）打包。

这些用例只使用本地 fake embedding / fake httpx，不访问真实向量化接口。
"""

import logging

import httpx
import pytest

from rag import milvus_client


def _token_count(text: str) -> int:
    return milvus_client._estimate_tokens(text)


def _patch_batching(
    monkeypatch,
    *,
    item_limit=10,
    max_tokens=8192,
    ratio=0.85,
    legacy_chars=0,
):
    """把批次配置固定到用例期望值，不依赖开发机上的 .env。"""
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_BATCH_ITEMS", item_limit)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_SIZE", item_limit)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_MAX_CHARS", legacy_chars)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_REQUEST_TOKENS", max_tokens)
    monkeypatch.setattr(milvus_client, "EMBEDDING_TOKEN_SAFETY_RATIO", ratio)


def _success_payload(count):
    return {
        "data": [
            {"index": index, "embedding": [0.1] * milvus_client.EMBEDDING_DIM}
            for index in range(count)
        ]
    }


class _FakeResponse:
    """最小响应替身：状态码、Retry-After 与 JSON 体都由用例指定。"""

    def __init__(self, status_code, *, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://embedding.example/v1/embeddings")
            raise httpx.HTTPStatusError(
                f"{self.status_code} error",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self):
        return self._payload


def _patch_embedding_http(monkeypatch, responses):
    """按调用顺序返回给定响应，并记录每次请求的输入条数。"""
    state = {"index": 0, "requests": []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, url, json=None, headers=None):
            state["requests"].append(list(json["input"]))
            index = min(state["index"], len(responses) - 1)
            state["index"] += 1
            return responses[index]

    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(milvus_client.httpx, "Client", FakeClient)
    return state


def _patch_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(milvus_client.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


@pytest.mark.parametrize(
    ("count", "expected_sizes"),
    [
        (9, [9]),
        (10, [10]),
        (11, [10, 1]),
    ],
)
def test_item_cap_boundaries(monkeypatch, count, expected_sizes):
    """9/10/11 条边界：条数上限为 10 时批次划分正确，且不丢切片、不改顺序。"""
    _patch_batching(monkeypatch)
    texts = [f"第{index}条制度" for index in range(count)]

    batches = list(milvus_client._embedding_batches(texts))

    assert [len(batch) for batch in batches] == expected_sizes
    assert [text for batch in batches for text in batch] == texts


def test_token_budget_holds_for_chinese_english_and_mixed(monkeypatch):
    """中文/英文/混排都按累计 token 打包，任何一批都不超过预算。"""
    max_tokens = 100
    ratio = 0.5
    budget = int(max_tokens * ratio)
    _patch_batching(monkeypatch, item_limit=10**6, max_tokens=max_tokens, ratio=ratio)

    texts = [
        "合规审查制度" * 2,
        "compliance and audit review " * 2,
        "混合 mixed 文本 text " * 2,
        "员工考勤管理办法" * 2,
        "leave request approval " * 2,
    ] * 2
    # 前提：每条自身都在预算内，否则用例会走「单条超预算」分支。
    assert max(_token_count(text) for text in texts) <= budget
    assert sum(_token_count(text) for text in texts) > budget

    batches = list(milvus_client._embedding_batches(texts))

    assert [text for batch in batches for text in batch] == texts
    assert all(
        sum(_token_count(text) for text in batch) <= budget for batch in batches
    )
    assert len(batches) > 1


def test_oversized_single_item_fails_before_any_request(monkeypatch):
    """单条自身超过 token 预算：明确失败并指出第几条/多少 token，不发出任何请求。"""
    calls = []
    _patch_batching(monkeypatch, max_tokens=10, ratio=0.8)
    oversized = "这是一段很长的制度文本" * 20
    oversized_tokens = _token_count(oversized)
    assert oversized_tokens > int(10 * 0.8)

    def recording_embedding(texts):
        calls.append(list(texts))
        return [[0.1] * milvus_client.EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(milvus_client, "_embedding_fn", recording_embedding)

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._embed_documents(["正常切片", oversized, "另一条正常切片"])

    assert calls == []
    message = str(exc_info.value)
    assert "#2" in message
    assert str(oversized_tokens) in message
    assert "token 预算" in message


def test_429_is_retried_then_succeeds(monkeypatch, caplog):
    """429 后重试成功，并记录 batch/条数/估算 token/尝试次数/耗时。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    state = _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(429, headers={"Retry-After": "0"}),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    sleeps = _patch_sleep(monkeypatch)

    with caplog.at_level(logging.INFO, logger="rag.milvus_client"):
        vectors = milvus_client._embed_documents(["hello"])

    assert len(vectors) == 1
    assert len(state["requests"]) == 2
    assert sleeps == [milvus_client._EMBEDDING_RETRY_BASE_DELAY_SECONDS]

    messages = [record.getMessage() for record in caplog.records]
    retry = next(message for message in messages if "batch retry" in message)
    completed = next(message for message in messages if "batch completed" in message)
    assert "batch=1" in retry
    assert "attempt=1" in retry
    assert "batch=1" in completed
    assert "items=1" in completed
    assert "estimated_tokens=" in completed
    assert "attempts=2" in completed
    assert "elapsed_ms=" in completed


@pytest.mark.parametrize("status", [500, 503])
def test_5xx_is_retried_then_succeeds(monkeypatch, status):
    """5xx 与 429 一样属于瞬态错误。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    state = _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(status),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    sleeps = _patch_sleep(monkeypatch)

    vectors = milvus_client._embed_documents(["hello"])

    assert len(vectors) == 1
    assert len(state["requests"]) == 2
    assert sleeps == [milvus_client._EMBEDDING_RETRY_BASE_DELAY_SECONDS]


def test_retries_stop_at_configured_limit(monkeypatch):
    """重试次数达到 EMBEDDING_MAX_RETRIES 后如实失败，不无限重试。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    state = _patch_embedding_http(
        monkeypatch,
        [_FakeResponse(429) for _ in range(10)],
    )
    sleeps = _patch_sleep(monkeypatch)

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._embed_documents(["hello"])

    assert "429" in str(exc_info.value)
    # 首次请求 + 2 次重试。
    assert len(state["requests"]) == 3
    assert sleeps == [
        milvus_client._EMBEDDING_RETRY_BASE_DELAY_SECONDS,
        milvus_client._EMBEDDING_RETRY_BASE_DELAY_SECONDS * 2,
    ]
    assert milvus_client.embedding_backend_status()["mode"] == "unavailable"


def test_400_is_not_retried(monkeypatch):
    """400 是永久错误：只发一次请求，不做退避。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    state = _patch_embedding_http(monkeypatch, [_FakeResponse(400)])
    sleeps = _patch_sleep(monkeypatch)

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._embed_documents(["hello"])

    assert "400" in str(exc_info.value)
    assert len(state["requests"]) == 1
    assert sleeps == []


def test_retry_after_overrides_exponential_backoff(monkeypatch):
    """上游给了 Retry-After 时按它等待，而不是只按指数退避。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(429, headers={"Retry-After": "3"}),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    sleeps = _patch_sleep(monkeypatch)

    milvus_client._embed_documents(["hello"])

    assert sleeps == [3.0]


def test_each_batch_retries_independently_and_logs_its_index(monkeypatch, caplog):
    """每批独立计数尝试次数；日志中的 batch 序号与实际批次一致。"""
    _patch_batching(monkeypatch, item_limit=1)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(200, payload=_success_payload(1)),
            _FakeResponse(429),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    _patch_sleep(monkeypatch)

    with caplog.at_level(logging.INFO, logger="rag.milvus_client"):
        vectors = milvus_client._embed_documents(["第一段", "第二段"])

    assert len(vectors) == 2
    completed = [
        record.getMessage()
        for record in caplog.records
        if "batch completed" in record.getMessage()
    ]
    assert [message.split("batch=")[1].split(" ")[0] for message in completed] == ["1", "2"]
    assert "attempts=1" in completed[0]
    assert "attempts=2" in completed[1]
