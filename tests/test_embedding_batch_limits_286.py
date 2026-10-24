"""issue #286：入库批次必须按模型能力（条数 + token 预算）打包。

这些用例只使用本地 fake embedding / fake httpx，不访问真实向量化接口。
"""

import logging
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

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
    legacy_items=None,
):
    """把批次配置固定到用例期望值，不依赖开发机上的 .env。"""
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_BATCH_ITEMS", item_limit)
    monkeypatch.setattr(
        milvus_client,
        "EMBEDDING_INGEST_BATCH_SIZE",
        item_limit if legacy_items is None else legacy_items,
    )
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_MAX_CHARS", legacy_chars)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_REQUEST_TOKENS", max_tokens)
    monkeypatch.setattr(milvus_client, "EMBEDDING_TOKEN_SAFETY_RATIO", ratio)
    # 退避上限依赖整份文档预算；固定它，避免开发机 .env 改变期望的 sleep 值。
    monkeypatch.setattr(milvus_client, "KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS", 300)


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


def _patch_clock(monkeypatch, now=1000.0):
    """假单调时钟：sleep 只推进假时钟，测试永远不会真的等待。"""
    clock = {"now": now}
    sleeps = []

    monkeypatch.setattr(milvus_client, "_monotonic_now", lambda: clock["now"])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(milvus_client.time, "sleep", fake_sleep)
    return clock, sleeps


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

    assert [len(batch) for batch, _tokens in batches] == expected_sizes
    assert [text for batch, _tokens in batches for text in batch] == texts


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

    assert [text for batch, _tokens in batches for text in batch] == texts
    assert all(tokens <= budget for _batch, tokens in batches)
    assert len(batches) > 1


def test_batches_carry_token_counts_without_recomputing(monkeypatch):
    """每批产出 (batch, batch_tokens)：token 只估算一次，日志直接复用该值。"""
    _patch_batching(monkeypatch, item_limit=2)

    texts = ["合规审查制度", "compliance review", "混合 mixed 文本", "员工考勤"]
    batches = list(milvus_client._embedding_batches(texts))

    assert [text for batch, _tokens in batches for text in batch] == texts
    for batch, tokens in batches:
        assert tokens == sum(_token_count(text) for text in batch)


def test_oversized_single_item_fails_before_any_request(monkeypatch):
    """单条自身超过单条安全预算：明确失败并指出第几条/多少 token，不发出任何请求。"""
    calls = []
    _patch_batching(monkeypatch, max_tokens=10, ratio=0.8)
    oversized = "这是一段很长的制度文本" * 20
    oversized_tokens = _token_count(oversized)
    assert oversized_tokens > 10

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
    assert "单条安全预算" in message
    # minor-3：单条判断用真正的单输入上限，安全系数只用于批打包。
    assert "EMBEDDING_MAX_REQUEST_TOKENS=10" in message
    assert str(int(10 * 0.8)) not in message


def test_single_item_between_packing_budget_and_model_limit_still_sends(monkeypatch):
    """单条超过「安全系数后的合批预算」但未超过模型单条上限时，独占一批发出去。"""
    max_tokens = 100
    _patch_batching(monkeypatch, item_limit=10, max_tokens=max_tokens, ratio=0.5)
    budget = int(max_tokens * 0.5)
    calls = []

    # 构造一条 token 数落在 (budget, max_tokens] 之间的文本。
    text = "word " * 70
    tokens = _token_count(text)
    assert budget < tokens <= max_tokens, (budget, tokens)

    monkeypatch.setattr(
        milvus_client,
        "_embedding_fn",
        lambda batch: (calls.append(list(batch)) or [[0.1] * milvus_client.EMBEDDING_DIM for _ in batch]),
    )

    vectors = milvus_client._embed_documents([text, "短"])

    assert len(vectors) == 2
    assert calls == [[text], ["短"]]


def test_legacy_char_cap_still_splits_batches(monkeypatch):
    """旧字符上限显式配置为正数时仍然生效（先到者约束）。"""
    _patch_batching(monkeypatch, item_limit=10, legacy_chars=10)

    batches = list(milvus_client._embedding_batches(["aaa", "bbb", "ccc", "ddd"]))

    assert [len(batch) for batch, _tokens in batches] == [3, 1]
    assert all(sum(len(text) for text in batch) <= 10 for batch, _tokens in batches)


def test_legacy_item_cap_can_only_tighten_the_model_limit(monkeypatch):
    """旧 64 条配置不能突破模型的 10 条上限：实际按 10 条分批。"""
    _patch_batching(monkeypatch, item_limit=10, legacy_items=64)

    texts = [f"第{index}条" for index in range(25)]
    batches = list(milvus_client._embedding_batches(texts))

    assert [len(batch) for batch, _tokens in batches] == [10, 10, 5]


def test_tokenizer_failure_raises_instead_of_falling_back_to_characters(monkeypatch):
    """encoder 失败时显式报错（含缓存目录/预置文件名），绝不退回字符数估算。"""
    _patch_batching(monkeypatch)
    state = _patch_embedding_http(monkeypatch, [_FakeResponse(200, payload=_success_payload(1))])

    def broken_encoding():
        raise RuntimeError("offline: cannot download BPE")

    monkeypatch.setattr(milvus_client, "_get_token_encoding", broken_encoding)

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._embed_documents(["hello"])

    assert state["requests"] == []
    message = str(exc_info.value)
    assert "token 估算失败" in message
    assert "缓存目录" in message
    assert "cl100k_base.tiktoken" in message


def test_token_safety_ratio_is_clamped_into_range(monkeypatch, caplog):
    """0 或 >1 的安全系数被夹回 (0, 1]：既不会全部失败，也不会关闭保护。"""
    _patch_batching(monkeypatch, max_tokens=1000, ratio=0)

    with caplog.at_level(logging.WARNING, logger="config"):
        assert milvus_client._token_budget() == int(1000 * 0.85)

    monkeypatch.setattr(milvus_client, "EMBEDDING_TOKEN_SAFETY_RATIO", 5.0)
    with caplog.at_level(logging.WARNING, logger="config"):
        assert milvus_client._token_budget() == 1000


def test_prewarm_token_encoding_is_non_fatal_and_names_cache_dir(monkeypatch, caplog):
    """启动预热失败不抛异常，但日志必须点明缓存目录与预置文件名。"""
    cache_dir = "D:/tmp/tiktoken-cache-286"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
    monkeypatch.setattr(milvus_client, "TIKTOKEN_CACHE_DIR", cache_dir)

    def broken_encoding():
        raise RuntimeError("offline: cannot download BPE")

    monkeypatch.setattr(milvus_client, "_get_token_encoding", broken_encoding)

    with caplog.at_level(logging.WARNING, logger="rag.milvus_client"):
        assert milvus_client.prewarm_token_encoding() is False

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert cache_dir in message
    assert "cl100k_base.tiktoken" in message
    assert "tokenizer unavailable" in message.lower()


def test_prewarm_token_encoding_logs_ready_cache_dir(monkeypatch, caplog):
    cache_dir = "D:/tmp/tiktoken-cache-286-ready"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
    monkeypatch.setattr(milvus_client, "TIKTOKEN_CACHE_DIR", cache_dir)

    with caplog.at_level(logging.INFO, logger="rag.milvus_client"):
        assert milvus_client.prewarm_token_encoding() is True

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "tokenizer ready" in message.lower()
    assert cache_dir in message


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


def test_permanent_error_logs_error_once(monkeypatch, caplog):
    """非可重试失败只在构造错误时记一次 ERROR，不因「批次失败」再打一条。"""
    _patch_batching(monkeypatch)
    _patch_embedding_http(monkeypatch, [_FakeResponse(400)])
    _patch_sleep(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="rag.milvus_client"):
        with pytest.raises(milvus_client.EmbeddingBackendError):
            milvus_client._embed_documents(["hello"])

    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert "400" in errors[0].getMessage()
    assert "400" in milvus_client.embedding_backend_status()["last_error"]


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


def test_retry_after_is_capped_by_max_retry_sleep(monkeypatch):
    """Retry-After: 3600 不能真的睡 1 小时：退避被 MAX_RETRY_SLEEP_SECONDS 截断。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    state = _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(429, headers={"Retry-After": "3600"}),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    sleeps = _patch_sleep(monkeypatch)

    vectors = milvus_client._embed_documents(["hello"])

    assert len(vectors) == 1
    assert len(state["requests"]) == 2
    assert sleeps == [milvus_client.MAX_RETRY_SLEEP_SECONDS]
    assert sleeps[0] <= 30


def test_retry_delay_is_clamped_to_remaining_budget(monkeypatch):
    """退避同时受整份文档剩余预算约束：剩余 5 秒时最多睡 5 秒。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(429, headers={"Retry-After": "3600"}),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    clock, sleeps = _patch_clock(monkeypatch, now=1000.0)
    clock["now"] = 1000.0

    milvus_client._embed_documents(["hello"], deadline=1005.0)

    assert sleeps == [5.0]


def test_exhausted_budget_fails_without_sleeping(monkeypatch):
    """剩余预算已耗尽：不再睡眠、不再重试，直接以最终失败抛出。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    state = _patch_embedding_http(
        monkeypatch,
        [_FakeResponse(429, headers={"Retry-After": "3600"})],
    )
    _clock, sleeps = _patch_clock(monkeypatch, now=1000.0)

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._embed_documents(["hello"], deadline=1000.0)

    assert sleeps == []
    assert len(state["requests"]) == 1
    assert "预算已耗尽" in str(exc_info.value)
    assert milvus_client.embedding_backend_status()["mode"] == "unavailable"


def test_retry_after_http_date_is_parsed(monkeypatch):
    """Retry-After 的 HTTP-date 形式也被解析成秒数（同样受上限约束）。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=5)
    _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)}),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    sleeps = _patch_sleep(monkeypatch)

    milvus_client._embed_documents(["hello"])

    assert len(sleeps) == 1
    assert 3.5 <= sleeps[0] <= 5.0


def test_zero_retries_fails_after_first_attempt(monkeypatch):
    """EMBEDDING_MAX_RETRIES=0：第一次 429 就是最终失败，不睡眠。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 0)
    state = _patch_embedding_http(monkeypatch, [_FakeResponse(429)])
    sleeps = _patch_sleep(monkeypatch)

    with pytest.raises(milvus_client.EmbeddingBackendError):
        milvus_client._embed_documents(["hello"])

    assert len(state["requests"]) == 1
    assert sleeps == []


def test_retryable_attempts_warn_and_success_clears_last_error(monkeypatch, caplog):
    """可重试尝试只记 WARNING；成功路径不留下 last_error，也不打 ERROR。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 2)
    _patch_embedding_http(
        monkeypatch,
        [
            _FakeResponse(429, headers={"Retry-After": "0"}),
            _FakeResponse(200, payload=_success_payload(1)),
        ],
    )
    _patch_sleep(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="rag.milvus_client"):
        milvus_client._embed_documents(["hello"])

    levels = {record.levelname for record in caplog.records}
    assert "ERROR" not in levels
    assert "WARNING" in levels
    assert milvus_client.embedding_backend_status()["last_error"] == ""


def test_final_give_up_logs_error_and_records_last_error(monkeypatch, caplog):
    """只有最终放弃才写 last_error 并打 ERROR。"""
    _patch_batching(monkeypatch)
    monkeypatch.setattr(milvus_client, "EMBEDDING_MAX_RETRIES", 1)
    _patch_embedding_http(monkeypatch, [_FakeResponse(429) for _ in range(5)])
    _patch_sleep(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="rag.milvus_client"):
        with pytest.raises(milvus_client.EmbeddingBackendError):
            milvus_client._embed_documents(["hello"])

    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert errors
    assert any("batch failed" in record.getMessage() for record in errors)
    assert "429" in milvus_client.embedding_backend_status()["last_error"]


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
