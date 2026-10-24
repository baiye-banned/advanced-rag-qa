import contextlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from rag import milvus_client


@pytest.fixture(autouse=True)
def _reset_embedding_state():
    """向量来源状态是进程内全局，用例前后都清空，避免用例之间相互影响。"""

    def reset():
        state = getattr(milvus_client, "_embedding_state", None)
        if state is None:
            # 状态缺失时跳过重置，让用例自身的断言去暴露问题。
            return
        state.update(source="", last_error="", last_error_at="", last_used_at="")

    reset()
    yield
    reset()


@contextlib.contextmanager
def _embedding_server(status_code: int, payload: dict):
    """本地上游桩：模拟向量化接口，默认返回指定状态码与 JSON。"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_lite_uri_is_file_path_not_server_uri():
    assert milvus_client._is_lite_uri("./milvus.db") is True
    assert milvus_client._is_lite_uri("http://127.0.0.1:19530") is False


def test_restore_milvus_uri_env_round_trip(monkeypatch):
    monkeypatch.setattr(milvus_client, "MILVUS_URI", "./milvus.db")
    monkeypatch.setenv("MILVUS_URI", "./milvus.db")

    hidden = milvus_client._hide_lite_uri_from_pymilvus_import()
    assert hidden == "./milvus.db"
    # hide 期间变量必须真的从环境里消失，否则 pymilvus 导入时仍会把它当成 server uri
    assert "MILVUS_URI" not in milvus_client.os.environ

    milvus_client._restore_milvus_uri_env(hidden)
    assert milvus_client.os.environ["MILVUS_URI"] == "./milvus.db"


def test_hide_keeps_server_uri_in_env_for_pymilvus(monkeypatch):
    monkeypatch.setattr(milvus_client, "MILVUS_URI", "http://127.0.0.1:19530")
    monkeypatch.setenv("MILVUS_URI", "http://127.0.0.1:19530")

    assert milvus_client._hide_lite_uri_from_pymilvus_import() is None
    assert milvus_client.os.environ["MILVUS_URI"] == "http://127.0.0.1:19530"


def test_hide_lite_uri_without_env_var_leaves_env_untouched(monkeypatch):
    monkeypatch.setattr(milvus_client, "MILVUS_URI", "./milvus.db")
    monkeypatch.delenv("MILVUS_URI", raising=False)

    assert milvus_client._hide_lite_uri_from_pymilvus_import() is None
    assert "MILVUS_URI" not in milvus_client.os.environ


def test_restore_with_nothing_hidden_does_not_write_placeholder(monkeypatch):
    monkeypatch.delenv("MILVUS_URI", raising=False)

    milvus_client._restore_milvus_uri_env(None)

    # 不能把字符串 "None" 写回环境变量（server uri 场景下 hide 返回 None）
    assert "MILVUS_URI" not in milvus_client.os.environ


def test_add_chunks_replaces_existing_file_chunks(monkeypatch):
    calls = []

    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def delete(self, collection_name, filter):
            calls.append(("delete", collection_name, filter))

        def insert(self, collection_name, data):
            calls.append(("insert", collection_name, data))

    def fake_embedding(texts):
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(milvus_client, "_ensure_collection", lambda: FakeClient())
    monkeypatch.setattr(milvus_client, "_embedding_fn", fake_embedding)

    milvus_client.add_chunks(
        [{"id": "0", "text": "第一段"}, {"id": "1", "text": "第二段"}],
        file_id=7,
        file_name="制度.txt",
        knowledge_base_id=3,
    )

    assert calls[0] == ("delete", milvus_client.COLLECTION_NAME, "file_id == 7")
    assert calls[1][0] == "insert"
    assert [row["id"] for row in calls[1][2]] == ["7_0", "7_1"]


def test_add_chunks_empty_list_still_clears_existing_file_chunks(monkeypatch):
    calls = []

    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def delete(self, collection_name, filter):
            calls.append(("delete", collection_name, filter))

        def insert(self, collection_name, data):
            calls.append(("insert", collection_name, data))

    monkeypatch.setattr(milvus_client, "_ensure_collection", lambda: FakeClient())

    milvus_client.add_chunks([], file_id=9, file_name="空文件.txt", knowledge_base_id=3)

    assert calls == [("delete", milvus_client.COLLECTION_NAME, "file_id == 9")]


def test_query_vectors_empty_query_returns_empty_without_connecting(monkeypatch):
    def fail_connect():
        raise AssertionError("empty query should not connect to Milvus")

    monkeypatch.setattr(milvus_client, "_connect_milvus", fail_connect)

    assert milvus_client.query_vectors("   ", knowledge_base_id=3) == []


def test_query_vectors_normalizes_invalid_hit_file_id(monkeypatch):
    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def load_collection(self, collection_name):
            return None

        def search(self, **kwargs):
            return [
                [
                    {
                        "id": "hit-1",
                        "distance": 0.2,
                        "entity": {
                            "content": "chunk",
                            "file_id": "bad-file-id",
                            "file_name": "制度.txt",
                            "chunk_id": "1",
                        },
                    }
                ]
            ]

    monkeypatch.setattr(milvus_client, "_connect_milvus", lambda: FakeClient())
    monkeypatch.setattr(milvus_client, "_embedding_fn", lambda texts: [[0.1, 0.2] for _ in texts])

    chunks = milvus_client.query_vectors("query", knowledge_base_id=3)

    assert chunks[0]["file_id"] == 0


def test_embedding_default_timeout_is_online_qa_scale():
    """默认超时必须收敛到线上问答量级，否则一轮 9 路召回会被单次调用拖住。"""
    if os.getenv("EMBEDDING_TIMEOUT_SECONDS", "").strip():
        # 部署方显式覆盖时「默认值」已不存在，避免用例在合法配置下变红。
        pytest.skip("EMBEDDING_TIMEOUT_SECONDS 已被环境显式覆盖")
    assert 0 < milvus_client.EMBEDDING_TIMEOUT_SECONDS <= 10


def test_embedding_client_uses_configured_timeout(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1] * milvus_client.EMBEDDING_DIM}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(milvus_client, "EMBEDDING_TIMEOUT_SECONDS", 7)
    monkeypatch.setattr(milvus_client.httpx, "Client", FakeClient)

    vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert captured["timeout"] == 7
    assert len(vectors) == 1


def test_embedding_timeout_does_not_leak_raw_transport_error(monkeypatch):
    """向量化接口超时必须抛出 EmbeddingBackendError，不能泄漏 httpx 原始异常。

    已配置向量化后端时不再静默降级为哈希向量（跨向量空间检索），
    该路由由召回段按 EmbeddingBackendError 逐路跳过。
    """

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            raise httpx.TimeoutException("embedding request timed out")

    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(milvus_client, "EMBEDDING_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(milvus_client.httpx, "Client", FakeClient)

    with pytest.raises(milvus_client.EmbeddingBackendError) as excinfo:
        milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    # 明确不是 httpx 原始异常泄漏，也不是静默降级成哈希向量。
    assert not isinstance(excinfo.value, httpx.HTTPError)
    assert str(excinfo.value)


class _FakeHttpClient:
    """模拟 httpx.Client 上下文管理器，返回固定 JSON 响应。"""

    def __init__(self, status_code=200, payload=None, error=None, url_sink=None):
        self._status_code = status_code
        self._payload = payload
        self._error = error
        self._url_sink = url_sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        if self._url_sink is not None:
            self._url_sink.append(url)
        if self._error is not None:
            raise self._error
        response = httpx.Response(self._status_code, json=self._payload, request=httpx.Request("POST", url))
        return response


def _patch_configured_embedding(monkeypatch, status_code=200, payload=None, error=None):
    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(
        milvus_client.httpx,
        "Client",
        lambda *args, **kwargs: _FakeHttpClient(status_code=status_code, payload=payload, error=error),
    )


def test_configured_embedding_api_429_raises_instead_of_hash_fallback(monkeypatch):
    with _embedding_server(429, {"error": {"message": "Requests rate limit exceeded"}}) as base_url:
        monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", base_url)
        monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")

        with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
            milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    message = str(exc_info.value)
    assert "向量化接口调用失败" in message
    assert "429" in message

    status = milvus_client.embedding_backend_status()
    assert status["configured"] is True
    assert status["mode"] == "unavailable"
    assert "429" in status["last_error"]
    assert status["last_error_at"]
    assert status["last_used_at"] == ""


def test_configured_embedding_request_error_raises(monkeypatch):
    _patch_configured_embedding(monkeypatch, error=RuntimeError("connection reset"))

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert "connection reset" in str(exc_info.value)
    assert milvus_client.embedding_backend_status()["mode"] == "unavailable"


def test_configured_embedding_invalid_response_shape_raises(monkeypatch):
    _patch_configured_embedding(monkeypatch, payload={"data": []})

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert "响应结构异常" in str(exc_info.value)


def _vector(value: float = 0.1) -> list[float]:
    return [value] * milvus_client.EMBEDDING_DIM


def test_configured_embedding_success_reports_openai_mode(monkeypatch):
    urls = []
    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(
        milvus_client.httpx,
        "Client",
        lambda *args, **kwargs: _FakeHttpClient(
            payload={"data": [{"index": 0, "embedding": _vector()}]},
            url_sink=urls,
        ),
    )

    vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert vectors == [_vector()]
    assert urls == ["https://embedding.example/v1/embeddings"]
    status = milvus_client.embedding_backend_status()
    assert status["mode"] == "openai-compatible"
    assert status["last_error"] == ""
    assert status["last_used_at"]


def test_configured_embedding_non_object_payload_raises(monkeypatch):
    _patch_configured_embedding(monkeypatch, payload=[1, 2, 3])

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert "向量化接口调用失败" in str(exc_info.value)
    assert milvus_client.embedding_backend_status()["mode"] == "unavailable"


def test_configured_embedding_wrong_dimension_raises(monkeypatch):
    _patch_configured_embedding(monkeypatch, payload={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    message = str(exc_info.value)
    assert "响应结构异常" in message
    assert str(milvus_client.EMBEDDING_DIM) in message


def test_configured_embedding_scalar_embedding_raises(monkeypatch):
    _patch_configured_embedding(monkeypatch, payload={"data": [{"index": 0, "embedding": "not-a-vector"}]})

    with pytest.raises(milvus_client.EmbeddingBackendError) as exc_info:
        milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert "响应结构异常" in str(exc_info.value)


def test_status_reports_unavailable_after_failure_following_success(monkeypatch):
    """先成功再失败：mode 必须翻转为 unavailable，而不是停留在上一次的成功来源。"""
    _patch_configured_embedding(monkeypatch, payload={"data": [{"index": 0, "embedding": _vector()}]})
    function = milvus_client._OpenAICompatibleEmbeddingFunction()
    function(["hello"])
    assert milvus_client.embedding_backend_status()["mode"] == "openai-compatible"

    _patch_configured_embedding(monkeypatch, status_code=429, payload={"error": {"message": "rate limited"}})
    with pytest.raises(milvus_client.EmbeddingBackendError):
        function(["hello"])

    status = milvus_client.embedding_backend_status()
    assert status["mode"] == "unavailable"
    assert "429" in status["last_error"]
    assert status["last_used_at"]


def test_unconfigured_embedding_keeps_hash_fallback_and_reports_hash_mode(monkeypatch):
    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "")

    vectors = milvus_client._OpenAICompatibleEmbeddingFunction()(["hello"])

    assert len(vectors) == 1
    assert len(vectors[0]) == milvus_client.EMBEDDING_DIM
    status = milvus_client.embedding_backend_status()
    assert status["configured"] is False
    assert status["mode"] == "hash-fallback"
    assert status["last_error"] == ""
    assert status["last_used_at"]


def test_add_chunks_embedding_failure_keeps_existing_index_untouched(monkeypatch):
    calls = []

    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def delete(self, collection_name, filter):
            calls.append(("delete", collection_name, filter))

        def insert(self, collection_name, data):
            calls.append(("insert", collection_name, data))

    def failing_embedding(texts):
        raise milvus_client.EmbeddingBackendError("向量化接口调用失败（https://embedding.example/v1/embeddings）：429 Too Many Requests")

    monkeypatch.setattr(milvus_client, "_ensure_collection", lambda: FakeClient())
    monkeypatch.setattr(milvus_client, "_embedding_fn", failing_embedding)

    with pytest.raises(milvus_client.EmbeddingBackendError):
        milvus_client.add_chunks(
            [{"id": "0", "text": "第一段"}, {"id": "1", "text": "第二段"}],
            file_id=7,
            file_name="制度.txt",
            knowledge_base_id=3,
        )

    assert calls == []


def test_query_vectors_embedding_failure_raises_before_search(monkeypatch):
    search_calls = []

    class FakeClient:
        def has_collection(self, collection_name):
            return True

        def load_collection(self, collection_name):
            return None

        def search(self, **kwargs):
            search_calls.append(kwargs)
            return [[]]

    def failing_embedding(texts):
        raise milvus_client.EmbeddingBackendError("向量化接口调用失败：429 Too Many Requests")

    monkeypatch.setattr(milvus_client, "_connect_milvus", lambda: FakeClient())
    monkeypatch.setattr(milvus_client, "_embedding_fn", failing_embedding)

    with pytest.raises(milvus_client.EmbeddingBackendError):
        milvus_client.query_vectors("迟到怎么处罚", knowledge_base_id=3)

    assert search_calls == []


class _FakeCollection:
    """只记录写入、不连真实 Milvus：分批用例只关心向量化调用的形态。"""

    def __init__(self, inserted):
        self._inserted = inserted

    def has_collection(self, collection_name):
        return True

    def delete(self, collection_name, filter):
        return None

    def insert(self, collection_name, data):
        self._inserted.append(data)
        return {"insert_count": len(data)}


def _patch_collection(monkeypatch, inserted):
    monkeypatch.setattr(milvus_client, "_ensure_collection", lambda: _FakeCollection(inserted))


def _recording_embedding(calls):
    def embed(texts):
        calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    return embed


def test_add_chunks_splits_embedding_requests_by_configured_batch_size(monkeypatch):
    """整份文档不再压进一次向量化请求：调用次数与单次条数都受配置约束。"""
    batch_size = 4
    chunk_count = 21
    calls = []
    inserted = []
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_SIZE", batch_size)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_MAX_CHARS", 10**9)
    monkeypatch.setattr(milvus_client, "_embedding_fn", _recording_embedding(calls))
    _patch_collection(monkeypatch, inserted)

    milvus_client.add_chunks(
        [{"id": str(index), "text": "条款"} for index in range(chunk_count)],
        file_id=21,
        file_name="制度.txt",
        knowledge_base_id=3,
    )

    assert len(calls) >= -(-chunk_count // batch_size)
    assert max(len(call) for call in calls) <= batch_size
    # 分批不得丢切片或改变顺序：向量与切片仍按 zip(strict=True) 对齐。
    assert [text for call in calls for text in call] == ["条款"] * chunk_count
    assert len(inserted) == 1
    assert len(inserted[0]) == chunk_count


def test_add_chunks_splits_embedding_requests_by_configured_char_limit(monkeypatch):
    """条数没超也要按字符数切：单次请求体大小同样有上限。"""
    max_chars = 100
    text = "条" * 30
    calls = []
    inserted = []
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_SIZE", 10**9)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_MAX_CHARS", max_chars)
    monkeypatch.setattr(milvus_client, "_embedding_fn", _recording_embedding(calls))
    _patch_collection(monkeypatch, inserted)

    milvus_client.add_chunks(
        [{"id": str(index), "text": text} for index in range(10)],
        file_id=22,
        file_name="制度.txt",
        knowledge_base_id=3,
    )

    assert sum(len(call) for call in calls) == 10
    assert all(sum(len(item) for item in call) <= max_chars for call in calls)
    # 每批 3 条（90 字符），第 4 条会越限，故 10 条切成 4 批。
    assert [len(call) for call in calls] == [3, 3, 3, 1]


def test_add_chunks_gives_an_oversized_chunk_its_own_batch(monkeypatch):
    """单条切片自身超过字符上限时独占一批：切片是检索最小单位，不再二次切分。"""
    oversized = "长" * 500
    calls = []
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_SIZE", 10**9)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_MAX_CHARS", 50)
    monkeypatch.setattr(milvus_client, "_embedding_fn", _recording_embedding(calls))
    _patch_collection(monkeypatch, [])

    milvus_client.add_chunks(
        [{"id": "0", "text": "短"}, {"id": "1", "text": oversized}, {"id": "2", "text": "短"}],
        file_id=23,
        file_name="制度.txt",
        knowledge_base_id=3,
    )

    assert [len(call) for call in calls] == [1, 1, 1]
    assert calls[1] == [oversized]


class _EmbeddingResponse:
    def __init__(self, count):
        self._count = count

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"index": index, "embedding": [0.5] * milvus_client.EMBEDDING_DIM} for index in range(self._count)]}


def test_ingest_timeout_is_separate_from_retrieval_timeout(monkeypatch):
    """入库按批用入库预算，检索仍用在线问答预算，且入库覆盖不残留到之后的调用。"""
    timeouts = []

    class CapturingClient:
        def __init__(self, *args, **kwargs):
            timeouts.append(kwargs.get("timeout"))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, url, json, headers):
            return _EmbeddingResponse(len(json["input"]))

    monkeypatch.setattr(milvus_client, "EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setattr(milvus_client, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(milvus_client.httpx, "Client", CapturingClient)
    monkeypatch.setattr(milvus_client, "EMBEDDING_TIMEOUT_SECONDS", 7)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_TIMEOUT_SECONDS", 41)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_SIZE", 1)
    monkeypatch.setattr(milvus_client, "EMBEDDING_INGEST_BATCH_MAX_CHARS", 10**9)
    _patch_collection(monkeypatch, [])

    milvus_client.add_chunks(
        [{"id": "0", "text": "第一段"}, {"id": "1", "text": "第二段"}],
        file_id=24,
        file_name="制度.txt",
        knowledge_base_id=3,
    )

    assert timeouts == [41, 41]

    # 同一个可调用对象在检索路径上仍读在线问答预算，说明覆盖只活在入库调用域内。
    milvus_client._embedding_fn(["迟到怎么处罚"])

    assert timeouts == [41, 41, 7]


def test_connect_milvus_constructs_client_once_under_concurrency(monkeypatch):
    """并发检索只能构造一个 Milvus Lite 客户端。

    Milvus Lite 对同一数据目录独占加锁（DataDirLockedError），第二个实例会让整轮
    检索 500。构造过程故意放慢，放大 check-then-set 的竞态窗口。
    """
    construct_calls = []

    class SlowClient:
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            time.sleep(0.05)

    monkeypatch.setattr(milvus_client, "MilvusClient", SlowClient)
    monkeypatch.setattr(milvus_client, "_client", None)

    thread_count = 8
    start = threading.Barrier(thread_count)
    results = []
    errors = []

    def connect():
        start.wait()
        try:
            results.append(milvus_client._connect_milvus())
        except Exception as exc:  # 竞态下的异常要作为断言失败暴露，而不是吞掉
            errors.append(exc)

    threads = [threading.Thread(target=connect) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads), "并发初始化出现死锁"
    assert errors == []
    assert len(construct_calls) == 1
    assert len(results) == thread_count
    # 所有线程拿到的必须是同一个实例。
    assert len({id(client) for client in results}) == 1


def test_connect_milvus_retries_after_initialization_failure(monkeypatch):
    """初始化失败不得缓存半初始化实例：后续调用要能重新构造。"""
    attempts = []

    class FlakyClient:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError("another process holds the lock on milvus.db")

    monkeypatch.setattr(milvus_client, "MilvusClient", FlakyClient)
    monkeypatch.setattr(milvus_client, "_client", None)

    with pytest.raises(RuntimeError, match="another process holds the lock"):
        milvus_client._connect_milvus()

    # 失败后不能留下不可用的实例，否则后续请求会一直复用坏对象。
    assert milvus_client._client is None

    client = milvus_client._connect_milvus()

    assert client is not None
    assert len(attempts) == 2
