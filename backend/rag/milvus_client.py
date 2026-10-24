import hashlib
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterator, Optional

import httpx
import numpy as np
import tiktoken

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_INGEST_BATCH_MAX_CHARS,
    EMBEDDING_INGEST_BATCH_SIZE,
    EMBEDDING_INGEST_TIMEOUT_SECONDS,
    EMBEDDING_MAX_BATCH_ITEMS,
    EMBEDDING_MAX_REQUEST_TOKENS,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT_SECONDS,
    EMBEDDING_TOKEN_SAFETY_RATIO,
    MILVUS_COLLECTION_NAME,
    MILVUS_DB_NAME,
    MILVUS_PASSWORD,
    MILVUS_TOKEN,
    MILVUS_URI,
    MILVUS_USER,
    RETRIEVAL_ROUTE_TOP_K,
)


def _is_lite_uri(uri: str) -> bool:
    return "://" not in (uri or "")


def _hide_lite_uri_from_pymilvus_import() -> str | None:
    """pymilvus treats MILVUS_URI as a server URI during import; Lite uses a file path."""
    if not _is_lite_uri(MILVUS_URI):
        return None
    return os.environ.pop("MILVUS_URI", None)


def _restore_milvus_uri_env(value: str | None) -> None:
    if value is not None:
        os.environ["MILVUS_URI"] = value


_milvus_uri_env = _hide_lite_uri_from_pymilvus_import()

# pymilvus 在导入期就把 MILVUS_URI 读进 settings.Config（类体赋值，导入即冻结）。
# 这两行 import 必须停在此处：上移到文件头会让 Lite 的文件路径被当成 server URI，
# 直接抛 ConnectionConfigException。故按 E402 豁免，而非重排。
from pymilvus import DataType, MilvusClient  # noqa: E402
from pymilvus.orm.schema import CollectionSchema, FieldSchema  # noqa: E402

_restore_milvus_uri_env(_milvus_uri_env)

_client: MilvusClient | None = None
# 进程内单例的初始化锁：检索经 asyncio.to_thread 并发进入，check-then-construct 必须原子化。
_milvus_client_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _collection_name() -> str:
    if MILVUS_COLLECTION_NAME:
        return MILVUS_COLLECTION_NAME
    model_part = "".join(ch if ch.isalnum() else "_" for ch in EMBEDDING_MODEL.lower()).strip("_")
    return f"knowledge_chunks_semantic_{model_part}_{EMBEDDING_DIM}"


COLLECTION_NAME = _collection_name()


class EmbeddingBackendError(RuntimeError):
    """向量化后端不可用或返回异常。

    调用方必须显式处理该异常（失败上传/检索降级/如实上报），
    禁止回退到哈希向量——哈希向量与语义向量不在同一空间，
    混写会让索引里的历史向量永远无法与查询向量正确比较。
    """


# 用户可见轨迹里的向量化失败文案：`_embedding_failure` 拼出的原文含上游 embedding 服务
# 地址与原始异常，进 `retrieval_trace` 后随消息历史响应体到达用户，因此只在写入轨迹的
# 副本上收敛；「当前是否可用」由 `mode` 字段如实表达，排障靠服务端日志。
EMBEDDING_UNAVAILABLE_MESSAGE = "向量化服务暂时不可用，本次未使用语义向量召回。"


class _HashEmbeddingFunction:
    """Local development embedding used only when no embedding API is configured."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            result.append(vec.tolist())
        return result


def _embedding_configured() -> bool:
    return bool(EMBEDDING_BASE_URL and EMBEDDING_API_KEY)


def _embeddings_url() -> str:
    base_url = EMBEDDING_BASE_URL.rstrip("/")
    return f"{base_url}/embeddings" if base_url.endswith("/v1") else f"{base_url}/v1/embeddings"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 最近一次实际生效的向量来源与失败信息（进程内），供状态接口如实上报。
_embedding_state: dict[str, str] = {
    "source": "",
    "last_error": "",
    "last_error_at": "",
    "last_used_at": "",
}


def _record_embedding_success(source: str) -> None:
    _embedding_state["source"] = source
    _embedding_state["last_error"] = ""
    _embedding_state["last_used_at"] = _utc_now()


def _embedding_failure(
    message: str,
    *,
    retryable: bool = False,
    retry_after: Optional[float] = None,
) -> EmbeddingBackendError:
    _embedding_state["last_error"] = message
    _embedding_state["last_error_at"] = _utc_now()
    logger.error("%s", message)
    error = EmbeddingBackendError(message)
    # 只有 429/5xx 打上可重试标记；400 等永久错误重试多少次都不会成功。
    error.retryable = retryable
    error.retry_after = retry_after
    return error


# token 估算是确定性的本地计算，绝不能拿字符数冒充 token：后者与真实 token 数
# 在中文/英文/混排下差异很大，按字符数打包仍会发出超限请求。
_TOKEN_ENCODING_NAME = "cl100k_base"  # scan-secrets:allow tokenizer encoding name, not a credential
_token_encoding = None
_token_encoding_lock = threading.Lock()


def _get_token_encoding():
    """惰性加载 tiktoken 编码器：导入 milvus_client 不应依赖首次下载分词表。"""
    global _token_encoding
    if _token_encoding is None:
        with _token_encoding_lock:
            if _token_encoding is None:
                _token_encoding = tiktoken.get_encoding(_TOKEN_ENCODING_NAME)
    return _token_encoding


def _estimate_tokens(text: str) -> int:
    """用 cl100k_base 估算单条文本的 token 数；失败时显式报错而不是退回字符数。"""
    try:
        # 文档正文可能出现 "<|endoftext|>" 这类字样；按普通文本编码，既不会抛错，
        # 估算也只会偏保守（拆成多个 token）。
        return len(_get_token_encoding().encode(text, disallowed_special=()))
    except Exception as exc:
        raise _embedding_failure(
            f"token 估算失败（{_TOKEN_ENCODING_NAME}）：{exc}"
        ) from exc


def _token_budget() -> int:
    """单请求 token 预算 = 模型上限 × 安全系数。"""
    limit = max(1, EMBEDDING_MAX_REQUEST_TOKENS)
    return max(1, int(limit * EMBEDDING_TOKEN_SAFETY_RATIO))


def _batch_item_limit() -> int:
    """条数上限取模型能力与旧配置的较小值，旧配置只能收紧、不能突破模型上限。"""
    return max(1, min(max(1, EMBEDDING_MAX_BATCH_ITEMS), max(1, EMBEDDING_INGEST_BATCH_SIZE)))


# 入库与检索的超时预算分开：检索保持在线问答量级，入库按批放大。
# 覆盖值放在 ContextVar 而不是模块全局：并发上传各自跑在 to_thread 工作线程里，
# asyncio.to_thread 会把调用方的上下文复制进工作线程，覆盖因此只跟随当前入库调用，
# 不会串到同一进程里并发的检索请求上。
_ingest_timeout_override: ContextVar[Optional[int]] = ContextVar("_embedding_ingest_timeout", default=None)


@contextmanager
def _ingest_embedding_timeout():
    token = _ingest_timeout_override.set(EMBEDDING_INGEST_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        _ingest_timeout_override.reset(token)


def _embedding_timeout_seconds() -> int:
    override = _ingest_timeout_override.get()
    return EMBEDDING_TIMEOUT_SECONDS if override is None else override


def _retry_after_seconds(response) -> Optional[float]:
    """解析 Retry-After（秒数或 HTTP 日期）；解析不了就交给指数退避。"""
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(raw)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


class _OpenAICompatibleEmbeddingFunction:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not _embedding_configured():
            logger.info("Embedding API is not configured; using local hash vectors (development only)")
            vectors = _hash_embedding_fn(texts)
            _record_embedding_success("hash-fallback")
            return vectors

        url = _embeddings_url()
        headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": EMBEDDING_MODEL,
            "input": list(texts),
            "dimensions": EMBEDDING_DIM,
        }
        try:
            with httpx.Client(timeout=_embedding_timeout_seconds()) as client:
                response = client.post(url, json=payload, headers=headers)
            # 429/5xx 是瞬态错误：标记成可重试，由入库循环按配置做有上限退避。
            # 400 等其他 4xx 是永久错误，交给 raise_for_status 收敛成普通失败。
            status = getattr(response, "status_code", None)
            if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
                retry_after = _retry_after_seconds(response)
                raise _embedding_failure(
                    f"向量化接口调用失败（{url}）：HTTP {status}"
                    + (f"，Retry-After={retry_after:g}s" if retry_after is not None else ""),
                    retryable=True,
                    retry_after=retry_after,
                )
            response.raise_for_status()
            data = response.json()
            # 解析也放进 try：响应不是对象、data 不是列表时同样收敛成 EmbeddingBackendError。
            vectors = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
            embeddings = [item.get("embedding") for item in vectors]
        except EmbeddingBackendError:
            raise
        except Exception as exc:
            raise _embedding_failure(f"向量化接口调用失败（{url}）：{exc}") from exc

        if len(embeddings) != len(texts):
            raise _embedding_failure(
                f"向量化接口响应结构异常（{url}）：期望 {len(texts)} 条向量，实际得到 {len(embeddings)} 条"
            )
        if not all(isinstance(vector, list) and len(vector) == EMBEDDING_DIM for vector in embeddings):
            # 维度不符的向量写进 Milvus 只会在更深处报错，这里提前给出可读原因。
            actual = [len(vector) if isinstance(vector, list) else type(vector).__name__ for vector in embeddings]
            raise _embedding_failure(
                f"向量化接口响应结构异常（{url}）：期望 {EMBEDDING_DIM} 维向量，实际得到 {actual[:3]}"
            )
        _record_embedding_success("openai-compatible")
        return embeddings


_hash_embedding_fn = _HashEmbeddingFunction()
_embedding_fn = _OpenAICompatibleEmbeddingFunction()


def _client_kwargs() -> dict:
    kwargs = {
        "uri": MILVUS_URI,
    }
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    else:
        if MILVUS_USER:
            kwargs["user"] = MILVUS_USER
        if MILVUS_PASSWORD:
            kwargs["password"] = MILVUS_PASSWORD
    if MILVUS_DB_NAME and not _is_lite_uri(MILVUS_URI):
        kwargs["db_name"] = MILVUS_DB_NAME
    return kwargs


def _connect_milvus() -> MilvusClient:
    global _client
    # 两重检查：已初始化的路径不加锁，避免每轮检索都串行化。
    if _client is None:
        # 首轮检索会从多个 to_thread 工作线程并发进来。Milvus Lite 对同一数据目录
        # 只允许一个 server 实例，未加锁时多个线程会同时构造 MilvusClient，后来的
        # 实例拿不到目录锁（DataDirLockedError），整轮检索直接 500。
        # 也不能改成每个线程各建一个 Lite client：锁是进程级文件锁，同一进程同样
        # 互斥，只会把并发请求变成持续失败；单例才是 Lite 的正确用法。
        with _milvus_client_lock:
            if _client is None:
                # 构造成功才赋值：失败时 _client 仍为 None，后续调用可以重试。
                _client = MilvusClient(**_client_kwargs())
    return _client


def _collection_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file_id", dtype=DataType.INT64),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="knowledge_base_id", dtype=DataType.INT64),
    ]
    return CollectionSchema(fields, description="Enterprise knowledge base chunks", enable_dynamic_field=False)


def _index_params():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    return index_params


def _ensure_collection() -> MilvusClient:
    client = _connect_milvus()
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=_collection_schema(),
            index_params=_index_params(),
        )
    client.load_collection(COLLECTION_NAME)
    return client


def embedding_backend_status() -> dict:
    """如实上报向量化后端状态：mode 反映最近一次调用的真实结果，而非配置推断。"""
    configured = _embedding_configured()
    source = _embedding_state["source"]
    if _embedding_state["last_error"]:
        # last_error 非空 ⇔ 最近一次调用失败（成功路径会清空它）：当前没有可用来源。
        mode = "unavailable"
    elif source:
        mode = source
    else:
        mode = "openai-compatible" if configured else "hash-fallback"
    return {
        "configured": configured,
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIM,
        "mode": mode,
        "vector_store": "milvus_lite" if _is_lite_uri(MILVUS_URI) else "milvus",
        "milvus_configured": bool(MILVUS_URI),
        "milvus_uri": MILVUS_URI,
        "collection": COLLECTION_NAME,
        "last_used_at": _embedding_state["last_used_at"],
        "last_error": _embedding_state["last_error"],
        "last_error_at": _embedding_state["last_error_at"],
    }


def embedding_trace_status(status: dict) -> dict:
    """把 `embedding_backend_status()` 的结果收敛成可进用户可见轨迹的副本。

    `last_error` 是 `_embedding_failure()` 拼的原文（上游 embedding 地址 + 原始异常），
    它会经 `retrieval_trace`（`retrieval.py` 与 `chat_service.py` 两处写入）落到 assistant
    消息，再由 `GET /api/chat/conversations/{cid}` 原样返回给用户；`_embedding_state` 是
    进程级状态，未命中该次失败的用户也会拿到它。这里只改副本，`embedding_backend_status()`
    的返回值保持原文不变——那是状态接口的诊断面，且由 `tests/test_milvus_client.py` 锁定。
    """
    masked = dict(status)
    if masked.get("last_error"):
        masked["last_error"] = EMBEDDING_UNAVAILABLE_MESSAGE
    return masked


# 指数退避的起点：第 1 次重试等 0.5s，第 2 次等 1s。Retry-After 更大时以它为准。
_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 0.5


def _retry_delay_seconds(attempt: int, retry_after: Optional[float]) -> float:
    delay = _EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    if retry_after is not None:
        delay = max(delay, retry_after)
    return delay


def _embedding_batches(texts: list[str]) -> Iterator[list[str]]:
    """按模型能力切分待向量化文本：条数、累计 token、旧字符上限三约束，先到者生效。

    - 条数上限 = min(EMBEDDING_MAX_BATCH_ITEMS, 旧 EMBEDDING_INGEST_BATCH_SIZE)。
    - token 预算 = EMBEDDING_MAX_REQUEST_TOKENS × EMBEDDING_TOKEN_SAFETY_RATIO，
      由 tiktoken 的 cl100k_base 确定性估算，不拿字符数冒充 token。
    - 旧字符上限仅在显式配置为正数时生效，默认关闭。

    单条文本自身超过 token 预算时直接抛 EmbeddingBackendError：该请求必然被上游拒绝，
    预先校验可以保证一个请求都不发；也不二次切分，切片是检索的最小语义单位。
    """
    batch_size = _batch_item_limit()
    max_chars = EMBEDDING_INGEST_BATCH_MAX_CHARS
    token_budget = _token_budget()

    # 先整篇估算再切分：任何一条超预算都在发出第一个请求之前失败，避免「前几批已发送、
    # 后面必失败」的半途状态。
    token_counts = [_estimate_tokens(text) for text in texts]
    for index, tokens in enumerate(token_counts):
        if tokens > token_budget:
            raise _embedding_failure(
                f"切片 #{index + 1} 估算 {tokens} tokens，超过单请求 token 预算 {token_budget}"
                f"（EMBEDDING_MAX_REQUEST_TOKENS={EMBEDDING_MAX_REQUEST_TOKENS} × "
                f"EMBEDDING_TOKEN_SAFETY_RATIO={EMBEDDING_TOKEN_SAFETY_RATIO}）；"
                "已拒绝发送必然失败的请求，且不拆分切片以免改变检索语义。"
            )

    batch: list[str] = []
    batch_tokens = 0
    batch_chars = 0
    for text, tokens in zip(texts, token_counts):
        over_chars = max_chars > 0 and batch_chars + len(text) > max_chars
        if batch and (
            len(batch) >= batch_size
            or batch_tokens + tokens > token_budget
            or over_chars
        ):
            yield batch
            batch, batch_tokens, batch_chars = [], 0, 0
        batch.append(text)
        batch_tokens += tokens
        batch_chars += len(text)
    if batch:
        yield batch


def _embed_documents(documents: list[str]) -> list[list[float]]:
    """整份文档分批向量化：每批一次请求，共用入库超时预算。

    429/5xx 按 EMBEDDING_MAX_RETRIES 做有上限指数退避（尊重 Retry-After），
    400 等永久错误立即失败。任一批最终失败即抛出 EmbeddingBackendError，已算出的
    向量不会写库，调用方据此回滚——旧索引保持原样，不会留下「删了旧的、没写新的」空洞。

    重试只发生在这里，整份文档的预算仍由调用方 KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS
    约束：退避等待同样消耗总预算，不能靠无限重试把超时伪装成成功。
    """
    embeddings: list[list[float]] = []
    batches = 0
    max_retries = max(0, EMBEDDING_MAX_RETRIES)
    with _ingest_embedding_timeout():
        for batch_index, batch in enumerate(_embedding_batches(documents), start=1):
            estimated_tokens = sum(_estimate_tokens(text) for text in batch)
            started = time.perf_counter()
            attempts = 0
            while True:
                attempts += 1
                try:
                    batch_embeddings = _embedding_fn(batch)
                    break
                except EmbeddingBackendError as exc:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if not getattr(exc, "retryable", False) or attempts > max_retries:
                        logger.error(
                            "Ingest embedding batch failed: batch=%s items=%s estimated_tokens=%s "
                            "attempts=%s elapsed_ms=%.1f",
                            batch_index,
                            len(batch),
                            estimated_tokens,
                            attempts,
                            elapsed_ms,
                        )
                        raise
                    delay = _retry_delay_seconds(attempts, getattr(exc, "retry_after", None))
                    logger.warning(
                        "Ingest embedding batch retry: batch=%s items=%s estimated_tokens=%s "
                        "attempt=%s max_retries=%s delay_s=%.2f",
                        batch_index,
                        len(batch),
                        estimated_tokens,
                        attempts,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Ingest embedding batch completed: batch=%s items=%s estimated_tokens=%s "
                "attempts=%s elapsed_ms=%.1f",
                batch_index,
                len(batch),
                estimated_tokens,
                attempts,
                elapsed_ms,
            )
            embeddings.extend(batch_embeddings)
            batches += 1
    logger.info(
        "Ingest embedding completed: chunks_count=%s batches=%s item_limit=%s "
        "token_budget=%s max_retries=%s",
        len(documents),
        batches,
        _batch_item_limit(),
        _token_budget(),
        max_retries,
    )
    return embeddings


def add_chunks(chunks: list[dict], file_id: int, file_name: str, knowledge_base_id: int):
    client = _ensure_collection()
    if not chunks:
        _delete_file_chunks(client, file_id)
        logger.info("Milvus replace completed: file_id=%s chunks_count=0 action=replace_empty", file_id)
        return
    documents = [chunk["text"] for chunk in chunks]
    # 先取向量再删旧数据：向量化失败时抛错，旧索引保持原样，由调用方回滚。
    embeddings = _embed_documents(documents)
    _delete_file_chunks(client, file_id)
    rows = [
        {
            "id": f"{file_id}_{chunk['id']}",
            "embedding": embedding,
            "content": text,
            "file_id": int(file_id),
            "file_name": file_name,
            "chunk_id": str(chunk["id"]),
            "knowledge_base_id": int(knowledge_base_id),
        }
        for chunk, embedding, text in zip(chunks, embeddings, documents, strict=True)
    ]
    client.insert(collection_name=COLLECTION_NAME, data=rows)
    logger.info(
        "Milvus replace completed: file_id=%s chunks_count=%s knowledge_base_id=%s action=replace",
        file_id,
        len(chunks),
        knowledge_base_id,
    )


def _delete_file_chunks(client: MilvusClient, file_id: int) -> None:
    if not client.has_collection(COLLECTION_NAME):
        logger.info("Milvus delete skipped: file_id=%s chunks_count=0 action=none", file_id)
        return
    client.delete(collection_name=COLLECTION_NAME, filter=f"file_id == {int(file_id)}")


def delete_file_chunks(file_id: int):
    client = _connect_milvus()
    _delete_file_chunks(client, file_id)
    logger.info("Milvus delete completed: file_id=%s action=delete", file_id)


def query_vectors(
    query: str,
    top_k: int = RETRIEVAL_ROUTE_TOP_K,
    knowledge_base_id: Optional[int] = None,
    route: str = "vector",
) -> list[dict]:
    if not str(query or "").strip():
        return []

    client = _connect_milvus()
    if not client.has_collection(COLLECTION_NAME):
        return []
    client.load_collection(COLLECTION_NAME)
    query_embedding = _embedding_fn([query])[0]
    filter_expr = f"knowledge_base_id == {int(knowledge_base_id)}" if knowledge_base_id is not None else ""
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        anns_field="embedding",
        search_params={"metric_type": "COSINE"},
        limit=min(max(top_k, 1), 20),
        filter=filter_expr,
        output_fields=["content", "file_id", "file_name", "chunk_id", "knowledge_base_id"],
    )
    if not results:
        return []

    chunks = []
    for hit in results[0]:
        hit_data = _normalize_hit(hit)
        chunks.append(
            {
                "id": str(hit_data.get("id", "")),
                "chunk_id": str(hit_data.get("chunk_id", "")),
                "content": str(hit_data.get("content", "")),
                "file_name": str(hit_data.get("file_name", "")),
                "file_id": _safe_int(hit_data.get("file_id")),
                "route": route,
                "distance": hit_data.get("distance"),
            }
        )
    return chunks


def _normalize_hit(hit) -> dict:
    if isinstance(hit, dict):
        entity = hit.get("entity")
        if isinstance(entity, dict):
            data = dict(entity)
        else:
            data = dict(hit)
        data.setdefault("id", hit.get("id"))
        data.setdefault("distance", hit.get("distance"))
        return data

    data = {}
    for key in ("id", "distance"):
        if hasattr(hit, key):
            data[key] = getattr(hit, key)
    entity = getattr(hit, "entity", None)
    if entity is not None:
        if isinstance(entity, dict):
            data.update(entity)
        else:
            for key in ("content", "file_id", "file_name", "chunk_id", "knowledge_base_id"):
                if hasattr(entity, key):
                    data[key] = getattr(entity, key)
                elif hasattr(entity, "get"):
                    try:
                        value = entity.get(key)
                    except Exception:
                        value = None
                    if value is not None:
                        data[key] = value
    return data


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
