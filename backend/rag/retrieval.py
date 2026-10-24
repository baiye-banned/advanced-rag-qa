from __future__ import annotations

import asyncio
import contextvars
import heapq
import logging
import math
import queue
import re
import threading
import unicodedata
from collections import OrderedDict
from typing import Any, Iterator, NamedTuple

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from config import (
    BM25_B,
    BM25_ENABLED,
    BM25_K1,
    BM25_MAX_DOCS,
    BM25_MAX_SCORED_CANDIDATES,
    BM25_MAX_TERMS,
    PARENT_CHILD_ENABLED,
    PARENT_CONTEXT_MAX_TOKENS,
    RETRIEVAL_ROUTE_TOP_K,
)
from crud.knowledge_file import count_chunk_tokens, strip_extraction_markers
from database.session import SessionLocal
from model.models import KnowledgeChunkMetadata, KnowledgeFile
from rag.bm25 import (
    BM25CorpusTooLarge,
    BM25Index,
    BM25VocabularyTooLarge,
    query_terms as bm25_query_terms,
)
from rag.llm import call_chat_json, call_router_json
from rag.milvus_client import (
    EMBEDDING_UNAVAILABLE_MESSAGE,
    EmbeddingBackendError,
    embedding_backend_status,
    embedding_trace_status,
    query_vectors,
)
from rag.rerank import (
    KEYWORD_CHUNK_NAMESPACE,
    chunk_content_key,
    chunk_key,
    chunk_scheme,
    rerank_chunks,
    select_final_chunks,
    trace_chunk,
)
from service.utils_service import _internal_error_detail


ROUTE_CONFIDENCE_THRESHOLD = 0.55

# 关键词召回（issue #59）：窗口几何、取数批大小与 SQL 预筛参数。
# 峰值内存由「本批取回的正文」决定，与知识库文件总数无关：批大小按上一批的实际体积自适应
# （KEYWORD_RECALL_BATCH_CHARS），小文件多取几行省往返，大文件少取几行压内存。但要说清
# 「上界」的量级，不能当硬性天花板：首个冷启动批固定 KEYWORD_RECALL_BATCH_SIZE 行、不按
# 预算裁剪，单个文件本身就超预算时也无法再小；命中文件在归一化阶段还有瞬时放大，去空白
# re.sub 一项实测 6.8 字节/字符（200 万字符中文语料，带空白），叠上 lower() 后 14.0 字节/
# 字符，case_fold=False 只免掉 lower()、免不掉 re.sub。实测 2 个 1000 万字符且全部命中的
# 文件：新实现峰值 94MB / 1.19s（旧实现 222MB / 2.70s）。
KEYWORD_CHUNK_SIZE = 900
KEYWORD_CHUNK_OVERLAP = 180
KEYWORD_RECALL_BATCH_SIZE = 2
KEYWORD_RECALL_BATCH_ROWS_MAX = 32
KEYWORD_RECALL_BATCH_CHARS = 4_000_000
LIKE_ESCAPE = "!"
# 归一化后的关键词里出现这些字符时，正文还可能用另一个「SQL 的 LOWER 折不到」的写法
# （键 = 归一化结果里的字符，值 = 它的另一个来源）。全码位扫描确认只有两处：
# İ(U+0130).lower() == 'i' + U+0307（'i' 与那个组合点各有来源）、K(U+212A).lower() == 'k'。
_UNFOLDED_CASE_SOURCES = {
    "i": "İ",  # 拉丁大写 I 带点
    "k": "K",  # KELVIN SIGN
}
# 反过来，一个原文字符折出两个归一化字符时，按「一个字符占一位」对齐的 LIKE 模式没法表达：
# İ 同时提供 'i' 与组合点 U+0307，而模式要求它们各占一位，两者对不上。关键词里出现这个
# 组合点时放弃预筛。（用转义写而不是裸字符：组合点不可见，裸写读不出是什么。）
_MULTI_CHAR_FOLD_TARGETS = "\u0307"
# 与 _keyword_score 一致的高权重场景词。
KEYWORD_BONUS_TERMS = frozenset({"迟到", "早退", "旷工", "罚款", "处罚", "考勤"})
KEYWORD_CLOSE_WINDOW = 120

_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")

# 让「整篇归一化」与「逐窗口归一化」不再等价的字符（详见 _needs_exact_window_scan）。
# U+0130 拉丁大写 I 带点：唯一一个 lower() 会变长（1→2 字符）的码位；
# U+03A3 希腊大写 sigma：唯一一个 lower() 依赖上下文的码位（词尾折成 ς）。
_EXACT_WINDOW_SCAN_RE = re.compile("[İΣ]")

logger = logging.getLogger(__name__)


class _Bm25CacheEntry(NamedTuple):
    """One knowledge base's BM25 corpus statistics.

    The cheap change signature is ``(file_id, size)``. Knowledge files are
    write-once through the API (``crud.knowledge_file`` has no content-update path),
    so add/delete/replace always changes the id set or a size. An out-of-band edit
    that rewrites a row without changing either value is not detected; it can only
    come from outside this application, and the cost of detecting it would be
    re-reading and hashing every LONGTEXT column on every lookup.
    """

    signature: tuple[tuple[int, int], ...]
    index: BM25Index | None
    reason: str
    observed_docs: int
    observed_terms: int


class _BM25ScoringFailed(RuntimeError):
    """Internal signal: BM25 scoring failed, rerun the scan with legacy ordering."""


# Cache key: (stable database identity, knowledge_base_id, k1, b, max_docs, max_terms).
# Parameter values are part of the key: tweaking BM25_K1/B or either cap must rebuild,
# not silently keep serving an index built under the previous settings.
#
# The database identity is the rendered connection URL (host, port and database; the
# password is hidden), not ``id(bind)``: the latter is an address that CPython reuses
# after garbage collection, so a new engine could inherit a dead one's cached corpus.
# In-memory databases have no stable URL, so they fall back to the object identity --
# there is no cross-process meaning to share.
_BM25_CACHE: OrderedDict[tuple, _Bm25CacheEntry] = OrderedDict()
# Eight knowledge bases are kept. With more than eight active bases the LRU evicts the
# coldest one and its next query rebuilds the full corpus (a one-off full scan, not a
# correctness problem). Raising the limit trades that rebuild for memory: each entry
# holds a DF Counter of up to BM25_MAX_TERMS terms (~25MB at the default cap).
_BM25_CACHE_LIMIT = 8
_BM25_CACHE_LOCK = threading.Lock()
# Per-cache-key single-flight locks. N concurrent first questions for the same
# (database, knowledge base) build the corpus once; the others wait, then hit the
# cache. The dict is tiny (one lock per key ever built) and is deliberately not
# evicted: removing a lock while a waiter is blocked on it would let a third caller
# create a second lock and start a duplicate scan.
_BM25_BUILD_LOCKS: dict[tuple, threading.Lock] = {}
_BM25_BUILD_LOCKS_GUARD = threading.Lock()
# The keyword recall runs in a worker thread; the diagnostics dict is scoped to that
# call so concurrent retrievals never overwrite each other's trace.
_BM25_DIAGNOSTICS: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "bm25_diagnostics", default=None
)

# Background prewarm (issue #288 revision). The full-corpus scan is expensive enough
# that doing it inside the first chat request would add its whole latency to that
# answer. Uploads schedule the affected knowledge base here after their rows commit;
# startup schedules every base. One daemon worker keeps bursts of uploads from
# spawning one scan thread each, and the single-flight lock above still prevents a
# concurrent first question from duplicating the work.
_BM25_PREWARM_QUEUE: queue.Queue[int | None] = queue.Queue(maxsize=64)
_BM25_PREWARM_GUARD = threading.Lock()
_BM25_PREWARM_WORKER: threading.Thread | None = None


async def decide_need_rag(
    question: str,
    memory_context: str = "",
    knowledge_base_name: str = "",
    attachments: list[dict] | None = None,
) -> dict:
    attachments = attachments or []
    payload = {
        "question": ("" if question is None else str(question)).strip(),
        "memory_context": _clip(memory_context, 1800),
        "knowledge_base_name": "" if knowledge_base_name is None else str(knowledge_base_name),
        "attachments_count": len(attachments),
    }
    try:
        data = await call_router_json(payload)
        return _normalize_decision(data)
    except Exception as exc:
        # reason 会进 SSE 轨迹帧、retrieval_trace 与消息负载，异常原文只落日志。
        logger.warning("Route model call failed, falling back to RAG: %s", exc, exc_info=True)
        return _fallback_decision("路由模型调用失败，保守进入 RAG。")


async def build_query_plan(question: str) -> dict:
    system_prompt = (
        "你是企业知识库 RAG 检索规划器。只输出 JSON，不要输出 Markdown。"
        "字段必须是：hyde_document、rewrites、keywords。"
    )
    user_prompt = (
        "请为下面的用户问题生成："
        "1. 一段可能出现在企业文档里的假设答案文档 hyde_document；"
        "2. 3 个语义不同但意图一致的检索改写 rewrites；"
        "3. 不超过 8 个中文关键词 keywords。"
        f"\n\n用户问题：{question}"
    )
    try:
        data = await call_chat_json(system_prompt, user_prompt)
    except Exception as exc:
        # query_plan 会进 SSE 轨迹帧与 retrieval_trace；error 只作失败标记，不携带异常原文。
        logger.warning("Query plan build failed: %s", exc, exc_info=True)
        return {
            "hyde_document": "",
            "rewrites": [],
            "keywords": _fallback_keywords(question),
            "error": "查询规划失败",
        }
    return {
        "hyde_document": str(data.get("hyde_document") or "").strip(),
        "rewrites": _clean_list(data.get("rewrites"))[:3],
        "keywords": _merge_keywords(_clean_list(data.get("keywords")), question)[:24],
        "error": "",
    }


async def retrieve_knowledge(
    question: str,
    knowledge_base_id: int,
    db: Session | None,
    trace_recorder: Any = None,
    query_plan: dict | None = None,
) -> tuple[list[dict], dict]:
    await _trace_add(
        trace_recorder,
        "retriever_started",
        "retrieve_knowledge",
        params={"question": question, "knowledge_base_id": knowledge_base_id},
        note="开始执行 retrieve_knowledge，进入多路召回、融合和重排。",
    )
    query_plan = _normalize_external_query_plan(query_plan, question) if query_plan else await build_query_plan(question)
    trace = {
        "embedding": embedding_trace_status(embedding_backend_status()),
        "query_plan": query_plan,
        "routes": [],
        "rrf": [],
        "rerank": {"status": "skipped", "items": []},
        "bm25": _new_bm25_diagnostics(),
    }
    bm25_diagnostics = trace["bm25"]

    route_results: list[tuple[str, list[dict]]] = []
    route_specs = _build_route_specs(question, query_plan)

    keyword_terms = _merge_keywords(
        [
            *(query_plan.get("keywords") or []),
            *(query_plan.get("required_evidence") or []),
        ],
        question,
    )

    # query_vectors / keyword_recall 都是同步实现（阻塞式网络与数据库 IO），直接在协程里
    # 调用会独占事件循环，一轮多路检索期间整个进程无法处理其它请求。注意 asyncio.to_thread
    # 返回的是惰性 coroutine：只赋值不交给事件循环并不会让它先跑起来，所以所有召回（含关键字）
    # 必须放进同一个 gather，否则关键字召回仍会排在向量召回之后串行执行。
    # (route, query, 召回调用) 三元组把描述信息与调用绑在一起，结果按描述符配对，
    # 不依赖「关键词结果恒为列表最后一个元素」的位置约定。
    recall_specs: list[tuple[str, str, Any]] = [
        (
            route,
            query,
            asyncio.to_thread(
                query_vectors,
                query,
                top_k=RETRIEVAL_ROUTE_TOP_K,
                knowledge_base_id=knowledge_base_id,
                route=route,
            ),
        )
        for route, query in route_specs
    ]
    if keyword_terms:
        recall_specs.append(
            (
                "keyword",
                " ".join(keyword_terms),
                asyncio.to_thread(
                    _keyword_recall_step,
                    db,
                    knowledge_base_id,
                    keyword_terms,
                    RETRIEVAL_ROUTE_TOP_K,
                    bm25_diagnostics,
                ),
            )
        )

    # return_exceptions=True 是「并发」与「逐路降级」的交点：向量化后端不可用时只让该路拿到
    # EmbeddingBackendError，其余路与关键词路照常返回；同时 gather 会等全部召回结束才返回，
    # 不会留下仍在触碰请求级 Session 的孤儿线程。非 EmbeddingBackendError 的异常仍按原语义上抛。
    recall_results = await asyncio.gather(*(spec[2] for spec in recall_specs), return_exceptions=True)

    keyword_chunks: list[dict] = []
    for (route, query, _), result in zip(recall_specs, recall_results, strict=True):
        if isinstance(result, EmbeddingBackendError):
            # 查询向量化失败时绝不退化为哈希向量（会造成跨空间检索），
            # 显式跳过该路并记录原因，后续仍可用关键词路由召回。
            logger.warning("Vector route skipped, embedding backend unavailable: route=%s error=%s", route, result)
            # `embedding_error` 与 `embedding` 都会经 retrieval_trace 落到 assistant 消息，
            # 再由消息历史接口原样返回给用户；原文（上游 embedding 地址 + 原始异常）只进
            # 日志，用户侧给固定文案 + 可与日志对照的编号。
            trace["embedding_error"] = _internal_error_detail(
                EMBEDDING_UNAVAILABLE_MESSAGE, "retrieval_embedding", result
            )
            result = []
        elif isinstance(result, BaseException):
            raise result
        chunks: list[dict] = result
        # 关键词路复用固定名字 "keyword"（_build_route_specs 只产出
        # planned/simplified/sub_question_N/hyde/rewrite_N），按名字分流而非按下标。
        # 关键词路同样进入 route_results 参与 RRF 融合，并且始终排在向量路之后
        # （recall_specs 末尾追加），与两侧既有语义一致。
        if route == "keyword":
            keyword_chunks = chunks
        route_results.append((route, chunks))
        trace["routes"].append(
            {
                "route": route,
                "query": query,
                "count": len(chunks),
                "items": [trace_chunk(item) for item in chunks[:5]],
            }
        )

    if trace.get("embedding_error"):
        trace["embedding"] = embedding_trace_status(embedding_backend_status())

    fused = rrf_fuse(route_results)
    trace["rrf"] = [trace_chunk(item) for item in fused[:10]]
    ranking_question = query_plan.get("original_question") or question
    reranked, rerank_trace = await rerank_chunks(ranking_question, fused[:12])
    trace["rerank"] = rerank_trace
    final_chunks = select_final_chunks(reranked or fused, keyword_chunks)
    # 父子切片（issue #289）：RRF 与 rerank 始终作用于 child，聚合只发生在最终选择
    # 之后——同一 parent 的多个 child 在这里合并成一个上下文槽位。旧数据、元数据表
    # 不可用或开关关闭时原样返回 child，聊天链路不受影响。
    final_chunks, parent_child_trace = await expand_parent_chunks(final_chunks, db)
    trace["parent_child"] = parent_child_trace
    await _trace_add(
        trace_recorder,
        "retriever_done",
        "retrieve_knowledge",
        creates={
            "query_plan": query_plan,
            "routes": trace["routes"],
            "rrf": trace["rrf"],
            "rerank": rerank_trace,
            "bm25": trace["bm25"],
            "parent_child": parent_child_trace,
        },
        result={"final_chunks_count": len(final_chunks)},
        note="retrieve_knowledge 完成，最终 chunk 会进入回答生成链。",
    )
    return final_chunks, trace


def _normalize_external_query_plan(plan: dict | None, question: str) -> dict:
    if not isinstance(plan, dict):
        return {
            "hyde_document": "",
            "rewrites": [],
            "keywords": _fallback_keywords(question),
            "error": "external query_plan is invalid",
        }
    question_text = _text_value(question).strip()
    simplified_question = _text_value(plan.get("simplified_question")).strip() or question_text
    rewrites = _clean_list(plan.get("rewrites"))[:3]
    sub_questions = _clean_list(plan.get("sub_questions"))[:3]
    required_evidence = _clean_list(plan.get("required_evidence"))[:6]
    keywords = _merge_keywords(_clean_list(plan.get("keywords")), " ".join([question_text, simplified_question]))
    return {
        **plan,
        "original_question": _text_value(plan.get("original_question")).strip() or question_text,
        "simplified_question": simplified_question or question_text,
        "sub_questions": sub_questions,
        "hyde_document": _text_value(plan.get("hyde_document")).strip(),
        "rewrites": rewrites,
        "keywords": keywords[:24],
        "required_evidence": required_evidence,
        "error": _text_value(plan.get("error")),
    }


def _build_route_specs(question: str, query_plan: dict) -> list[tuple[str, str]]:
    route_specs: list[tuple[str, str]] = []
    _append_route(route_specs, "planned", question)
    simplified_question = _text_value(query_plan.get("simplified_question")).strip()
    if simplified_question and simplified_question != question:
        _append_route(route_specs, "simplified", simplified_question)
    for index, sub_question in enumerate(query_plan.get("sub_questions") or [], start=1):
        _append_route(route_specs, f"sub_question_{index}", sub_question)
    if query_plan.get("hyde_document"):
        _append_route(route_specs, "hyde", query_plan["hyde_document"])
    for index, rewrite in enumerate(query_plan.get("rewrites") or [], start=1):
        _append_route(route_specs, f"rewrite_{index}", rewrite)
    return route_specs


def _append_route(route_specs: list[tuple[str, str]], route: str, query: str) -> None:
    text = _text_value(query).strip()
    if not text:
        return
    normalized = _normalize_for_match(text)
    if any(_normalize_for_match(existing_query) == normalized for _, existing_query in route_specs):
        return
    route_specs.append((route, text))


def _keyword_recall_step(
    db: Session | None,
    knowledge_base_id: int,
    keywords: list[str],
    top_k: int,
    diagnostics: dict | None = None,
) -> list[dict]:
    """关键词召回这一步的会话归属（issue #187）。

    `db` 为 None 表示调用方没有请求级 Session 可以借用——`stream_chat` 在 #187 之后不再持有
    它，因为那个 Session 一路被 `db.query/commit/close` 用在事件循环线程上。这时这一步自带
    会话：`SessionLocal()` 与 `close()` 都发生在 `asyncio.to_thread` 的工作线程里，
    「开 session → 用 → 关」整段不跨线程。

    传了 `db` 的调用方（启动重建、用例里的真实库）沿用它原来的会话，行为不变。

    ``diagnostics`` 是本次关键词召回的 BM25 trace 槽位（issue #288）。用 ContextVar 传递
    而不是改 `keyword_recall` 的返回值：公开返回形状是既有契约，测试也会直接替换该函数。
    """
    token = _BM25_DIAGNOSTICS.set(diagnostics) if diagnostics is not None else None
    try:
        if db is not None:
            return keyword_recall(db, knowledge_base_id, keywords, top_k)
        own_session = SessionLocal()
        try:
            return keyword_recall(own_session, knowledge_base_id, keywords, top_k)
        finally:
            own_session.close()
    finally:
        if token is not None:
            _BM25_DIAGNOSTICS.reset(token)


def keyword_recall(db: Session, knowledge_base_id: int, keywords: list[str], top_k: int) -> list[dict]:
    """关键词召回：成本与「命中的内容量」相关，而不是与知识库总量相关（issue #59）。

    旧实现每次提问都把整库文件的 LONGTEXT ``content`` 全量读进内存、逐文件整篇归一化、
    再逐窗口重新归一化，最后只返回 ``top_k`` 条——内存与耗时随知识库总量线性膨胀，而
    ``top_k`` 只截断输出、不截断输入。这里把成本分三层收敛：

    * 取数层 ``_iter_keyword_candidate_files``：SQL 侧先按关键词做 LIKE 预筛，再按主键
      游标分批（带 ``LIMIT``）取数，单批内存有上界且与知识库文件总数无关；
    * 文件层：每个文件只做一次整篇归一化，用它精确判定「该文件是否可能命中」；
    * 窗口层 ``_collect_file_keyword_candidates``：窗口在归一化文本上的区间由空白映射推出，
      再在整篇归一化文本上带区间边界统计命中，不再物化窗口文本、也不逐窗口重新归一化；
      候选只保留分值最高的 ``top_k`` 条（有界堆），不再为每个窗口各留一个字典。

    返回值与旧实现逐条一致（字段、顺序、分值），等价性由
    tests/test_keyword_recall_memory_59.py 中冻结的旧实现做回归基线。

    唯一的例外是含 ``İ``/``Σ`` 的正文（``_needs_exact_window_scan``）：这两个字符让
    「整篇归一化后取区间」不再等于「对窗口切片单独归一化」，该文件回退到与旧实现逐行同构的
    逐窗口路径（``_collect_file_keyword_candidates_exact``）。回退按文件粒度触发，中文语料
    不会命中，取数层的批上界也不受影响。

    issue #288：``BM25_ENABLED`` 打开时，窗口排序键换成 BM25（``bm25_score``），但
    ``keyword_score`` 字段与旧打分函数原样保留；关闭或 BM25 构建/打分失败时完全退回旧排序。
    ``bm25_rank`` 是最终关键词候选的 1-based 名次。
    """
    diagnostics = _BM25_DIAGNOSTICS.get()
    if top_k <= 0:
        _set_bm25_diagnostics(diagnostics, status="skipped", reason="top_k_non_positive")
        return []
    clean_keywords = _expand_keywords(keywords)
    if not clean_keywords:
        _set_bm25_diagnostics(diagnostics, status="skipped", reason="no_keywords")
        return []
    case_fold = _needs_case_fold(clean_keywords)
    prepared_keywords = _prepare_keywords(clean_keywords, case_fold)
    if not prepared_keywords:
        _set_bm25_diagnostics(diagnostics, status="skipped", reason="no_prepared_keywords")
        return []

    index: BM25Index | None = None
    terms: tuple[str, ...] = ()
    cache_hit = False
    if not BM25_ENABLED:
        _set_bm25_diagnostics(diagnostics, status="disabled")
    else:
        terms = bm25_query_terms(" ".join(clean_keywords))
        if not terms:
            # Nothing BM25 can match (e.g. a script the tokenizer does not cover).
            # Keep the legacy ordering instead of ranking every window at zero.
            _set_bm25_diagnostics(diagnostics, status="fallback", reason="no_query_tokens")
        elif not _keyword_candidates_exist(db, knowledge_base_id, clean_keywords, case_fold):
            # The LIKE pre-filter is a necessary condition, so zero rows means zero
            # matches. Do not pay for a full-corpus BM25 build just to return [].
            _set_bm25_diagnostics(diagnostics, status="skipped", reason="no_candidates")
            return []
        else:
            try:
                index, cache_hit, reason, observed_docs, observed_terms = _bm25_index_for(
                    db, knowledge_base_id
                )
            except Exception as exc:
                # 异常原文只进日志；trace 只记录固定原因码（沿用 #263 的护栏）。
                logger.warning(
                    "BM25 index build failed, falling back to legacy keyword scoring: %s",
                    exc,
                    exc_info=True,
                )
                _set_bm25_diagnostics(
                    diagnostics, status="fallback", degraded=True, reason="index_build_failed"
                )
            else:
                if index is None:
                    degraded = reason in {"corpus_too_large", "vocabulary_too_large"}
                    _set_bm25_diagnostics(
                        diagnostics,
                        status="degraded" if degraded else "fallback",
                        degraded=degraded,
                        reason=reason,
                        corpus_docs=observed_docs,
                        corpus_terms=observed_terms,
                    )

    budget = max(top_k, BM25_MAX_SCORED_CANDIDATES) if index is not None else top_k
    try:
        chunks, candidates, scored, budget_exceeded = _keyword_recall_ranked(
            db,
            knowledge_base_id,
            clean_keywords,
            prepared_keywords,
            case_fold,
            top_k,
            index,
            terms,
            budget,
        )
    except _BM25ScoringFailed:
        logger.warning("BM25 scoring failed, falling back to legacy keyword scoring", exc_info=True)
        index = None
        budget = top_k
        _set_bm25_diagnostics(
            diagnostics, status="fallback", degraded=True, reason="scoring_failed"
        )
        # Re-scan with the legacy rank: the first pass may already have dropped
        # candidates from its BM25-ordered heap, so reusing it would not be a true
        # fallback.
        chunks, candidates, scored, budget_exceeded = _keyword_recall_ranked(
            db,
            knowledge_base_id,
            clean_keywords,
            prepared_keywords,
            case_fold,
            top_k,
            None,
            (),
            budget,
        )

    if index is not None:
        _set_bm25_diagnostics(
            diagnostics,
            status="ok",
            degraded=False,
            reason="",
            corpus_docs=index.doc_count,
            corpus_terms=index.term_count,
            cache_hit=cache_hit,
        )
    _set_bm25_diagnostics(
        diagnostics,
        candidates=candidates,
        scored_candidates=scored,
        candidate_budget=budget,
        scoring_budget_exceeded=budget_exceeded,
    )
    return chunks


def _keyword_recall_ranked(
    db: Session,
    knowledge_base_id: int,
    clean_keywords: list[str],
    prepared_keywords: list[tuple[str, str, float]],
    case_fold: bool,
    top_k: int,
    index: BM25Index | None,
    terms: tuple[str, ...],
    budget: int,
) -> tuple[list[dict], int, int, bool]:
    """Scan candidates and return ``(top_k chunks, candidates, scored, truncated)``.

    When BM25 is active the scan does **not** tokenise every matching window. Windows
    are first pre-ranked by a cheap proxy (``_bm25_cheap_upper``) and only the best
    ``budget`` of them get a full BM25 score, so the keyword-route cost is bounded by
    the budget instead of the number of ``LIKE`` matches. ``truncated`` says whether
    the scan saw more matches than the budget. On truncation the windows outside the
    retained heap are discarded and the returned top-k is an approximate ordering;
    the trace's ``scoring_budget_exceeded`` makes that visible instead of silent.
    """
    best: list[tuple[tuple[float, float], int, dict]] = []
    order = 0
    bm25 = None
    if index is not None:
        # IDF is corpus-level, so look it up once per query term instead of once per
        # candidate window: ``idf()`` is a dict lookup plus a log, and the scan may
        # visit thousands of windows.
        bm25 = (index, tuple((term, index.idf(term)) for term in terms))
    for file_id, file_name, content in _iter_keyword_candidate_files(
        db, knowledge_base_id, clean_keywords, case_fold
    ):
        order = _collect_file_keyword_candidates(
            file_id,
            file_name,
            content,
            clean_keywords,
            prepared_keywords,
            case_fold,
            budget,
            best,
            order,
            bm25,
        )
    if index is None:
        # 与旧实现 candidates.sort(key=keyword_score, reverse=True)[:top_k] 等价：Python 排序稳定，
        # 同分候选保持「文件主键序、窗口起点序」；这里用负的 rank key 复刻同一顺序。
        final = [item[2] for item in sorted(best, key=_candidate_sort_key)]
        return final, order, 0, False

    # Full BM25 score, exactly once per retained window (at most ``budget``).
    scored: list[tuple[float, float, int, dict]] = []
    for _cheap, neg_order, candidate in best:
        score = _score_bm25((index, terms), candidate["content"])
        # ``_score_bm25`` raises instead of returning None for the non-None context.
        assert score is not None
        candidate["bm25_score"] = score
        scored.append((score, candidate["keyword_score"], neg_order, candidate))
    # ``neg_order`` is ``-order``; negating again restores the earlier-first order.
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    final = [item[3] for item in scored[:top_k]]
    for rank, candidate in enumerate(final, start=1):
        candidate["bm25_rank"] = rank
    return final, order, len(scored), order > budget


def _candidate_sort_key(item: tuple[tuple[float, float], int, dict]) -> tuple[float, float, int]:
    rank_key, neg_order, _candidate = item
    return (-rank_key[0], -rank_key[1], -neg_order)


def _new_bm25_diagnostics() -> dict:
    """Trace shape for the BM25 stage (issue #288).

    The trace carries parameters, corpus/candidate sizes and a fixed reason code.
    It never carries an exception message from the index/scoring path.
    """
    return {
        "enabled": bool(BM25_ENABLED),
        "k1": float(BM25_K1),
        "b": float(BM25_B),
        "max_docs": int(BM25_MAX_DOCS),
        "max_terms": int(BM25_MAX_TERMS),
        "candidate_budget": int(BM25_MAX_SCORED_CANDIDATES),
        "status": "skipped",
        "candidates": 0,
        "scored_candidates": 0,
        # True means the pre-ranking heap was truncated: only ``candidate_budget``
        # windows got a full BM25 score and the returned top-k is approximate.
        "scoring_budget_exceeded": False,
        "corpus_docs": 0,
        "corpus_terms": 0,
        "cache_hit": False,
        "degraded": False,
        "reason": "",
    }


def _set_bm25_diagnostics(diagnostics: dict | None, **fields) -> None:
    if diagnostics is None:
        return
    diagnostics.update(fields)


def clear_bm25_cache() -> None:
    """Drop all cached BM25 corpus statistics (tests and explicit invalidation)."""
    with _BM25_CACHE_LOCK:
        _BM25_CACHE.clear()


def schedule_bm25_prewarm(knowledge_base_id: int | None = None) -> None:
    """Queue a background corpus build so the first question does not pay for it.

    Uploads call this after their row commits; startup calls it with no id to warm
    every knowledge base. ``None`` means "all". A full queue drops the item: the cache
    still builds on demand, so a dropped prewarm only means the next query pays the
    scan itself.
    """
    if not BM25_ENABLED:
        return
    _ensure_bm25_prewarm_worker()
    try:
        _BM25_PREWARM_QUEUE.put_nowait(knowledge_base_id)
    except queue.Full:
        logger.debug("BM25 prewarm queue is full; leaving the build to the next query")


def _ensure_bm25_prewarm_worker() -> None:
    global _BM25_PREWARM_WORKER
    with _BM25_PREWARM_GUARD:
        if _BM25_PREWARM_WORKER is not None and _BM25_PREWARM_WORKER.is_alive():
            return
        worker = threading.Thread(
            target=_bm25_prewarm_worker, name="bm25-prewarm", daemon=True
        )
        _BM25_PREWARM_WORKER = worker
        worker.start()


def _bm25_prewarm_worker() -> None:
    while True:
        knowledge_base_id = _BM25_PREWARM_QUEUE.get()
        try:
            _prewarm_bm25(knowledge_base_id)
        except Exception as exc:
            # 后台预热失败只影响「首问是否已热」，检索路径会自己重建并记录降级；
            # 异常原文只进日志。
            logger.warning(
                "BM25 prewarm failed: knowledge_base_id=%s error=%s",
                knowledge_base_id,
                exc,
                exc_info=True,
            )
        finally:
            _BM25_PREWARM_QUEUE.task_done()


def _prewarm_bm25(knowledge_base_id: int | None) -> None:
    """Build one knowledge base's corpus, or every base when the id is ``None``.

    Warming more than ``_BM25_CACHE_LIMIT`` bases leaves only the last ones in the
    LRU; the earlier ones are rebuilt on their next query (documented in .env.example).
    """
    if not BM25_ENABLED:
        return
    db = SessionLocal()
    try:
        if knowledge_base_id is not None:
            _bm25_index_for(db, knowledge_base_id)
            return
        rows = (
            db.query(KnowledgeFile.knowledge_base_id)
            .filter(KnowledgeFile.knowledge_base_id.isnot(None))
            .distinct()
            .all()
        )
        for (kb_id,) in rows:
            if not BM25_ENABLED:
                return
            _bm25_index_for(db, int(kb_id))
    finally:
        db.close()


def _bm25_cache_scope(db: Session) -> str:
    """Stable identity shared by sessions of the same database.

    The rendered URL (host, port, database, password hidden) is stable across
    sessions and process lifetimes.
    ``id(bind)`` is not: it is an address, and CPython reuses addresses after garbage
    collection, so a brand-new engine could inherit a dead one's corpus. In-memory
    databases have no stable URL, so they fall back to object identity -- there is no
    cross-connection meaning to share anyway.
    """
    try:
        bind = db.get_bind()
    except Exception:
        bind = None
    if bind is None:
        return f"session:{id(db)}"
    url = getattr(bind, "url", None)
    if url is None:
        return f"bind:{id(bind)}"
    database = url.database
    if database and database != ":memory:":
        # Render the whole URL (password hidden): host, port and database all
        # distinguish deployments, and a shared database name on two hosts must not
        # share a corpus.
        return url.render_as_string(hide_password=True)
    return f"{url.drivername}:memory:{id(bind)}"


def _bm25_cache_key(scope: str, knowledge_base_id: int) -> tuple:
    """Cache key including every parameter that changes the resulting scores."""
    return (
        scope,
        int(knowledge_base_id),
        float(BM25_K1),
        float(BM25_B),
        int(BM25_MAX_DOCS),
        int(BM25_MAX_TERMS),
    )


def _bm25_build_lock(cache_key: tuple) -> threading.Lock:
    """Per-key single-flight lock (see ``_BM25_BUILD_LOCKS``)."""
    with _BM25_BUILD_LOCKS_GUARD:
        lock = _BM25_BUILD_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _BM25_BUILD_LOCKS[cache_key] = lock
        return lock


def _cached_bm25_entry(cache_key: tuple, signature: tuple) -> tuple | None:
    """Return the cached result if the signature still matches, else ``None``."""
    with _BM25_CACHE_LOCK:
        cached = _BM25_CACHE.get(cache_key)
        if cached is None or cached.signature != signature:
            return None
        _BM25_CACHE.move_to_end(cache_key)
        return (
            cached.index,
            True,
            cached.reason,
            cached.observed_docs,
            cached.observed_terms,
        )


def _bm25_signature(db: Session, knowledge_base_id: int) -> tuple[tuple[int, int], ...]:
    """Cheap change signal for a knowledge base: ``(file_id, size)`` per file.

    Knowledge files are create/delete-only in this application (``crud.knowledge_file``
    has no content-update path), so this pair changes on every add/delete/replace.
    The full content digests are computed once when the index is built and stored in
    the cache entry; re-reading every LONGTEXT column just to hash it would defeat
    the point of caching the corpus scan.
    """
    rows = (
        db.query(KnowledgeFile.id, KnowledgeFile.size)
        .filter(KnowledgeFile.knowledge_base_id == knowledge_base_id)
        .order_by(KnowledgeFile.id)
        .all()
    )
    return tuple((int(row[0]), int(row[1] or 0)) for row in rows)


def _iter_knowledge_base_contents(
    db: Session, knowledge_base_id: int
) -> Iterator[tuple[int, str]]:
    """Stream every file's content with the same paged/adaptive batching as recall."""
    last_id = 0
    batch_rows = KEYWORD_RECALL_BATCH_SIZE
    while True:
        rows = (
            db.query(KnowledgeFile.id, KnowledgeFile.content)
            .filter(
                KnowledgeFile.knowledge_base_id == knowledge_base_id,
                KnowledgeFile.id > last_id,
            )
            .order_by(KnowledgeFile.id)
            .limit(batch_rows)
            .all()
        )
        if not rows:
            return
        batch_chars = 0
        for file_id, content in rows:
            content = content or ""
            batch_chars += len(content)
            yield int(file_id), content
        last_id = int(rows[-1][0])
        average_chars = max(batch_chars // len(rows), 1)
        batch_rows = max(
            1, min(KEYWORD_RECALL_BATCH_ROWS_MAX, KEYWORD_RECALL_BATCH_CHARS // average_chars)
        )


def _bm25_window_texts(content: str) -> Iterator[str]:
    """Same ``KEYWORD_CHUNK_SIZE``/overlap windows the keyword route retrieves.

    DOCX 正文里可能带着抽取阶段的表格边界哨兵；索引/打分都必须先把哨兵剥掉，否则
    ``__RAG_TABLE_BOUNDARY__`` 会成为一个文档频率很高的伪词项，NUL 也会进入窗口正文。
    """
    content = strip_extraction_markers(content)
    for start in _keyword_chunk_starts(len(content)):
        text = content[start : start + KEYWORD_CHUNK_SIZE].strip()
        if text:
            yield text


def _bm25_index_for(
    db: Session, knowledge_base_id: int
) -> tuple[BM25Index | None, bool, str, int, int]:
    """Return ``(index, cache_hit, reason, observed_docs, observed_terms)``.

    DF is accumulated over **all** retrievable windows; there is deliberately no
    fallback to the current query's ``LIKE`` candidates. A cap breach (windows or
    distinct terms) returns ``index=None`` with a fixed reason so the caller records
    an explicit degradation and keeps the legacy ordering.

    Concurrent first lookups for the same key are serialised: the first builds, the
    rest wait and then read the cache. Without that lock, N simultaneous first
    questions would each scan the whole knowledge base.
    """
    signature = _bm25_signature(db, knowledge_base_id)
    cache_key = _bm25_cache_key(_bm25_cache_scope(db), knowledge_base_id)
    cached = _cached_bm25_entry(cache_key, signature)
    if cached is not None:
        return cached

    with _bm25_build_lock(cache_key):
        # Re-read after waiting: another thread may have built it, and an upload may
        # have committed while we waited, which changes the signature.
        signature = _bm25_signature(db, knowledge_base_id)
        cached = _cached_bm25_entry(cache_key, signature)
        if cached is not None:
            return cached

        index, reason, observed_docs, observed_terms = _build_bm25_index(
            db, knowledge_base_id
        )
        entry = _Bm25CacheEntry(
            signature=signature,
            index=index,
            reason=reason,
            observed_docs=observed_docs,
            observed_terms=observed_terms,
        )
        with _BM25_CACHE_LOCK:
            _BM25_CACHE[cache_key] = entry
            _BM25_CACHE.move_to_end(cache_key)
            while len(_BM25_CACHE) > _BM25_CACHE_LIMIT:
                _BM25_CACHE.popitem(last=False)
        if reason:
            # A cap breach is a permanent degradation for this knowledge base until
            # its files change. Make it operator-visible instead of trace-only.
            logger.warning(
                "BM25 corpus degraded for knowledge_base_id=%s reason=%s "
                "observed_docs=%s observed_terms=%s; the keyword route keeps legacy ordering",
                knowledge_base_id,
                reason,
                observed_docs,
                observed_terms,
            )
        return index, False, reason, observed_docs, observed_terms


def _build_bm25_index(
    db: Session, knowledge_base_id: int
) -> tuple[BM25Index | None, str, int, int]:
    """Scan one knowledge base into corpus statistics, or report why it was capped."""
    observed_docs = 0

    def documents() -> Iterator[str]:
        nonlocal observed_docs
        for _file_id, content in _iter_knowledge_base_contents(db, knowledge_base_id):
            for text in _bm25_window_texts(content):
                observed_docs += 1
                yield text

    try:
        index = BM25Index(
            documents(),
            k1=BM25_K1,
            b=BM25_B,
            max_docs=BM25_MAX_DOCS,
            max_terms=BM25_MAX_TERMS,
        )
        return index, "", index.doc_count, index.term_count
    except BM25CorpusTooLarge as exc:
        return None, "corpus_too_large", exc.doc_count, 0
    except BM25VocabularyTooLarge as exc:
        return None, "vocabulary_too_large", observed_docs, exc.term_count


def _keyword_candidates_exist(
    db: Session, knowledge_base_id: int, clean_keywords: list[str], case_fold: bool
) -> bool:
    """Cheap existence probe before paying for a full-corpus BM25 build.

    The SQL pre-filter only over-selects (it is a necessary condition), so zero rows
    means zero real candidates and the index would be built for nothing. When the
    pre-filter cannot be pushed down this only checks that the knowledge base has any
    file at all, which is still cheaper than the corpus scan.
    """
    query = db.query(KnowledgeFile.id).filter(
        KnowledgeFile.knowledge_base_id == knowledge_base_id
    )
    prefilter = _sql_prefilter_patterns(clean_keywords, case_fold)
    if prefilter is not None:
        column = func.lower(KnowledgeFile.content) if case_fold else KnowledgeFile.content
        query = query.filter(
            or_(*[column.like(pattern, escape=LIKE_ESCAPE) for pattern in prefilter])
        )
    return bool(query.limit(1).all())


def _bm25_cheap_upper(idf_terms: tuple[tuple[str, float], ...], window_norm: str) -> float:
    """Cheap pre-ranking proxy for a window's BM25 score, without tokenising it.

    For every query term whose NFKC spelling occurs in the window, add its IDF. This
    skips tokenisation, term frequency and length normalisation, so it is a proxy for
    the real score, **not** a tight mathematical bound and not a ranking guarantee.

    It is only used to choose which ``BM25_MAX_SCORED_CANDIDATES`` windows get a full
    ``score_text``. When the budget is exceeded, windows outside the retained heap are
    dropped after this pre-ranking, so the returned top-k is an approximation rather
    than an exact BM25 ordering. ``scoring_budget_exceeded`` in the trace records that
    truncation; it must not be read as "no candidates were lost".
    """
    total = 0.0
    for term, weight in idf_terms:
        if weight > 0 and term in window_norm:
            total += weight
    return total


def _score_bm25(bm25: tuple[BM25Index, tuple[str, ...]] | None, content: str) -> float | None:
    """Score one candidate window; ``None`` means the legacy path is active."""
    if bm25 is None:
        return None
    index, terms = bm25
    try:
        score = float(index.score_text(content, terms))
    except Exception as exc:
        raise _BM25ScoringFailed() from exc
    if not math.isfinite(score):
        raise _BM25ScoringFailed()
    return score


def _iter_keyword_candidate_files(
    db: Session, knowledge_base_id: int, clean_keywords: list[str], case_fold: bool = True
) -> Iterator[tuple[int, str, str]]:
    """按主键游标分批产出 (file_id, name, content)，只取「可能命中」的文件。

    两个上界同时生效：SQL 侧 ``LIMIT`` 限制单批行数，``id > last_id`` 只向前扫描（不会
    重复扫已经处理过的行）；预筛推不下去时退化为整库分批扫描，内存上界不变。

    批次大小按上一批的实际体积调整，让「单批累计字符数」落在预算附近：小文件书库少几次
    往返，大文件书库把峰值压在预算量级（单个文件本身就超预算时无法再小）。
    """
    prefilter = _sql_prefilter_patterns(clean_keywords, case_fold)
    last_id = 0
    batch_rows = KEYWORD_RECALL_BATCH_SIZE
    while True:
        query = db.query(KnowledgeFile.id, KnowledgeFile.name, KnowledgeFile.content).filter(
            KnowledgeFile.knowledge_base_id == knowledge_base_id,
            KnowledgeFile.id > last_id,
        )
        if prefilter is not None:
            # 关键词含字母时先在库侧折叠大小写，再交给 LIKE；纯汉字关键词不必多做一遍。
            column = func.lower(KnowledgeFile.content) if case_fold else KnowledgeFile.content
            query = query.filter(or_(*[column.like(pattern, escape=LIKE_ESCAPE) for pattern in prefilter]))
        rows = query.order_by(KnowledgeFile.id).limit(batch_rows).all()
        if not rows:
            return
        batch_chars = 0
        for file_id, file_name, content in rows:
            batch_chars += len(content or "")
            yield int(file_id), file_name, content or ""
        last_id = int(rows[-1][0])
        average_chars = max(batch_chars // len(rows), 1)
        batch_rows = max(
            1, min(KEYWORD_RECALL_BATCH_ROWS_MAX, KEYWORD_RECALL_BATCH_CHARS // average_chars)
        )


def _sql_prefilter_patterns(clean_keywords: list[str], case_fold: bool = True) -> list[str] | None:
    """SQL 侧 LIKE 预筛模式（命中结果的必要条件），None 表示只能整库分批扫描。

    关键词匹配的语义是「去掉空白 + 忽略大小写后的子串匹配」，SQL 里没法逐字复刻。这里只
    要求「各字符按序出现」（``%`` 夹住每个字符）：被匹配掉的空白可能出现在正文任意位置，
    所以不能要求字面子串——正文写成「考 假」时去掉空白才等于「考假」，字面 LIKE 会漏掉
    它。字面版只是这条必要条件的一个特例，为不漏召回一律用按序版。

    真正命中的行一定过筛，因此不会漏召回，只可能多取几行；多取的行由 Python 侧的精确
    判定再筛掉。

    含非 ASCII 大小写字母的关键词放弃下推：SQLite 的 ``LOWER``/``LIKE`` 只折叠 ASCII，
    下推可能漏掉正文用另一种大小写写法的命中——宁可慢，不可漏。
    """
    if case_fold and any(not _sql_case_foldable(keyword) for keyword in clean_keywords):
        return None
    patterns = []
    for keyword in clean_keywords:
        normalized = _normalize_for_match(keyword, case_fold)
        if len(normalized) < 2:
            continue
        pattern = _like_necessary_pattern(normalized, case_fold)
        if pattern is None:
            # 该关键词的预筛退化成「任意一行」，整条预筛也就没有意义了：宁可整库分批扫。
            return None
        patterns.append(pattern)
    return patterns or None


def _like_necessary_pattern(normalized_keyword: str, case_fold: bool = True) -> str | None:
    """单个「各字符按序出现」的 LIKE 模式；退化时返回 None（调用方放弃整条预筛）。

    含大小写折叠时，有两条字符在 Python 里另有来源、而 SQL 的 ``LOWER`` 折不到：
    ``'i'`` 还能来自 ``U+0130``（İ，折成 ``i`` + 组合点），``'k'`` 还能来自
    ``U+212A``（KELVIN SIGN）。要求字面出现会让这些行在取数层就被丢掉——旧实现没有
    预筛、能召回，就是本 PR 引入的漏召回。这类位置改用一个单字符通配 ``_`` 顶替：
    ``_`` 匹配任意单字符，所以真命中仍然一定过筛（预筛只是筛得更松），而 SQLite 与
    MySQL 的 LIKE 都支持。全字符都被顶替时模式变成「任意非空行」，返回 None。
    """
    if any(char in _MULTI_CHAR_FOLD_TARGETS for char in normalized_keyword):
        return None
    pieces = []
    substituted = 0
    for char in normalized_keyword:
        if case_fold and char in _UNFOLDED_CASE_SOURCES:
            pieces.append("_")
            substituted += 1
        else:
            pieces.append(_escape_like_char(char))
    if substituted == len(normalized_keyword):
        return None
    return "%" + "%".join(pieces) + "%"


def _sql_case_foldable(keyword: str) -> bool:
    """关键词能否在 SQL 侧安全比较大小写：ASCII 交给 LOWER/LIKE，无大小写的字符原样比较。

    这里只管「库侧折叠强度够不够」：SQLite 的 ``LOWER`` 只折叠 ASCII，西里尔/希腊文这类
    非 ASCII 大小写字母折不动，所以含它们的关键词整体放弃下推（见 _sql_prefilter_patterns）。
    至于「Python 折得到、SQL 折不到」的那两处非 ASCII 来源（İ / KELVIN SIGN），由
    ``_like_necessary_pattern`` 用单字符通配 ``_`` 顶替，不靠本函数兜底——本函数返回 True
    的关键词同样不会漏召回。
    """
    return all(char.isascii() or char.lower() == char.upper() for char in keyword)


def _escape_like_char(char: str) -> str:
    # 反斜杠一并转义：MySQL 的习惯是把 \ 当默认转义符，`\%` 会被当成字面百分号，
    # 正好会把我插入的通配符吃掉（预筛变窄、可能漏召回）。
    if char == LIKE_ESCAPE or char in {"%", "_", "\\"}:
        return LIKE_ESCAPE + char
    return char


def _needs_case_fold(clean_keywords: list[str]) -> bool:
    """关键词里是否含有大小写字母。

    全是汉字/数字/标点时不需要折叠大小写：这类字符的大小写形式就是它自己，正文里能命中
    的位置与折叠后完全相同。省掉整篇 ``str.lower()`` 很值——它对大字符串会临时放大数倍
    内存（实测 200 万字符约 27MB 峰值），而中文语料的提问关键词几乎都不含字母。
    """
    return any(char.lower() != char.upper() for keyword in clean_keywords for char in keyword)


def _prepare_keywords(clean_keywords: list[str], case_fold: bool = True) -> list[tuple[str, str, float]]:
    """(原始关键词, 归一化关键词, 权重)，权重规则与 ``_keyword_score`` 保持一致。"""
    prepared = []
    for keyword in clean_keywords:
        normalized = _normalize_for_match(keyword, case_fold)
        if len(normalized) < 2:
            continue
        weight = 1.0
        if _DIGIT_RE.search(normalized):
            weight += 3.0
        if len(normalized) >= 4:
            weight += 2.0
        if normalized in KEYWORD_BONUS_TERMS:
            weight += 4.0
        prepared.append((keyword, normalized, weight))
    return prepared


def _needs_exact_window_scan(content: str) -> bool:
    """该文件的窗口命中必须逐窗口重新归一化，不能走「整篇归一化 + 区间统计」的快路径。

    快路径成立的前提是 ``str.lower()`` 逐字符可组合：``normalize(content[a:b])`` 等于
    ``normalize(content)[rank(a):rank(b)]``。有两个码位打破它（已全码位扫描确认各自唯一）：

    * ``U+0130``（İ）是 Python 中唯一个 ``lower()`` 会变长的码位（1→2）。整篇归一化后
      文本变长，而 ``_NormalizedOffsetMap`` 按字符计数，窗口右边界会整体偏小，文件尾部
      ``İ 的个数`` 个字符不再落在任何窗口里，命中直接丢失；
    * ``U+03A3``（Σ）是唯一个 ``lower()`` 依赖上下文的码位，词尾折成 ``ς``、其余折成
      ``σ``。整篇归一化拿到的是全文上下文，窗口切片拿到的是窗口内上下文，同一段文本在
      两条路径上会得到不同的归一化结果（命中集合与分值都会变）。

    这两类字符在中文语料里不出现，所以按文件粒度回退的代价可以忽略；判据是「正文是否含
    这两个字符」，与关键词无关（关键词不含希腊字母时正文里的 Σ 同样会改变窗口边界）。
    """
    return _EXACT_WINDOW_SCAN_RE.search(content) is not None


def _collect_file_keyword_candidates(
    file_id: int,
    file_name: str,
    content: str,
    clean_keywords: list[str],
    prepared_keywords: list[tuple[str, str, float]],
    case_fold: bool,
    budget: int,
    best: list[tuple[tuple[float, float], int, dict]],
    order: int,
    bm25: tuple[BM25Index, tuple[tuple[str, float], ...]] | None = None,
) -> int:
    """把单个文件里命中的窗口并入有界堆，返回更新后的候选序号。

    ``budget`` 在 BM25 开启时是「最多对多少个窗口做完整打分」的预算（`top_k` 与
    ``BM25_MAX_SCORED_CANDIDATES`` 的较大值），关闭时就是 ``top_k``。窗口先按廉价代理
    预排序入堆，完整 BM25 只在扫描结束后对堆内窗口做一次（见 ``_keyword_recall_ranked``）。
    触顶时堆外窗口会被截断，最终 top-k 是近似排序。
    """
    # 候选正文会原样返回给重排/回答链路；抽取哨兵只服务于入库切片，不能从这里泄漏。
    content = strip_extraction_markers(content)
    if _needs_exact_window_scan(content):
        return _collect_file_keyword_candidates_exact(
            file_id, file_name, content, clean_keywords, budget, best, order, bm25
        )
    normalized_content = _normalize_for_match(content, case_fold)
    occurrences = []
    for keyword, normalized, weight in prepared_keywords:
        if normalized in normalized_content:
            occurrences.append((keyword, normalized, weight))
    if not occurrences:
        return order
    starts = _NormalizedOffsetMap(content, len(normalized_content))
    ends = _NormalizedOffsetMap(content, len(normalized_content))
    for raw_start in _keyword_chunk_starts(len(content)):
        raw_end = min(raw_start + KEYWORD_CHUNK_SIZE, len(content))
        window_start = starts.rank(raw_start)
        window_end = ends.rank(raw_end)
        hits = []
        for keyword, normalized, weight in occurrences:
            # 归一化后的窗口就是归一化全文的一段（去空白是逐字符过滤，前缀仍映射到前缀），
            # 所以直接在全文上带 [window_start, window_end) 边界统计即可，既不用把窗口文本
            # 物化出来，也不用重新归一化。位置用全文坐标：打分只看首次位置的相对间隔与是否
            # 互不相同，整体平移不影响分值。
            count = normalized_content.count(normalized, window_start, window_end)
            if count > 0:
                hits.append(
                    (keyword, normalized, weight, count, normalized_content.find(normalized, window_start, window_end))
                )
        if not hits:
            # 窗口内没有任何关键词出现：旧实现下该窗口得 0 分且命中集为空，必然落选。
            continue
        score, matched_keywords = _score_from_hits(hits)
        if score <= 0 or not matched_keywords:
            continue
        chunk_content = content[raw_start:raw_end].strip()
        if not chunk_content:
            continue
        order += 1
        chunk_id = str(raw_start)
        candidate = {
            "id": f"{file_id}_{chunk_id}",
            "chunk_id": chunk_id,
            "content": chunk_content,
            "file_name": file_name,
            "file_id": file_id,
            "route": "keyword",
            "keyword_score": score,
            # 命中证据随候选一起带出，供 select_final_chunks 校验（issue #56）。
            "keyword_hits": len(matched_keywords),
            "matched_keywords": matched_keywords,
        }
        if bm25 is None:
            rank_key = (score, 0.0)
        else:
            # ``normalized_content`` 已去空白并折小写；NFKC 让全角写法与 ``tokenize``
            # 的查询词项对齐。只对命中窗口做，未命中窗口不付这份成本。
            window_norm = unicodedata.normalize(
                "NFKC", normalized_content[window_start:window_end]
            )
            rank_key = (_bm25_cheap_upper(bm25[1], window_norm), score)
        _keep_best_candidate(best, budget, order, rank_key, candidate)
    return order


def _keep_best_candidate(
    best: list[tuple[tuple[float, float], int, dict]],
    budget: int,
    order: int,
    rank_key: tuple[float, float],
    candidate: dict,
) -> None:
    """有界堆：只保留排序键最高的 ``budget`` 条，同分保留更早出现的候选。

    ``rank_key`` 在 BM25 开启时是 ``(廉价代理分, keyword_score)``（完整 BM25 分数要等扫描
    结束后再算，见 ``_keyword_recall_ranked``），关闭时是 ``(keyword_score, 0.0)``。
    堆顶是「最差」条目（排序键最低；同分时 -order 最小即出现最晚），因此新候选只要比堆顶
    更好就替换它。这样内存只与预算相关，不再为每个窗口都留一个候选字典。
    """
    entry = (rank_key, -order, candidate)
    if len(best) < budget:
        heapq.heappush(best, entry)
    elif entry[:2] > best[0][:2]:
        heapq.heapreplace(best, entry)


def _collect_file_keyword_candidates_exact(
    file_id: int,
    file_name: str,
    content: str,
    clean_keywords: list[str],
    budget: int,
    best: list[tuple[tuple[float, float], int, dict]],
    order: int,
    bm25: tuple[BM25Index, tuple[tuple[str, float], ...]] | None = None,
) -> int:
    """含 ``İ``/``Σ`` 的文件：逐窗口重新归一化的精确路径（见 _needs_exact_window_scan）。

    这里刻意复用 ``_matched_query_keywords`` / ``_keyword_score``——修复前的 ``keyword_recall``
    调用的就是这两个函数——所以本路径与旧实现逐行同构，等价性由构造保证，而不是靠推理。
    与旧实现唯一的有意差异：窗口不再一次性全部物化成列表（旧实现先切出整篇的所有窗口再逐个
    打分，大文件上正是内存膨胀的来源），改成按起点现切一个窗口，峰值只与单个窗口有关。
    """
    normalized_content = _normalize_for_match(content)
    if not any(_normalize_for_match(keyword) in normalized_content for keyword in clean_keywords):
        # 文件层闸门：与旧实现同口径（整篇折叠后判定），不命中就整个文件跳过。
        # 这一步不能省：窗口切片独立归一化时可能命中而整篇判定不命中（词尾 Σ 折成 ς），
        # 省略会让本路径比旧实现多召回。
        return order
    for raw_start in _keyword_chunk_starts(len(content)):
        chunk_content = content[raw_start : raw_start + KEYWORD_CHUNK_SIZE].strip()
        if not chunk_content:
            continue
        matched_keywords = _matched_query_keywords(chunk_content, clean_keywords)
        if not matched_keywords:
            continue
        score = _keyword_score(chunk_content, clean_keywords)
        if score <= 0:
            continue
        order += 1
        candidate = {
            "id": f"{file_id}_{raw_start}",
            "chunk_id": str(raw_start),
            "content": chunk_content,
            "file_name": file_name,
            "file_id": file_id,
            "route": "keyword",
            "keyword_score": score,
            # 命中证据随候选一起带出，供 select_final_chunks 校验（issue #56）。
            "keyword_hits": len(matched_keywords),
            "matched_keywords": matched_keywords,
        }
        if bm25 is None:
            rank_key = (score, 0.0)
        else:
            window_norm = unicodedata.normalize(
                "NFKC", _normalize_for_match(chunk_content)
            )
            rank_key = (_bm25_cheap_upper(bm25[1], window_norm), score)
        _keep_best_candidate(best, budget, order, rank_key, candidate)
    return order


def rrf_fuse(route_results: list[tuple[str, list[dict]]], k: int = 60) -> list[dict]:
    fused: dict[str, dict] = {}
    # 两套分块方案给出同一段文本时（短文档整篇就是 offset=0 的那个关键字窗口）也要并成
    # 一条：否则同一段文字会以两条候选的身份走到重排，白占一个重排名额与上下文配额，
    # 「两条路由都召回了它」这个 RRF 信号也随之丢掉。
    content_owner: dict[str, tuple[str, str]] = {}
    for route_entry in route_results:
        if not isinstance(route_entry, (list, tuple)) or len(route_entry) != 2:
            continue
        route, chunks = route_entry
        if not isinstance(chunks, list):
            continue
        for rank, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            key = chunk_key(chunk)
            scheme = chunk_scheme(chunk)
            content = chunk_content_key(chunk)
            if content:
                owner = content_owner.get(content)
                if owner is not None and owner[0] != scheme:
                    key = owner[1]
            entry = fused.setdefault(key, {**chunk, "routes": [], "rrf_score": 0.0})
            if content:
                content_owner.setdefault(content, (scheme, key))
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["routes"].append({"route": route, "rank": rank})
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def _candidate_file_id(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _expand_parent_chunks_sync(
    db: Session | None, chunks: list[dict]
) -> tuple[list[dict], dict]:
    """按 ``(file_id, chunk_id)`` 读元数据，把 child 命中合并成 parent（issue #289）。

    关键字窗口来自原文偏移，不是入库 child 的 ``chunk_id``；两套编号都可能从 0 开始，
    按 id 查会把关键字窗口错配到别的 child，所以关键字候选一律按 child 内容降级。
    没有元数据行（旧文件）或缺少 ``parent_id``/``parent_text`` 时同样原样返回。
    """
    session = db
    owns_session = session is None
    if owns_session:
        session = SessionLocal()
    try:
        candidates: list[tuple[int, str]] = []
        keyword_unmapped = 0
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            if chunk_scheme(chunk) == KEYWORD_CHUNK_NAMESPACE:
                keyword_unmapped += 1
                continue
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            if chunk_id:
                candidates.append((_candidate_file_id(chunk.get("file_id")), chunk_id))
            else:
                keyword_unmapped += 1

        stats = {
            "children": len(chunks),
            "candidates": len(candidates),
            "mapped": 0,
            "expanded": 0,
            "merged": 0,
            "fallback": 0,
            "keyword_unmapped": keyword_unmapped,
        }
        if not candidates:
            return chunks, stats

        file_ids = {file_id for file_id, _ in candidates}
        chunk_ids = {chunk_id for _, chunk_id in candidates}
        child_rows = (
            session.query(KnowledgeChunkMetadata)
            .filter(
                KnowledgeChunkMetadata.file_id.in_(file_ids),
                KnowledgeChunkMetadata.chunk_id.in_(chunk_ids),
            )
            .all()
        )
        by_key = {(int(row.file_id), str(row.chunk_id)): row for row in child_rows}

        # parent 正文只存一份在 ``p:`` 行上；这里按 child 的 parent_id 回查，
        # 避免每个 child 行都复制一遍父级全文。
        parent_ids = {str(row.parent_id) for row in child_rows if row.parent_id}
        parent_by_key: dict[tuple[int, str], KnowledgeChunkMetadata] = {}
        if parent_ids:
            parent_rows = (
                session.query(KnowledgeChunkMetadata)
                .filter(
                    KnowledgeChunkMetadata.file_id.in_(file_ids),
                    KnowledgeChunkMetadata.chunk_id.in_(parent_ids),
                )
                .all()
            )
            parent_by_key = {
                (int(row.file_id), str(row.chunk_id)): row for row in parent_rows
            }

        expanded: list[dict] = []
        seen_parents: set[tuple[int, str]] = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                expanded.append(chunk)
                continue
            if chunk_scheme(chunk) == KEYWORD_CHUNK_NAMESPACE:
                expanded.append(chunk)
                continue
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
            file_id = _candidate_file_id(chunk.get("file_id"))
            child_row = by_key.get((file_id, chunk_id))
            if child_row is None:
                # 旧文件没有元数据行，或关键字窗口的编号不属于入库 child。
                stats["fallback"] += 1
                expanded.append(chunk)
                continue
            stats["mapped"] += 1
            parent_row = (
                parent_by_key.get((file_id, str(child_row.parent_id)))
                if child_row.parent_id
                else None
            )
            if parent_row is None or not parent_row.parent_text:
                # #287 的行没有父子关系，或 parent 行缺失/为空：按单层切片降级。
                stats["fallback"] += 1
                expanded.append(chunk)
                continue
            parent_key = (file_id, str(parent_row.chunk_id))
            if parent_key in seen_parents:
                # 同一 parent 只占一个最终上下文槽位；先出现的 child 决定顺序与分值。
                stats["merged"] += 1
                continue
            seen_parents.add(parent_key)
            stats["expanded"] += 1
            expanded.append(
                {
                    **chunk,
                    "content": parent_row.parent_text,
                    "parent_id": str(parent_row.chunk_id),
                    "parent_text": parent_row.parent_text,
                    "parent_section_path": parent_row.section_path or child_row.section_path or "",
                    "parent_article_no": parent_row.article_no or child_row.article_no or "",
                    "parent_expanded": True,
                }
            )
        return expanded, stats
    finally:
        if owns_session:
            session.close()


def _apply_parent_context_budget(
    chunks: list[dict], max_tokens: int
) -> tuple[list[dict], int, int]:
    """按 parent 边界把聚合后的上下文压到绝对 token 上限以内。

    返回 ``(保留的切片, 保留后的 token 数, 丢弃的切片数)``。只在超限时从尾部整块丢弃：
    把 parent 从中间截断会丢掉但书/例外，正是父子聚合要保留的内容。至少保留第一个
    槽位——单个 parent 本身超限时没有更细的边界可退，保留它并在 trace 里标记 truncated。
    """
    tokens = [count_chunk_tokens(str(chunk.get("content") or "")) for chunk in chunks]
    total = sum(tokens)
    if total <= max_tokens:
        return chunks, total, 0
    kept = list(chunks)
    kept_tokens = total
    dropped = 0
    while len(kept) > 1 and kept_tokens > max_tokens:
        kept_tokens -= tokens[len(kept) - 1]
        kept.pop()
        dropped += 1
    return kept, kept_tokens, dropped


async def expand_parent_chunks(
    chunks: list[dict], db: Session | None = None
) -> tuple[list[dict], dict]:
    """检索侧父子聚合入口：任何失败都退回 child 内容并写 trace，不抛给聊天链路。"""
    if not PARENT_CHILD_ENABLED:
        return chunks, {
            "enabled": False,
            "status": "disabled",
            "children": len(chunks),
            "expanded_tokens": 0,
            "truncated": False,
        }
    if not chunks:
        return chunks, {
            "enabled": True,
            "status": "empty",
            "children": 0,
            "expanded_tokens": 0,
            "truncated": False,
        }
    try:
        expanded, stats = await asyncio.to_thread(_expand_parent_chunks_sync, db, chunks)
    except Exception as exc:
        # 异常原文只进日志；trace 只留固定原因码（沿用 #263 的护栏）。
        logger.warning(
            "Parent chunk expansion failed, falling back to child content: %s",
            exc,
            exc_info=True,
        )
        return chunks, {
            "enabled": True,
            "status": "fallback",
            "reason": "metadata_unavailable",
            "children": len(chunks),
            "expanded_tokens": 0,
            "truncated": False,
        }

    if stats["expanded"]:
        status = "done" if stats["fallback"] == 0 and stats["keyword_unmapped"] == 0 else "partial"
    else:
        status = "fallback"

    if stats["expanded"]:
        # 只有真的聚合出 parent 时才施加父级上下文上限；纯 child 回退必须保持 #287
        # 的单层行为，不能因为打开开关而悄悄丢掉候选。
        expanded, expanded_tokens, dropped = _apply_parent_context_budget(
            expanded, PARENT_CONTEXT_MAX_TOKENS
        )
        stats["truncated"] = dropped > 0
        stats["dropped"] = dropped
    else:
        expanded_tokens = 0
        stats["truncated"] = False
    stats["expanded_tokens"] = expanded_tokens

    trace = {"enabled": True, "status": status, **stats}
    if status == "fallback":
        if not stats["candidates"]:
            trace["reason"] = "no_child_candidates"
        elif stats["mapped"] == 0:
            trace["reason"] = "metadata_missing"
        else:
            trace["reason"] = "no_parent_metadata"
    return expanded, trace


def _normalize_decision(data: dict) -> dict:
    if not isinstance(data, dict) or "need_rag" not in data:
        return _fallback_decision("路由模型未返回有效 JSON，保守进入 RAG。")
    need_rag = _to_bool(data.get("need_rag"))
    confidence = _to_confidence(data.get("confidence"))
    reason = str(data.get("reason") or "").strip() or "路由模型已完成判断。"
    if need_rag is None:
        return _fallback_decision("路由模型缺少 need_rag 布尔值，保守进入 RAG。")
    if confidence < ROUTE_CONFIDENCE_THRESHOLD:
        return _fallback_decision(f"路由模型置信度过低（{confidence:.2f}），保守进入 RAG。")
    return {
        "need_rag": need_rag,
        "route": "rag" if need_rag else "direct",
        "confidence": confidence,
        "reason": reason,
        "source": "router_model",
    }


def _fallback_decision(reason: str) -> dict:
    return {
        "need_rag": True,
        "route": "rag",
        "confidence": 0.0,
        "reason": reason,
        "source": "fallback",
    }


def _to_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "rag"}:
            return True
        if normalized in {"false", "no", "0", "direct"}:
            return False
    return None


def _to_confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _clip(value: str, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...(truncated)"


def _text_value(value: Any) -> str:
    return "" if value is None else str(value)


def _clean_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _fallback_keywords(question: str) -> list[str]:
    text = _text_value(question)
    keywords: list[str] = []
    numeric_phrases = re.findall(r"\d+\s*(?:分钟|元|次|天|小时)(?:以内|以上|以下|内|外)?", text)
    keywords.extend(numeric_phrases)
    policy_terms = [
        "考勤",
        "迟到",
        "早退",
        "旷工",
        "处罚",
        "罚款",
        "员工",
        "分钟",
        "以内",
        "以上",
        "制度",
        "规定",
        "流程",
        "报销",
        "请假",
        "加班",
    ]
    keywords.extend(term for term in policy_terms if term in text)
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", text):
        keywords.append(token)
        if re.search(r"[\u4e00-\u9fff]", token) and not re.search(r"\d", token) and len(token) <= 8:
            keywords.extend(token[index : index + 2] for index in range(0, max(len(token) - 1, 0)))
    return _dedupe_keywords(keywords)[:24]


def _merge_keywords(keywords: list[str], question: str) -> list[str]:
    return _dedupe_keywords([*(keywords or []), *_fallback_keywords(question)])


def _dedupe_keywords(keywords: list[str]) -> list[str]:
    result = []
    seen = set()
    for keyword in keywords:
        value = _text_value(keyword).strip()
        normalized = _normalize_for_match(value)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _expand_keywords(keywords: list[str]) -> list[str]:
    expanded = []
    for keyword in keywords:
        value = _text_value(keyword).strip()
        if not value:
            continue
        expanded.append(value)
        for phrase in re.findall(r"\d+\s*(?:分钟|元|次|天|小时)(?:以内|以上|以下|内|外)?", value):
            expanded.append(phrase)
        if re.search(r"[\u4e00-\u9fff]", value) and not re.search(r"\d", value) and 2 < len(value) <= 8:
            expanded.extend(value[index : index + 2] for index in range(0, len(value) - 1))
    return _dedupe_keywords(expanded)


def _keyword_score(content: str, keywords: list[str]) -> float:
    """一段文本的关键词分值（字符串入口：自行归一化入参）。

    keyword_recall 的常规路径走窗口游标版本（``_score_from_hits``），但含 İ/Σ 的正文走
    ``_collect_file_keyword_candidates_exact``，那里与旧实现一样按窗口切片调用这个函数；
    等价性基线用例（tests/test_keyword_recall_memory_59.py）也走它。
    """
    normalized_content = _normalize_for_match(content)
    hits = _keyword_hits(normalized_content, _prepare_keywords(keywords))
    return _score_from_hits(hits)[0]


def _matched_query_keywords(content: str, keywords: list[str]) -> list[str]:
    """返回内容里真正命中的查询关键词（保持传入顺序、按归一化形式去重）。

    issue #56：窗口是否与本次提问相关，只由这个命中集合决定；内容自身的场景
    特征词（考勤、迟到、罚款…）不构成相关性证据。

    与 ``_keyword_score`` 一样，含 İ/Σ 的正文在回退路径上按窗口切片调用本函数。
    """
    normalized_content = _normalize_for_match(content)
    hits = _keyword_hits(normalized_content, _prepare_keywords(keywords))
    matched: list[str] = []
    seen: set[str] = set()
    for keyword, normalized, _, count, _ in hits:
        if count <= 0 or normalized in seen:
            continue
        seen.add(normalized)
        matched.append(keyword)
    return matched


def _keyword_hits(
    normalized_content: str, prepared_keywords: list[tuple[str, str, float]]
) -> list[tuple[str, str, float, int, int]]:
    """在已归一化文本上算命中：(原始关键词, 归一化关键词, 权重, 非重叠命中数, 首次出现位置)。"""
    hits = []
    for keyword, normalized, weight in prepared_keywords:
        count = normalized_content.count(normalized)
        if count <= 0:
            continue
        hits.append((keyword, normalized, weight, count, normalized_content.find(normalized)))
    return hits


def _score_from_hits(hits: list[tuple[str, str, float, int, int]]) -> tuple[float, list[str]]:
    """关键词打分的唯一实现，返回 (分值, 命中的查询关键词)。

    hits 里的首次出现位置在「同一坐标系」内比较（字符串入口是整段文本，窗口路径是窗口
    内偏移），因为相邻命中判定用的是相对距离。
    """
    score = 0.0
    matched_positions = []
    matched: list[str] = []
    seen: set[str] = set()
    for keyword, normalized, weight, count, first_position in hits:
        if count <= 0:
            continue
        score += count * weight
        if first_position >= 0:
            matched_positions.append(first_position)
        if normalized not in seen:
            seen.add(normalized)
            matched.append(keyword)
    # 这里不再按内容自身的特征短语（考勤/迟到+早退/30分钟以内/罚款50元…）无条件加分：
    # 那与「本次查询问的是什么」无关，会让整段跑题内容拿到高分并被插到上下文首位。
    # 相关性只由上面的「命中本次查询关键词」决定，零命中内容得 0 分。
    unique_hits = len({pos for pos in matched_positions if pos >= 0})
    score += unique_hits * 1.5
    if _has_close_positions(matched_positions, KEYWORD_CLOSE_WINDOW):
        score += 8.0
    return score, matched


def _has_close_positions(positions: list[int], window: int = KEYWORD_CLOSE_WINDOW) -> bool:
    if len(positions) < 3:
        return False
    positions.sort()
    return any(positions[index + 2] - positions[index] <= window for index in range(len(positions) - 2))


class _NormalizedOffsetMap:
    """原始文本偏移 → 归一化文本偏移（去掉空白后的下标），查询偏移必须单调不减。

    窗口边界在原始文本坐标上（``chunk_id`` 就是原始偏移），而关键词命中位置在归一化文本
    坐标上，两者必须换算。每次只统计「距上次查询新增的那一段」里的非空白字符数，整篇恰好
    被扫一遍，不做逐字符或逐空白游程的 Python 级循环（空白多的文档上差一个数量级）。

    归一化长度与原文长度相同时，文本既没有空白、大小写折叠也没有改变长度，映射就是恒等
    映射，连这一次扫描都省掉——中文语料大多走这条快路径。

    前提：正文不含 ``İ``/``Σ``（``_needs_exact_window_scan`` 会把这类文件交给回退路径）。
    İ 是唯一一个 ``lower()`` 会变长的码位，含它就等于「折叠改变了长度」，上面的恒等映射判据
    与按字符计数的偏移换算都不再成立；Σ 不改长度，映射本身没错，但它让「归一化后的这一段」
    不等于「对这一段再做归一化」（词尾折 ς 取决于窗口外的邻居），区间等价同样失效。
    """

    __slots__ = ("_content", "_identity", "_base", "_count")

    def __init__(self, content: str, normalized_length: int):
        self._content = content
        self._identity = normalized_length == len(content)
        self._base = 0  # 已经统计到的原始偏移
        self._count = 0  # content[: self._base] 里的非空白字符数

    def rank(self, offset: int) -> int:
        if self._identity:
            return offset
        if offset > self._base:
            self._count += len(_WHITESPACE_RE.sub("", self._content[self._base : offset]))
            self._base = offset
        return self._count


def _normalize_for_match(text: str, case_fold: bool = True) -> str:
    """去空白 + 转小写，顺序与旧实现一致：先 ``lower()`` 再去空白。

    顺序不能对调：Σ 的折写取决于它后面是不是空白——``'ΑΣ ΟΔΟΣ'`` 先 lower 再去空白得
    ``'αςοδος'``，先去空白再 lower 得 ``'ασοδος'``，同一段文本两个结果。两种顺序的瞬时峰值
    实测相当（200 万字符中文语料：26.7MB vs 29.5MB，视空白密度各有胜负），没有拿顺序换内存
    的理由。

    ``case_fold=False`` 时省掉整篇 str.lower()（见 _needs_case_fold）。
    """
    value = "" if text is None else str(text)
    lowered = value.lower() if case_fold else value
    return _WHITESPACE_RE.sub("", lowered)


def _split_keyword_chunks(
    content: str, chunk_size: int = KEYWORD_CHUNK_SIZE, chunk_overlap: int = KEYWORD_CHUNK_OVERLAP
) -> list[dict]:
    """整篇物化关键字窗口：召回路径不再走它（issue #59 的内存修复），但它是**在用的契约面**。

    ``tests/test_chunk_key_namespace.py``（issue #55）直接以它为准钉住「关键字窗口按字符偏移
    从 0 编号、与入库切片的顺序序号撞键」这个性质；``rerank.chunk_key`` 的命名空间隔离也按同
    一契约描述。删掉它会让那条守卫失去被测面（CI 在 merge ref 上红过一次，见 #82）。

    起点与切片规则都走生产路径同一套 ``_keyword_chunk_starts`` + ``KEYWORD_CHUNK_SIZE``，所以
    它不会与召回路径漂移；调用方只该把它当契约/排查入口，不要拿它做整篇召回。
    """
    chunks = []
    for start in _keyword_chunk_starts(len(content), chunk_size, chunk_overlap):
        text = content[start : start + chunk_size].strip()
        if text:
            chunks.append({"chunk_id": str(start), "content": text})
    return chunks


def _keyword_chunk_starts(
    content_length: int, chunk_size: int = KEYWORD_CHUNK_SIZE, chunk_overlap: int = KEYWORD_CHUNK_OVERLAP
) -> range:
    """窗口起点（原始偏移），步长与旧的切块实现一致。"""
    return range(0, content_length, max(chunk_size - chunk_overlap, 1))


async def _trace_add(trace_recorder: Any, *args, **kwargs) -> None:
    """记录一条检索侧事件；失败静默，检索结果不受轨迹影响。

    `TraceRecorder.add` 是协程（写库交给工作线程，issue #201）；`retrieve_knowledge` 是
    事件循环线程上的协程，两个调用点都必须 await，否则写回又回到循环线程上。
    """
    if not trace_recorder:
        return
    try:
        await trace_recorder.add(*args, **kwargs)
    except Exception:
        pass
