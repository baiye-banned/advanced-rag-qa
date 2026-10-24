from __future__ import annotations

import json
import logging
import math
from typing import Any

import httpx

from config import (
    RERANK_API_KEY,
    RERANK_BASE_URL,
    RERANK_LLM_FALLBACK_ENABLED,
    RERANK_MODEL,
    RERANK_PROVIDER,
    RERANK_TIMEOUT_SECONDS,
    RETRIEVAL_RERANK_TOP_N,
)
from rag.llm import call_chat_json

logger = logging.getLogger(__name__)

# rerank trace 会随消息落库，并由消息接口反复回吐给客户端；上游 URL、异常类型、
# key、配置路径与原始错误串一律只进服务端日志，trace 里只写这些固定文案（issue #263）。
RERANK_FAILED_MESSAGE = "重排服务暂时不可用"
RERANK_FALLBACK_MESSAGE = "主重排服务不可用，已切换备用重排"
RERANK_ALL_FAILED_MESSAGE = "主重排与备用重排均不可用"
RERANK_FALLBACK_FAILED_MESSAGE = "备用重排服务暂时不可用"


async def rerank_chunks(question: str, chunks: list[dict]) -> tuple[list[dict], dict]:
    if not chunks:
        return [], {"status": "skipped", "items": []}
    try:
        return await _rerank_chunks_with_qwen3(question, chunks)
    except Exception as exc:
        if not RERANK_LLM_FALLBACK_ENABLED:
            logger.warning("rerank failed [scope=rerank_primary]: %s", exc, exc_info=True)
            return [], {
                "status": "failed",
                "provider": RERANK_PROVIDER,
                "model": RERANK_MODEL,
                "error": RERANK_FAILED_MESSAGE,
                "items": [],
            }
        reranked, trace = await _rerank_chunks_with_llm(question, chunks)
        if trace.get("status") == "done":
            logger.warning(
                "rerank fallback succeeded [scope=rerank_fallback_success]: %s", exc, exc_info=True
            )
            trace["provider"] = "deepseek_fallback"
            trace["fallback_from"] = RERANK_MODEL
            trace["fallback_reason"] = RERANK_FALLBACK_MESSAGE
        else:
            logger.warning(
                "rerank fallback failed [scope=rerank_fallback_failed]: %s", exc, exc_info=True
            )
            trace["fallback_from"] = RERANK_MODEL
            trace["fallback_reason"] = RERANK_ALL_FAILED_MESSAGE
        return reranked, trace


async def _rerank_chunks_with_qwen3(question: str, chunks: list[dict]) -> tuple[list[dict], dict]:
    if not RERANK_API_KEY:
        raise RuntimeError("RERANK_API_KEY or DASHSCOPE_API_KEY is not configured")

    documents = [_text_value(chunk.get("content"))[:4000] for chunk in chunks]
    payload = {
        "model": RERANK_MODEL,
        "query": question,
        "documents": documents,
        "top_n": min(RETRIEVAL_RERANK_TOP_N, len(chunks)),
        "instruct": "根据用户问题判断企业知识库片段的相关性，优先选择能够直接回答问题的片段。",
    }
    headers = {
        "Authorization": f"Bearer {RERANK_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=RERANK_TIMEOUT_SECONDS) as client:
        response = await client.post(RERANK_BASE_URL, json=payload, headers=headers)
    response.raise_for_status()

    rows = _extract_qwen3_rerank_rows(response.json())
    if not rows:
        raise ValueError("qwen3-rerank response has no results")

    reranked: list[dict] = []
    trace_items: list[dict] = []
    for row in rows:
        index = _to_int(row.get("index"))
        if index is None or index < 0 or index >= len(chunks):
            continue
        score = _to_float(row.get("relevance_score", row.get("score")))
        chunk = chunks[index]
        next_chunk = {
            **chunk,
            "rerank_score": score,
            "rerank_reason": "qwen3-rerank relevance score",
        }
        reranked.append(next_chunk)
        trace_items.append(trace_chunk(next_chunk))

    if not reranked:
        raise ValueError("qwen3-rerank response indexes did not match candidates")

    reranked.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
    return reranked, {
        "status": "done",
        "provider": RERANK_PROVIDER,
        "model": RERANK_MODEL,
        "items": trace_items,
    }


def _extract_qwen3_rerank_rows(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        return []
    candidates = [
        data.get("results"),
        data.get("data"),
        (data.get("output") or {}).get("results") if isinstance(data.get("output"), dict) else None,
    ]
    for rows in candidates:
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float:
    number = _to_number(value)
    return max(0.0, min(1.0, number))


def _to_number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


async def _rerank_chunks_with_llm(question: str, chunks: list[dict]) -> tuple[list[dict], dict]:
    compact_candidates = [
        {
            "id": index,
            "file_name": chunk.get("file_name", ""),
            "content": _text_value(chunk.get("content"))[:700],
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    system_prompt = (
        "你是企业知识库 RAG 重排器。只输出 JSON，不要输出 Markdown。"
        "根据用户问题评估候选片段的相关性，返回字段 results，数组元素包含 id、score、reason。"
        "score 范围 0 到 1。"
    )
    user_prompt = json.dumps(
        {
            "question": question,
            "candidates": compact_candidates,
            "top_n": RETRIEVAL_RERANK_TOP_N,
        },
        ensure_ascii=False,
    )
    try:
        data = await call_chat_json(system_prompt, user_prompt, max_tokens=1200)
        rows = data.get("results") or []
        by_id = {index: chunk for index, chunk in enumerate(chunks, start=1)}
        reranked = []
        trace_items = []
        for row in rows:
            candidate_id = _to_int(row.get("id"))
            if candidate_id is None:
                continue
            chunk = by_id.get(candidate_id)
            if not chunk:
                continue
            score = _to_float(row.get("score"))
            next_chunk = {**chunk, "rerank_score": score, "rerank_reason": str(row.get("reason", ""))}
            reranked.append(next_chunk)
            trace_items.append(trace_chunk(next_chunk))
        reranked.sort(key=lambda item: _to_float(item.get("rerank_score")), reverse=True)
        return reranked, {"status": "done", "provider": "deepseek", "items": trace_items}
    except Exception as exc:
        logger.warning(
            "llm rerank failed [scope=rerank_llm_fallback]: %s", exc, exc_info=True
        )
        return [], {
            "status": "failed",
            "provider": "deepseek",
            "error": RERANK_FALLBACK_FAILED_MESSAGE,
            "items": [],
        }


def select_final_chunks(ranked_chunks: list[dict], keyword_chunks: list[dict]) -> list[dict]:
    # 先判重再截断：同文候选（不同分块方案产出同一段文本）若占着一个配额再被去掉，
    # 会让排在 TOP_N 之后的那条真实候选白白落选。
    selected = _dedupe_chunks(ranked_chunks)[:RETRIEVAL_RERANK_TOP_N]
    clean_keyword_chunks = [chunk for chunk in keyword_chunks if isinstance(chunk, dict)]
    if clean_keyword_chunks:
        best_keyword = clean_keyword_chunks[0]
        best_score = _to_number(best_keyword.get("keyword_score"))
        already_selected = any(_same_context(chunk, best_keyword) for chunk in selected)
        # 绝对分值只说明「它像自己文档里的关键字内容」，不说明「它跟本次提问有关」：
        # 插前还必须确认候选命中了查询关键词（issue #56），配额不被跑题候选挤占。
        if best_score >= 10 and _keyword_chunk_hits_query(best_keyword) and not already_selected:
            selected = [best_keyword, *selected]
    return _dedupe_chunks(selected)[:RETRIEVAL_RERANK_TOP_N]


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_keys = set()
    seen_contents: dict[str, str] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        key = chunk_key(chunk)
        if key in seen_keys:
            continue
        scheme = chunk_scheme(chunk)
        content = chunk_content_key(chunk)
        if content:
            # 只并「两套分块方案给出同一段文本」这一种情形；同一套方案里的重复文本
            # （页眉页脚、表格表头这类）维持原有「按 id 各算一条」的行为。
            owner = seen_contents.get(content)
            if owner is not None and owner != scheme:
                continue
            seen_contents.setdefault(content, scheme)
        seen_keys.add(key)
        deduped.append(chunk)
    return deduped


def _same_context(left: dict, right: dict) -> bool:
    """两条候选是否是「同一段上下文」：同一条切片，或两套分块方案给出的同一段文本。

    前者靠融合键判断（同一条切片被多路召回命中）；后者兜住分块方案不同、文本却完全
    一样的情形（短文档整篇就是 offset=0 的那个窗口），那样两条候选进上下文只会重复
    占用配额。
    """
    if chunk_key(left) == chunk_key(right):
        return True
    if chunk_scheme(left) == chunk_scheme(right):
        return False
    left_content = chunk_content_key(left)
    return bool(left_content) and left_content == chunk_content_key(right)


def chunk_scheme(chunk: dict) -> str:
    """候选来自哪一套分块方案。

    关键字窗口与入库切片各自从 0 编号，是两套方案；多路向量召回命中的都是同一条
    入库切片，算同一套。
    """
    return KEYWORD_CHUNK_NAMESPACE if chunk.get("route") == "keyword" else INGEST_CHUNK_NAMESPACE


def chunk_content_key(chunk: dict) -> str:
    """「同一份文件里的同一段文本」这个身份；没有正文时返回空串，不参与判重。

    空正文若也参与判重，同一份文件里两条没有正文的候选会被误判成同一条。
    """
    content = "".join(str(chunk.get("content") or "").split())
    return f"{chunk.get('file_id', 0)}:{content}" if content else ""


def _keyword_chunk_hits_query(chunk: dict) -> bool:
    """关键字候选是否命中了本次查询关键词。

    keyword_recall 会把命中数写进 ``keyword_hits``；没有该字段的候选来自不经过
    关键字召回的调用方（历史行为），保持原有的「只看绝对分值」语义。
    """
    if "keyword_hits" not in chunk:
        return True
    return _to_number(chunk.get("keyword_hits")) > 0


KEYWORD_CHUNK_NAMESPACE = "kw"
INGEST_CHUNK_NAMESPACE = "ingest"


def chunk_key(chunk: dict) -> str:
    """候选在融合与去重时的身份键：``file_id`` + 分块方案 + 块位置。

    关键字窗口（``_split_keyword_chunks``，id 是字符偏移）与入库切片（``chunk_text``，
    id 是顺序序号）是两套互不相干的分块方案，两边的编号都从 0 开始，只看
    ``file_id + chunk_id`` 会把 offset=0 的关键字窗口误判成「同一条入库切片」，
    在 RRF 融合里被静默去重掉、连内容与 keyword_score 一并丢失（issue #55）。

    这里区分的是**分块方案**，不是召回路由：多路向量召回（planned/simplified/
    sub_question_*/hyde/rewrite_*）命中的是同一条入库切片，必须继续按
    ``file_id + chunk_id`` 合并，RRF 排序才有意义，所以只有关键字窗口另起命名空间。
    """
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    if chunk_scheme(chunk) == KEYWORD_CHUNK_NAMESPACE:
        return f"{chunk.get('file_id', 0)}:{KEYWORD_CHUNK_NAMESPACE}:{chunk_id}"
    return f"{chunk.get('file_id', 0)}:{chunk_id}"


def trace_chunk(chunk: dict) -> dict:
    return {
        "file_id": chunk.get("file_id", 0),
        "file_name": chunk.get("file_name", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "route": chunk.get("route", ""),
        "routes": chunk.get("routes", []),
        "rrf_score": chunk.get("rrf_score"),
        "rerank_score": chunk.get("rerank_score"),
        "rerank_reason": chunk.get("rerank_reason", ""),
        "keyword_score": chunk.get("keyword_score"),
        "excerpt": _text_value(chunk.get("content"))[:120],
    }


def _text_value(value: Any) -> str:
    return "" if value is None else str(value)
