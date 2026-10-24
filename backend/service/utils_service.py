
import logging
import mimetypes
import re
from pathlib import Path
from uuid import uuid4

from config import KNOWLEDGE_UPLOAD_MAX_MB


logger = logging.getLogger(__name__)

IMAGE_UPLOAD_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
CHAT_ATTACHMENT_MAX_BYTES = 5 * 1024 * 1024
AVATAR_MAX_BYTES = 2 * 1024 * 1024

# 知识库上传白名单：扩展名必须与 crud.knowledge_file.extract_file_text 的抽取链一致。
KNOWLEDGE_UPLOAD_TYPES = {
    ".txt": frozenset({"text/plain"}),
    ".md": frozenset({"text/markdown", "text/plain"}),
    ".docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    ".pdf": frozenset({"application/pdf"}),
}
# 浏览器和命令行工具常把任意文件标成这些通用类型，无法据此判断真实格式，按扩展名放行。
GENERIC_UPLOAD_CONTENT_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})
KNOWLEDGE_UPLOAD_MAX_BYTES = KNOWLEDGE_UPLOAD_MAX_MB * 1024 * 1024
KNOWLEDGE_UPLOAD_TYPE_ERROR_MESSAGE = (
    "仅支持 " + "、".join(ext.lstrip(".") for ext in KNOWLEDGE_UPLOAD_TYPES) + " 格式的文件"
)


def _build_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    for index, chunk in enumerate(chunks, start=1):
        content = _text_value(chunk.get("content")).strip()
        if not content:
            continue
        source = {
            "index": index,
            "file_id": chunk.get("file_id", 0),
            "file_name": chunk.get("file_name", "Untitled"),
            "chunk_id": chunk.get("chunk_id", ""),
            "route": chunk.get("route", ""),
            "routes": chunk.get("routes", []),
            "rrf_score": chunk.get("rrf_score"),
            "rerank_score": chunk.get("rerank_score"),
            "rerank_reason": chunk.get("rerank_reason", ""),
            "content": content,
            "excerpt": content[:180] + ("..." if len(content) > 180 else ""),
        }
        # 父子切片（issue #289）：命中 child、返回 parent 时附带可选元数据。
        # 旧客户端忽略这些新增键；没有 parent 的候选保持与原响应完全一致。
        if chunk.get("parent_id"):
            source["parent_id"] = chunk.get("parent_id")
            source["parent_section_path"] = chunk.get("parent_section_path", "")
            source["parent_article_no"] = chunk.get("parent_article_no", "")
            source["parent_expanded"] = True
        sources.append(source)
    return sources


def resolve_image_upload_type(content_type: str | None, filename: str | None = None, *, allow_filename_fallback: bool = False) -> tuple[str, str] | None:
    resolved_type = _normalize_content_type(content_type)
    # 回退触发条件是「客户端给的类型不携带格式信息」，不是「客户端没给类型」：通用类型
    # （空、两种 octet-stream，见 :29）无法据此判断真实格式，按扩展名回退；**明确**的类型
    # （如 text/plain）不回退——它与「未知」不同类，放行它等于让判据退化成「只看扩展名」。
    # 与 resolve_knowledge_upload_type 共用同一张表（issue #237）。
    if resolved_type in GENERIC_UPLOAD_CONTENT_TYPES and allow_filename_fallback:
        resolved_type = _normalize_content_type(mimetypes.guess_type(filename or "")[0])
    ext = IMAGE_UPLOAD_TYPES.get(resolved_type)
    if not ext:
        return None
    return resolved_type, ext


def resolve_knowledge_upload_type(content_type: str | None, filename: str | None) -> str | None:
    """返回白名单内的归一化扩展名；扩展名或声明的 MIME 不符时返回 None。"""
    ext = Path(filename or "").suffix.lower()
    allowed_types = KNOWLEDGE_UPLOAD_TYPES.get(ext)
    if not allowed_types:
        return None
    normalized_type = _normalize_content_type(content_type)
    if normalized_type in GENERIC_UPLOAD_CONTENT_TYPES or normalized_type in allowed_types:
        return ext
    return None


def knowledge_upload_too_large_message() -> str:
    return f"文件不能超过 {KNOWLEDGE_UPLOAD_MAX_MB}MB"


def _internal_error_detail(user_message: str, scope: str, exc: Exception) -> str:
    """内部异常只落服务端日志，回给用户的是固定文案 + 可与日志对照的编号。

    异常原文（数据库/驱动报错、文件路径、上游服务地址）对用户没有价值，却能给后续
    针对性攻击提供信息，因此一律不回显；排障靠这条 warning 的 exc_info 与用户报出的
    error_id 对照。与 main.py 兜底 IntegrityError 的判据一致。

    exc_info 传异常对象而不是 True：调用点不一定在 except 体内（如
    `retrieval.py` 从 `asyncio.gather(return_exceptions=True)` 的返回值里取异常），
    那种位置 `sys.exc_info()` 是空的，traceback 会打成无意义的 "NoneType: None"。
    异常对象自带 `__traceback__`，两种情况都能落出与消息里同一个异常的真实栈。
    """
    error_id = uuid4().hex[:8]
    logger.warning("%s failed [error_id=%s]: %s", scope, error_id, exc, exc_info=exc)
    return f"{user_message}（错误编号：{error_id}）"


def _normalize_content_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _clip_text(text: str, max_chars: int) -> str:
    value = "" if text is None else str(text).strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _check_answer_grounding(answer: str, contexts: list[str]) -> dict:
    clean_context = _normalize_grounding_text("\n".join(contexts or []))
    if not answer or not clean_context:
        return {
            "status": "skipped",
            "reason": "missing answer or retrieved contexts",
            "checked_sentences": 0,
            "unsupported_claims": [],
        }

    unsupported = []
    checked = 0
    for sentence in _split_answer_sentences(answer):
        if len(_normalize_grounding_text(sentence)) < 8:
            continue
        checked += 1
        if not _sentence_supported(sentence, clean_context):
            unsupported.append(sentence[:160])
        if len(unsupported) >= 5:
            break

    if checked == 0:
        status = "skipped"
        reason = "no checkable answer sentences"
    elif unsupported:
        status = "needs_review"
        reason = "some answer sentences have weak lexical support in retrieved contexts"
    else:
        status = "passed"
        reason = "answer sentences are lexically supported by retrieved contexts"

    return {
        "status": status,
        "reason": reason,
        "checked_sentences": checked,
        "unsupported_count": len(unsupported),
        "unsupported_claims": unsupported,
    }


def _split_answer_sentences(answer: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[。！？!?；;])\s*|\n+", str(answer or ""))
        if item.strip()
    ]


def _sentence_supported(sentence: str, context: str) -> bool:
    normalized = _normalize_grounding_text(sentence)
    if not normalized:
        return True
    numbers = re.findall(r"\d+(?:\.\d+)?", sentence)
    if numbers and not all(number in context for number in numbers):
        return False
    if normalized in context:
        return True
    grams = {normalized[index : index + 2] for index in range(max(len(normalized) - 1, 0))}
    if not grams:
        return True
    shared = sum(1 for gram in grams if gram in context)
    return shared >= 6 and shared / max(len(grams), 1) >= 0.18


def _normalize_grounding_text(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9.]+", _text_value(text).lower()))


def _text_value(value) -> str:
    return "" if value is None else str(value)
