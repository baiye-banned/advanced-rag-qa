import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from config import (
    MEMORY_API_KEY,
    MEMORY_MODEL,
    MEMORY_RECENT_MAX_CHARS,
    MEMORY_RECENT_MAX_TOKENS,
    MEMORY_SUMMARY_MAX_CHARS,
    MEMORY_SUMMARY_MAX_TOKENS,
    MEMORY_WINDOW_TURNS,
)
from database.session import SessionLocal
from model.models import Conversation, Message
from rag.learning_trace import append_trace_event
from rag.llm import call_chat_text_with_metadata
from service.utils_service import _clip_text

if TYPE_CHECKING:
    # 只在类型检查期导入：`service.chat_service` 反过来 import 本模块，运行期导入会成环。
    from service.chat_service import _ConversationState


logger = logging.getLogger(__name__)

# 轨迹事件会经 get_chat_trace/get_message_trace 回查给用户，因此只放固定文案；
# 异常原文由上面的 logger.warning(..., exc_info=True) 留在服务端。
MEMORY_SUMMARY_UPDATE_FAILED_MESSAGE = "长期记忆更新失败"


def _read_recent_messages(conversation_id: str, current_message_id: int, summary_upto: int) -> list[Message]:
    """读最近窗口的消息。

    整段同步 DB 工作（开 session → 查询 → 关 session）都发生在这个函数里，调用方一律经
    `asyncio.to_thread` 进来，于是它整段跑在同一个工作线程上：Session 既不在事件循环线程上
    开，也不跨线程传递（issue #187；`retrieval.py:163-166` 是本仓对该形态的成文规范）。

    返回的 Message 已经是 detached 只读快照：`close()` 只释放连接、不抹掉已加载的列，
    调用方只读 `id`/`role`/`content` 这些查询时就带回来的标量列，不会触发懒加载。
    """
    db = SessionLocal()
    try:
        return (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.id < current_message_id,
                Message.id > summary_upto,
            )
            .order_by(Message.id.desc())
            .limit(max(MEMORY_WINDOW_TURNS, 1) * 2)
            .all()
        )
    finally:
        db.close()


async def _build_recent_memory_text(
    conversation_id: str,
    current_message_id: int,
    summary_upto: int = 0,
    trace_id: str | None = None,
) -> str:
    """构造最近窗口文本。

    早先这个协程自己接一个请求级 `Session`，查询直接跑在事件循环线程上；现在它只收标量，
    真正的查询交给 `_read_recent_messages` 在工作线程里完成。
    """
    recent_messages = await asyncio.to_thread(
        _read_recent_messages,
        conversation_id,
        current_message_id,
        summary_upto or 0,
    )
    return await _format_recent_memory_messages(list(reversed(recent_messages)), trace_id=trace_id)


def _build_memory_context(
    conversation: "_ConversationState",
    recent_text: str | None = None,
) -> str:
    summary = (conversation.memory_summary or "").strip()
    if recent_text is None:
        recent_text = ""

    sections = []
    if summary:
        sections.append(f"长期摘要记忆：\n{_clip_text(summary, MEMORY_SUMMARY_MAX_CHARS)}")
    if recent_text:
        sections.append(f"最近对话窗口：\n{recent_text}")
    return "\n\n".join(sections).strip()


def _build_memory_aware_retrieval_question(question: str, memory_context: str) -> str:
    if not memory_context:
        return question
    compact_memory = _clip_text(memory_context, 1500)
    return (
        "以下会话记忆仅用于消解当前问题中的指代和省略，不作为事实依据。\n"
        f"{compact_memory}\n\n"
        f"当前检索问题：{question}"
    )


async def _format_recent_memory_messages(messages: list[Message], trace_id: str | None = None) -> str:
    turns = _group_message_texts_into_turns(messages)
    if not turns:
        return ""

    rendered_turns = ["\n".join(turn) for turn in turns]
    joined_text = "\n\n".join(rendered_turns)
    if len(joined_text) <= MEMORY_RECENT_MAX_CHARS:
        return joined_text

    await asyncio.to_thread(
        append_trace_event,
        trace_id,
        "recent_memory_compaction_triggered",
        "_format_recent_memory_messages",
        params={
            "recent_chars": len(joined_text),
            "recent_limit": MEMORY_RECENT_MAX_CHARS,
            "turns": len(rendered_turns),
        },
        creates={"transcript": joined_text},
        note="短期记忆超过上限，系统将最近窗口压缩为一条近期记忆。",
    )
    summary = await _summarize_recent_memory(joined_text, trace_id=trace_id)
    if summary:
        compacted = f"近期记忆压缩：\n{summary}"
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "recent_memory_compacted",
            "_summarize_recent_memory",
            creates={"recent_text": compacted},
            result={"recent_chars": len(compacted)},
            note="短期记忆已完成语义压缩，近期窗口只保留这一条压缩记忆。",
        )
        return _clip_text(compacted, MEMORY_RECENT_MAX_CHARS)

    fallback = _fallback_compact_recent_memory(rendered_turns)
    await asyncio.to_thread(
        append_trace_event,
        trace_id,
        "recent_memory_compaction_fallback",
        "_fallback_compact_recent_memory",
        creates={"recent_text": fallback},
        result={"recent_chars": len(fallback)},
        note="短期记忆语义压缩失败，系统使用裁剪文本兜底，避免主回答中断。",
    )
    return fallback


def _schedule_memory_summary_update(conversation_id: str, trace_id: str | None = None):
    try:
        asyncio.create_task(_update_memory_summary_from_sliding_window(conversation_id, trace_id))
    except RuntimeError:
        logger.warning("Unable to schedule memory summary update: conversation_id=%s", conversation_id)
        append_trace_event(
            trace_id,
            "memory_summary_update_schedule_failed",
            "_schedule_memory_summary_update",
            result={"conversation_id": conversation_id},
            note="当前事件循环不可用，滑出窗口记忆没有成功启动长期摘要更新。",
        )


def _read_conversation_window(conversation_id: str) -> tuple[str, list[Message]] | None:
    """读「会话当前摘要 + 摘要上界之后的消息」。

    整段同步 DB 工作跑在工作线程里（见 `_read_recent_messages` 的说明）。会话不存在返回
    None，让调用方区分「没有会话」与「摘要为空」。消息同样是 detached 只读快照。
    """
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            return None
        summary_upto = conversation.memory_summary_upto_message_id or 0
        messages = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.id > summary_upto,
            )
            .order_by(Message.id.asc())
            .all()
        )
        return conversation.memory_summary or "", messages
    finally:
        db.close()


def _read_memory_summary(conversation_id: str) -> str | None:
    """读会话当前的长期摘要（已 strip）。会话不存在返回 None，同样是为了区分两种「空」。"""
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            return None
        return (conversation.memory_summary or "").strip()
    finally:
        db.close()


def _write_memory_summary_text(conversation_id: str, summary: str, summary_upto_message_id: int | None = None) -> None:
    """把新的长期摘要写回会话，整段同步 DB 工作在工作线程里完成。

    `summary_upto_message_id` 为 None 表示只改摘要文本（二次压缩不动上界）。会话在读取与
    写入之间被删掉时静默跳过——这条链路本来就是尽力而为的旁路，不该让主回答失败。
    """
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter_by(id=conversation_id).first()
        if not conversation:
            return
        conversation.memory_summary = summary
        if summary_upto_message_id is not None:
            conversation.memory_summary_upto_message_id = summary_upto_message_id
        conversation.memory_updated_at = datetime.now()
        db.commit()
    finally:
        db.close()


async def _update_memory_summary_from_sliding_window(conversation_id: str, trace_id: str | None = None):
    """把滑出最近窗口的轮次并进长期摘要。

    读、写两段同步 DB 工作各自在自己的工作线程里「开 session → 用 → 关」：中间夹着一次模型
    外呼（`await _summarize_conversation_memory`），跨不过去一个 Session，所以这里按「每步用
    短生命周期 session」落地（issue #187 验收第 3 条给的两种修法之一）。早先的写法是把一个
    请求级 Session 一直握在协程里，`db.query/commit/close` 全部落在事件循环线程上。
    """
    try:
        window = await asyncio.to_thread(_read_conversation_window, conversation_id)
        if window is None:
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "memory_summary_update_skipped",
                "_update_memory_summary_from_sliding_window",
                result={"reason": "conversation_not_found"},
                note="长期记忆更新跳过：会话不存在。",
            )
            return

        current_summary, messages = window
        turns = _group_messages_into_turns(messages)
        if len(turns) <= MEMORY_WINDOW_TURNS:
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "memory_summary_update_skipped",
                "_update_memory_summary_from_sliding_window",
                result={
                    "turns_after_summary": len(turns),
                    "window_turns": MEMORY_WINDOW_TURNS,
                    "reason": "within_recent_window",
                },
                note="长期记忆更新跳过：未摘要轮次仍在最近窗口内，没有会话滑出。",
            )
            await _compact_summary_if_needed(conversation_id, trace_id)
            return

        slipped_turns = turns[: -MEMORY_WINDOW_TURNS]
        slipped_messages = [message for turn in slipped_turns for message in turn]
        transcript = _format_messages_for_summary(slipped_messages)
        if not transcript:
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "memory_summary_update_skipped",
                "_update_memory_summary_from_sliding_window",
                result={"reason": "empty_slipped_transcript"},
                note="长期记忆更新跳过：滑出窗口的消息没有可摘要文本。",
            )
            return

        previous_summary = (current_summary or "").strip()
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "memory_summary_update_triggered",
            "_update_memory_summary_from_sliding_window",
            params={
                "slipped_turns": len(slipped_turns),
                "slipped_messages": len(slipped_messages),
                "previous_summary_chars": len(previous_summary),
                "window_turns": MEMORY_WINDOW_TURNS,
            },
            creates={"transcript": transcript},
            note="最近窗口发生滑动，系统将滑出窗口的完整问答轮次合并进长期记忆。",
        )

        next_summary = await _summarize_conversation_memory(
            previous_summary, transcript, trace_id=trace_id
        )
        if not next_summary:
            next_summary = _fallback_merge_summary(previous_summary, transcript)
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "memory_summary_update_fallback",
                "_fallback_merge_summary",
                result={"summary_chars": len(next_summary)},
                note="模型摘要失败，系统使用可读文本兜底合并，避免滑出窗口记忆直接丢失。",
            )

        summary_text = next_summary.strip()
        summary_upto = max(message.id for message in slipped_messages)
        await asyncio.to_thread(_write_memory_summary_text, conversation_id, summary_text, summary_upto)
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "memory_summary_updated",
            "_summarize_conversation_memory",
            creates={
                "memory_summary": summary_text,
                "summary_upto_message_id": summary_upto,
            },
            result={
                "summary_chars": len(summary_text),
                "slipped_turns": len(slipped_turns),
            },
            note="滑出最近窗口的会话已写入长期记忆，并推进 summary_upto_message_id，避免后续重复摘要。",
        )

        await _compact_summary_if_needed(conversation_id, trace_id)
    except Exception as exc:
        logger.warning(
            "Conversation memory summary update failed: conversation_id=%s error=%s",
            conversation_id,
            exc,
            exc_info=True,
        )
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "memory_summary_update_failed",
            "_update_memory_summary_from_sliding_window",
            result={"error": MEMORY_SUMMARY_UPDATE_FAILED_MESSAGE},
            note="滑出窗口长期记忆更新失败，但不会影响主回答完成。",
        )


async def _maybe_compact_memory_summary(conversation_id: str, trace_id: str | None = None):
    """会话存在时检查长期摘要是否超过上限。会话不存在由 `_compact_summary_if_needed` 静默返回。"""
    await _compact_summary_if_needed(conversation_id, trace_id)


async def _compact_summary_if_needed(conversation_id: str, trace_id: str | None = None) -> None:
    """长期摘要超过上限时再做一次压缩。

    早先这个协程接的是「调用方手里的 `db` + `Conversation` 对象」，于是它必然在事件循环
    线程上写库——正是 issue #187 边界一节点名的那个坑（调用方把同一个 `db` 一路传下来）。
    改成只收 `conversation_id`，摘要的读与写各自在自己那一段工作线程里完成，
    「开 session → 用 → 关」不跨线程，也不再和调用方共用会话。
    """
    summary = await asyncio.to_thread(_read_memory_summary, conversation_id)
    if summary is None:
        return
    summary_length = len(summary)
    if not summary or summary_length <= MEMORY_SUMMARY_MAX_CHARS:
        return

    await asyncio.to_thread(
        append_trace_event,
        trace_id,
        "memory_summary_compaction_triggered",
        "_compact_summary_if_needed",
        params={
            "summary_length": summary_length,
            "summary_limit": MEMORY_SUMMARY_MAX_CHARS,
        },
        creates={"transcript": summary},
        note="长期记忆超过上限，系统将对已有摘要再做一次压缩。",
    )
    next_summary = await _summarize_conversation_memory("", summary, trace_id=trace_id)
    if not next_summary:
        trimmed = _clip_text(summary, MEMORY_SUMMARY_MAX_CHARS)
        await asyncio.to_thread(_write_memory_summary_text, conversation_id, trimmed)
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "memory_summary_compaction_failed",
            "_compact_summary_if_needed",
            result={"reason": "empty_summary_or_model_failed"},
            note="长期记忆二次摘要失败，系统保留裁剪后的原摘要。",
        )
        return

    compacted = next_summary.strip()
    await asyncio.to_thread(_write_memory_summary_text, conversation_id, compacted)
    await asyncio.to_thread(
        append_trace_event,
        trace_id,
        "memory_summary_compacted",
        "_compact_summary_if_needed",
        creates={"memory_summary": compacted},
        note="长期记忆已完成二次摘要并压缩到上限内。",
    )


def _format_messages_for_summary(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        content = _clip_text(message.content, 1200)
        if not content:
            continue
        role = "用户" if message.role == "user" else "助手"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def _group_messages_into_turns(messages: list[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    current_turn: list[Message] = []

    def flush_current_turn():
        nonlocal current_turn
        if current_turn:
            turns.append(current_turn)
            current_turn = []

    for message in messages:
        if message.role == "user":
            flush_current_turn()
            current_turn = [message]
        else:
            if not current_turn:
                current_turn = [message]
            else:
                current_turn.append(message)
    flush_current_turn()
    return turns


def _group_message_texts_into_turns(messages: list[Message]) -> list[list[str]]:
    turns = []
    for turn in _group_messages_into_turns(messages):
        rendered_turn = []
        for message in turn:
            content = _clip_text(message.content, 1200)
            if not content:
                continue
            role = "用户" if message.role == "user" else "助手"
            rendered_turn.append(f"{role}：{content}")
        if rendered_turn:
            turns.append(rendered_turn)
    return turns


def _fallback_merge_summary(previous_summary: str, transcript: str) -> str:
    sections = []
    if previous_summary.strip():
        sections.append(previous_summary.strip())
    sections.append(f"滑出窗口会话记忆：\n{transcript.strip()}")
    return _clip_text("\n\n".join(sections), MEMORY_SUMMARY_MAX_CHARS)


def _fallback_compact_recent_memory(rendered_turns: list[str]) -> str:
    prefix = "近期记忆压缩：\n"
    budget = max(MEMORY_RECENT_MAX_CHARS - len(prefix), 1)
    per_turn_budget = max(120, budget // max(len(rendered_turns), 1))
    compacted_turns = []
    for index, turn_text in enumerate(rendered_turns, start=1):
        one_line = turn_text.replace("\n", " / ")
        compacted_turns.append(f"第{index}轮对话：{_clip_text(one_line, per_turn_budget)}")
    return _clip_text(prefix + "\n".join(compacted_turns), MEMORY_RECENT_MAX_CHARS)


async def _summarize_recent_memory(transcript: str, trace_id: str | None = None) -> str:
    if not MEMORY_API_KEY:
        logger.warning("Skip recent memory compaction because memory API Key is not configured")
        return ""

    system_prompt = (
        "你是企业知识库问答系统的短期记忆压缩器。"
        "请把最近对话窗口压缩成一条近期记忆，保留用户身份、偏好、约束、当前任务、明确问题、关键结论和待办。"
        "不要逐字复述长文本，不要编造未确认事实，不要输出 Markdown 标题。"
    )
    user_prompt = (
        f"压缩长度上限：{MEMORY_RECENT_MAX_TOKENS} token。\n\n"
        f"最近窗口原文：\n{transcript}\n\n"
        "请输出一条压缩后的近期记忆："
    )
    try:
        content, finish_reason = await call_chat_text_with_metadata(
            system_prompt, user_prompt, max_tokens=MEMORY_RECENT_MAX_TOKENS, stage="memory"
        )
        if finish_reason == "length":
            logger.warning(
                "Recent memory compaction truncated: model=%s max_tokens=%s",
                MEMORY_MODEL,
                MEMORY_RECENT_MAX_TOKENS,
            )
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "recent_memory_compaction_truncated",
                "_summarize_recent_memory",
                result={
                    "finish_reason": finish_reason,
                    "max_tokens": MEMORY_RECENT_MAX_TOKENS,
                    "output_length": len(content),
                },
                note="近期记忆压缩被 token 上限截断，系统拒绝该结果并改用确定性兜底。",
            )
            return ""
        return _clip_text(content.strip(), MEMORY_RECENT_MAX_CHARS)
    except Exception as exc:
        logger.warning("Recent memory compaction request failed: %s", exc, exc_info=True)
        return ""


async def _summarize_conversation_memory(
    previous_summary: str, transcript: str, trace_id: str | None = None
) -> str:
    if not MEMORY_API_KEY:
        logger.warning("Skip memory summary update because memory API Key is not configured")
        return ""

    system_prompt = (
        "你是企业知识库问答系统的会话记忆压缩器。"
        "请把给定的记忆内容压缩成可用于后续追问理解的中文摘要。"
        "只保留用户身份、长期偏好、明确约束、当前任务背景、未解决问题和待办。"
        "不要保存知识库原文大段内容，不要编造未确认事实，不要输出 Markdown 标题。"
    )
    user_prompt = (
        f"摘要长度上限：{MEMORY_SUMMARY_MAX_TOKENS} token。\n\n"
        f"已有摘要：\n{previous_summary or '（无）'}\n\n"
        f"待合并记忆：\n{transcript}\n\n"
        "请输出合并后的长期记忆："
    )
    try:
        content, finish_reason = await call_chat_text_with_metadata(
            system_prompt, user_prompt, max_tokens=MEMORY_SUMMARY_MAX_TOKENS, stage="memory"
        )
        if finish_reason == "length":
            logger.warning(
                "Memory summary truncated: model=%s max_tokens=%s",
                MEMORY_MODEL,
                MEMORY_SUMMARY_MAX_TOKENS,
            )
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "memory_summary_truncated",
                "_summarize_conversation_memory",
                result={
                    "finish_reason": finish_reason,
                    "max_tokens": MEMORY_SUMMARY_MAX_TOKENS,
                    "output_length": len(content),
                },
                note="长期记忆摘要被 token 上限截断，系统拒绝该结果并改用确定性兜底。",
            )
            return ""
        return _clip_text(content.strip(), MEMORY_SUMMARY_MAX_CHARS)
    except Exception as exc:
        logger.warning("Memory summary network request failed: %s", exc, exc_info=True)
        return ""
