"""回归（issue #21）：流式中途失败切换后备模型时，必须作废已输出内容，避免重复拼接与落库。

打桩方式：替换 `rag.llm._stream_model_chunks`，第一次调用（DeepSeek）先吐两段内容再断开，
第二次调用（文本后备模型）从头输出完整回答。
会话侧只打桩真正的边界（鉴权/库/知识库解析/检索），SSE 组帧、证据校验与消息落库跑真实实现。
"""

import asyncio

import httpx
from openai import APIConnectionError

import rag.llm as llm
from conftest import FakeKnowledgeBase, collect_stream, frames_of_type, parse_sse_frames
from model.models import User
from schema.schemas import ChatRequest
from service import chat_service


DEEPSEEK_CHUNKS = ["根据《员工手册》考勤管理", "，迟到30分钟以内"]
FALLBACK_CHUNKS = ["根据《员工手册》考勤管理章节，", "迟到30分钟以内罚款50元。", "（以上为完整回答）"]
FALLBACK_TEXT = "".join(FALLBACK_CHUNKS)
CHUNK = {"file_name": "制度.txt", "content": "迟到规则", "file_id": 1, "chunk_id": "a"}


def _stub_fallback_stream(monkeypatch, *, deepseek_chunks, fallback_chunks, failure=None):
    """第一次调用按 deepseek_chunks 输出后（可选）抛异常，第二次调用输出 fallback_chunks。"""
    calls = {"count": 0}

    async def fake_stream(model, messages):
        calls["count"] += 1
        if calls["count"] == 1:
            for piece in deepseek_chunks:
                yield piece
            if failure is not None:
                raise failure
            return
        for piece in fallback_chunks:
            yield piece

    monkeypatch.setattr(llm, "_stream_model_chunks", fake_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    return calls


def _collect_events(*args, **kwargs):
    async def _run():
        return [event async for event in llm.stream_answer_events(*args, **kwargs)]

    return asyncio.run(_run())


def _render_like_frontend(events):
    """按前端协议渲染事件流：reset 清空缓冲，content 追加，error 直接失败。"""
    rendered = ""
    resets = 0
    for event in events:
        if isinstance(event, dict):
            if event.get("type") == "reset":
                rendered = ""
                resets += 1
                continue
            if event.get("type") == "error":
                raise AssertionError(f"unexpected error event: {event.get('message')}")
            chunk = event.get("content", "")
        else:
            chunk = event
        rendered += chunk
    return rendered, resets


def _chunk_text(event):
    return event.get("content", "") if isinstance(event, dict) else event


def _reset_index(events):
    return next(
        index
        for index, event in enumerate(events)
        if isinstance(event, dict) and event.get("type") == "reset"
    )


def test_mid_stream_failure_emits_reset_before_fallback_content(monkeypatch):
    calls = _stub_fallback_stream(
        monkeypatch,
        deepseek_chunks=DEEPSEEK_CHUNKS,
        fallback_chunks=FALLBACK_CHUNKS,
        failure=RuntimeError("peer closed connection without sending complete message body"),
    )

    events = _collect_events("迟到30分钟以内怎么罚款？", "上下文", memory_context="")

    assert calls["count"] == 2
    rendered, resets = _render_like_frontend(events)
    assert resets == 1
    assert rendered == FALLBACK_TEXT
    assert "根据《员工手册》考勤管理，迟到30分钟以内" not in rendered

    index = _reset_index(events)
    assert _chunk_text(events[index - 1]) == DEEPSEEK_CHUNKS[-1]
    assert _chunk_text(events[index + 1]) == FALLBACK_CHUNKS[0]
    assert events[index].get("reason") == "text_fallback"


def test_successful_deepseek_stream_does_not_emit_reset(monkeypatch):
    _stub_fallback_stream(
        monkeypatch,
        deepseek_chunks=["完整回答"],
        fallback_chunks=[],
    )

    events = _collect_events("问题", "上下文", memory_context="")

    assert not [event for event in events if isinstance(event, dict) and event.get("type") == "reset"]
    rendered, resets = _render_like_frontend(events)
    assert resets == 0
    assert rendered == "完整回答"


def test_fallback_disabled_still_reports_error_without_reset(monkeypatch):
    _stub_fallback_stream(
        monkeypatch,
        deepseek_chunks=DEEPSEEK_CHUNKS,
        fallback_chunks=[],
        failure=RuntimeError("connection reset"),
    )
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)

    events = _collect_events("问题", "上下文", memory_context="")

    assert [event for event in events if isinstance(event, dict) and event.get("type") == "error"]
    assert not [event for event in events if isinstance(event, dict) and event.get("type") == "reset"]


def _patch_chat_service_boundaries(monkeypatch, fake_db, trace_cls):
    monkeypatch.setattr(
        chat_service,
        "authenticate",
        lambda db, authorization: db.query(User).filter_by(username="alice").first(),
    )
    monkeypatch.setattr(chat_service, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(chat_service, "TraceRecorder", trace_cls)
    monkeypatch.setattr(chat_service, "resolve_knowledge_base", lambda db, kid, user_id: FakeKnowledgeBase())

    async def fake_build_effective_question(question, attachments):
        return question, {"status": "skipped"}

    async def fake_recent_memory_text(*_args, **_kwargs):
        return ""

    async def fake_decide_need_rag(*_args, **_kwargs):
        return {"need_rag": True, "route": "rag", "confidence": 1.0, "source": "test", "reason": "needs retrieval"}

    async def fake_retrieve_knowledge(question, knowledge_base_id, db, trace_recorder=None):
        return (
            [dict(CHUNK)],
            {"query_plan": {}, "routes": [], "rrf": [], "rerank": {"status": "done", "items": []}},
        )

    monkeypatch.setattr(chat_service, "_build_effective_question", fake_build_effective_question)
    monkeypatch.setattr(chat_service, "_build_recent_memory_text", fake_recent_memory_text)
    monkeypatch.setattr(chat_service, "decide_need_rag", fake_decide_need_rag)
    monkeypatch.setattr(chat_service, "retrieve_knowledge", fake_retrieve_knowledge)
    monkeypatch.setattr(chat_service, "_schedule_memory_summary_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "schedule_ragas_evaluation", lambda *args, **kwargs: None)


def test_chat_service_persists_only_content_after_reset(monkeypatch, fake_db, trace_recorder_cls):
    _patch_chat_service_boundaries(monkeypatch, fake_db, trace_recorder_cls)
    _stub_fallback_stream(
        monkeypatch,
        deepseek_chunks=DEEPSEEK_CHUNKS,
        fallback_chunks=FALLBACK_CHUNKS,
        failure=RuntimeError("peer closed connection without sending complete message body"),
    )

    async def real_chain(question, context, memory_context, trace, use_rag=True):
        async for event in llm.stream_answer_events(question, context, memory_context, trace, use_rag):
            yield event

    monkeypatch.setattr(chat_service, "stream_rag_answer", real_chain)

    response = asyncio.run(
        chat_service.stream_chat(ChatRequest(question="迟到30分钟以内怎么罚款？"), authorization="Bearer token")
    )
    body = collect_stream(response.body_iterator)

    frames = parse_sse_frames(body)
    resets = frames_of_type(frames, "reset")
    assert len(resets) == 1
    assert resets[0]["reason"] == "text_fallback"
    assert frames_of_type(frames, "error") == []

    saved = fake_db.added_by_role("assistant")
    assert len(saved) == 1
    assert saved[0].content == FALLBACK_TEXT
    assert "根据《员工手册》考勤管理，迟到30分钟以内根据《员工手册》考勤管理章节，" not in saved[0].content
    assert fake_db.closed is True

    # 前端按同一份事件流渲染出的最终文本必须与落库内容一致
    rendered, frontend_resets = _render_like_frontend(_events_from_sse(frames))
    assert frontend_resets == 1
    assert rendered == saved[0].content


def _events_from_sse(frames):
    """把已解析的 SSE 帧还原成流式事件（前端消费的形态）。"""
    events = []
    for frame in frames:
        if "type" in frame:
            events.append(frame)
        else:
            events.append(frame.get("content", ""))
    return events


# --- issue #292：生成失败按「缺 Key / 网络 / 后备可用性」分流用户文案 -------

def _error_messages(events):
    return [
        event["message"]
        for event in events
        if isinstance(event, dict) and event.get("type") == "error"
    ]


def _has_reset(events):
    return any(isinstance(event, dict) and event.get("type") == "reset" for event in events)


def _fallback_only_stream(monkeypatch):
    """DeepSeek 在配置校验阶段就失败，因此只会走到后备模型这一次流式调用。"""

    async def fake_stream(model, messages):
        for piece in FALLBACK_CHUNKS:
            yield piece

    monkeypatch.setattr(llm, "_stream_model_chunks", fake_stream)


def test_missing_deepseek_key_reports_config_hint_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)

    events = _collect_events("问题", "上下文", memory_context="")

    messages = _error_messages(events)
    assert len(messages) == 1
    assert "DEEPSEEK_API_KEY" in messages[0]
    assert "TEXT_FALLBACK_ENABLED" in messages[0]
    assert "网络" not in messages[0]
    assert not _has_reset(events)


def test_missing_deepseek_key_uses_fallback_when_available(monkeypatch):
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "sk-test")
    _fallback_only_stream(monkeypatch)

    events = _collect_events("问题", "上下文", memory_context="")

    assert _error_messages(events) == []
    rendered, resets = _render_like_frontend(events)
    assert resets == 1
    assert rendered == FALLBACK_TEXT


def test_missing_deepseek_key_reports_fallback_key_when_unconfigured(monkeypatch):
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "")

    events = _collect_events("问题", "上下文", memory_context="")

    messages = _error_messages(events)
    assert len(messages) == 1
    assert "DEEPSEEK_API_KEY" in messages[0]
    assert "TEXT_FALLBACK_API_KEY" in messages[0]
    assert "DASHSCOPE_API_KEY" in messages[0]
    assert "网络" not in messages[0]
    assert not _has_reset(events)


def test_transport_failure_keeps_network_wording(monkeypatch):
    async def failing_stream(model, messages):
        raise APIConnectionError(
            request=httpx.Request("POST", "https://api.deepseek.test/v1/chat/completions")
        )
        yield ""  # pragma: no cover - 该分支只验证异常分流

    monkeypatch.setattr(llm, "_stream_model_chunks", failing_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)

    events = _collect_events("问题", "上下文", memory_context="")

    messages = _error_messages(events)
    assert len(messages) == 1
    assert "网络/上游请求失败" in messages[0]
    assert "DEEPSEEK_API_KEY 未配置" not in messages[0]
