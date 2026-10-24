"""回归（issue #21）：流式中途失败切换后备模型时，必须作废已输出内容，避免重复拼接与落库。

打桩方式：替换 `rag.llm._stream_model_chunks`，第一次调用（DeepSeek）先吐两段内容再断开，
第二次调用（文本后备模型）从头输出完整回答。
会话侧只打桩真正的边界（鉴权/库/知识库解析/检索），SSE 组帧、证据校验与消息落库跑真实实现。
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

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


# --- issue #292 返修：content_filter 流式可达性 / 传输分类 / trace 同源 -------

def _request():
    return httpx.Request("POST", "https://api.deepseek.test/v1/chat/completions")


def _status_error(cls, status_code, message="upstream rejected the request"):
    """构造真实的 openai 状态异常；httpx 只在内存里建响应，不产生任何网络调用。"""
    return cls(message, response=httpx.Response(status_code, request=_request()), body=None)


def _fake_chunk(content="", finish_reason=None, metadata_key="response_metadata"):
    """模拟 AIMessageChunk：finish_reason 可能落在 response_metadata 或 additional_kwargs。"""
    chunk = SimpleNamespace(content=content, response_metadata={}, additional_kwargs={})
    if finish_reason is not None:
        setattr(chunk, metadata_key, {"finish_reason": finish_reason})
    return chunk


def _stream_text(events):
    return "".join(event for event in events if isinstance(event, str))


@pytest.mark.parametrize("metadata_key", ["response_metadata", "additional_kwargs"])
def test_content_filter_chunk_emits_single_error_and_skips_fallback(
    monkeypatch, trace_recorder_cls, metadata_key
):
    """流式 content_filter 必须变成唯一的 error 事件：不静默、不双发、不切后备模型。"""
    calls = {"count": 0}

    async def fake_stream(model, messages):
        calls["count"] += 1
        yield _fake_chunk("半截回答")
        yield _fake_chunk("", "content_filter", metadata_key)
        yield _fake_chunk("过滤后不应继续输出", "stop")

    monkeypatch.setattr(llm, "_stream_model_chunks", fake_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "sk-test")
    trace = trace_recorder_cls()

    events = _collect_events("问题", "上下文", memory_context="", trace=trace)

    assert calls["count"] == 1, "content_filter 不是传输故障，不得切换后备模型"
    assert _error_messages(events) == [llm.DEEPSEEK_CONTENT_FILTERED_MESSAGE]
    assert _stream_text(events) == "半截回答"
    assert not _has_reset(events)

    failed = trace.event("langchain_generation_failed")
    assert failed["result"]["failure_kind"] == "content_filtered"
    assert failed["result"]["error"] == llm.DEEPSEEK_CONTENT_FILTERED_MESSAGE


def test_content_filter_on_empty_stream_is_not_silent(monkeypatch):
    """只有终止 chunk、没有任何正文时，也必须发 error；否则消费方看到的是空流正常结束。"""
    async def fake_stream(model, messages):
        yield _fake_chunk("", "content_filter")

    monkeypatch.setattr(llm, "_stream_model_chunks", fake_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)

    events = _collect_events("问题", "上下文", memory_context="")

    assert events == [{"type": "error", "message": llm.DEEPSEEK_CONTENT_FILTERED_MESSAGE}]


def test_fallback_content_filter_uses_neutral_message(monkeypatch, trace_recorder_cls):
    """拦截内容的是后备模型，文案里不能出现 DeepSeek，trace 与 SSE 必须同源。"""
    calls = {"count": 0}

    async def fake_stream(model, messages):
        calls["count"] += 1
        if calls["count"] == 1:
            raise APIConnectionError(request=_request())
        yield _fake_chunk("后备半截回答")
        yield _fake_chunk("", "content_filter")

    monkeypatch.setattr(llm, "_stream_model_chunks", fake_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "sk-test")
    trace = trace_recorder_cls()

    events = _collect_events("问题", "上下文", memory_context="", trace=trace)

    assert calls["count"] == 2
    assert _error_messages(events) == [llm.TEXT_FALLBACK_CONTENT_FILTERED_MESSAGE]
    assert "DeepSeek" not in llm.TEXT_FALLBACK_CONTENT_FILTERED_MESSAGE
    failed = trace.event("langchain_text_fallback_failed")
    assert failed["result"]["failure_kind"] == "content_filtered"
    assert failed["result"]["error"] == llm.TEXT_FALLBACK_CONTENT_FILTERED_MESSAGE


def test_normal_stream_is_unaffected_by_content_filter_handling(monkeypatch):
    async def fake_stream(model, messages):
        yield _fake_chunk("完整")
        yield _fake_chunk("回答", "stop")

    monkeypatch.setattr(llm, "_stream_model_chunks", fake_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")

    events = _collect_events("问题", "上下文", memory_context="")

    assert events == ["完整", "回答"]
    assert _error_messages(events) == []


def test_api_status_400_content_exists_risk_is_content_filtered(monkeypatch):
    """DeepSeek 也可能用 400 Content Exists Risk 拒绝；归入内容过滤且不切后备模型。"""
    async def failing_stream(model, messages):
        raise _status_error(
            APIStatusError,
            400,
            "Error code: 400 - {'error': {'message': 'Content Exists Risk'}}",
        )
        yield ""  # pragma: no cover - 只验证异常分流

    monkeypatch.setattr(llm, "_stream_model_chunks", failing_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "sk-test")

    events = _collect_events("问题", "上下文", memory_context="")

    assert _error_messages(events) == [llm.DEEPSEEK_CONTENT_FILTERED_MESSAGE]
    assert not _has_reset(events)


def test_generation_failure_trace_uses_the_same_prefix_as_sse(monkeypatch, trace_recorder_cls):
    """缺 Key 时 trace 与用户看到的文案必须同源，不能一个说配置、一个说网络。"""
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)
    trace = trace_recorder_cls()

    events = _collect_events("问题", "上下文", memory_context="", trace=trace)

    messages = _error_messages(events)
    assert len(messages) == 1
    failed = trace.event("langchain_generation_failed")
    assert failed["result"]["failure_kind"] == "missing_credentials"
    assert failed["result"]["error"] == llm.DEEPSEEK_API_KEY_MISSING_MESSAGE
    assert messages[0].startswith(failed["result"]["error"])


@pytest.mark.parametrize("status_code", [401])
def test_authentication_error_reports_invalid_key(monkeypatch, status_code):
    """401 是凭据被拒绝，不是网络故障，也不能误报成「未配置」。"""
    async def failing_stream(model, messages):
        raise _status_error(AuthenticationError, status_code)
        yield ""  # pragma: no cover - 只验证异常分流

    monkeypatch.setattr(llm, "_stream_model_chunks", failing_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)

    events = _collect_events("问题", "上下文", memory_context="")

    messages = _error_messages(events)
    assert len(messages) == 1
    assert messages[0].startswith(llm.DEEPSEEK_API_KEY_INVALID_MESSAGE)
    assert "未配置" not in messages[0]
    assert "网络" not in messages[0]


def test_permission_denied_403_reports_invalid_key(monkeypatch, trace_recorder_cls):
    """SDK 把 403 映射为 PermissionDeniedError（与 AuthenticationError 兄弟），不是网络故障。

    这里必须构造真实的 PermissionDeniedError：用 AuthenticationError 伪造 403 无法覆盖
    SDK 的实际映射，也就测不出 403 是否被漏判。
    """
    exc = _status_error(PermissionDeniedError, 403)
    assert llm._classify_deepseek_failure(exc) == (
        "invalid_credentials",
        llm.DEEPSEEK_API_KEY_INVALID_MESSAGE,
    )
    assert llm._is_transport_error(exc) is False

    async def failing_stream(model, messages):
        raise exc
        yield ""  # pragma: no cover - 只验证异常分流

    monkeypatch.setattr(llm, "_stream_model_chunks", failing_stream)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)
    trace = trace_recorder_cls()

    events = _collect_events("问题", "上下文", memory_context="", trace=trace)

    messages = _error_messages(events)
    assert len(messages) == 1
    assert messages[0].startswith(llm.DEEPSEEK_API_KEY_INVALID_MESSAGE)
    for forbidden in ("未配置", "网络", "生成失败"):
        assert forbidden not in messages[0]
    failed = trace.event("langchain_generation_failed")
    assert failed["result"]["failure_kind"] == "invalid_credentials"
    assert failed["result"]["error"] == llm.DEEPSEEK_API_KEY_INVALID_MESSAGE


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (APITimeoutError(request=_request()), True),
        (_status_error(RateLimitError, 429), True),
        (_status_error(APIStatusError, 500), True),
        (_status_error(APIStatusError, 503), True),
        (_status_error(APIStatusError, 400), False),
        (_status_error(PermissionDeniedError, 403), False),
    ],
)
def test_is_transport_error_classifies_status_codes(exc, expected):
    """纯函数分类，零网络：超时/429/5xx 是传输类，400 不是。"""
    assert llm._is_transport_error(exc) is expected


def test_chat_service_error_event_without_message_uses_neutral_fallback(
    monkeypatch, fake_db, trace_recorder_cls
):
    """下游 error 事件缺 message 时，兜底文案必须是中性失败提示，不能再甩锅网络。"""
    _patch_chat_service_boundaries(monkeypatch, fake_db, trace_recorder_cls)

    async def error_chain(*_args, **_kwargs):
        yield {"type": "error"}

    monkeypatch.setattr(chat_service, "stream_rag_answer", error_chain)

    response = asyncio.run(
        chat_service.stream_chat(ChatRequest(question="迟到怎么处理"), authorization="Bearer token")
    )
    frames = parse_sse_frames(collect_stream(response.body_iterator))

    assert frames_of_type(frames, "error") == [
        {"type": "error", "message": llm.ANSWER_GENERATION_FAILED_MESSAGE}
    ]
    trace = trace_recorder_cls.instances[-1]
    assert trace.event("generation_failed")["result"]["message"] == llm.ANSWER_GENERATION_FAILED_MESSAGE
    assert "网络" not in llm.ANSWER_GENERATION_FAILED_MESSAGE

