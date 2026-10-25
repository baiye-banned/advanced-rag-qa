"""回归（issue #264）：记忆压缩必须按 token 预算生成，并拒绝被截断的摘要。

字符上限只约束落库文本，提示词里的「上限」必须与模型实际生效的 token 预算一致。
非流式响应的 finish_reason=length 是截断的唯一信号：一旦出现，半截摘要只允许出现在
服务端告警的元数据里（模型、token 上限、长度），绝不返回或落库。
"""

import asyncio
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import rag.llm as llm
from rag import memory_service


PARTIAL_SUMMARY = "这是一段被截断的半句摘要，不应被返回或落库"


class _FakeMessage:
    def __init__(self, content, response_metadata):
        self.content = content
        self.response_metadata = response_metadata


class _FakeModel:
    def __init__(self, message):
        self.message = message

    async def ainvoke(self, messages):
        return self.message


def _patch_model(monkeypatch, message):
    captured = []

    def fake_get_deepseek_model(**kwargs):
        captured.append(kwargs)
        return _FakeModel(message)

    monkeypatch.setattr(llm, "get_deepseek_model", fake_get_deepseek_model)
    return captured


def _patch_summarizer(monkeypatch, result, finish_reason):
    """替换非流式调用，记录提示词并返回指定终态。"""
    calls = []

    async def fake_call(system_prompt, user_prompt, *, max_tokens, stage="planner"):
        calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "stage": stage,
            }
        )
        return result, finish_reason

    monkeypatch.setattr(memory_service, "call_chat_text_with_metadata", fake_call)
    monkeypatch.setattr(memory_service, "MEMORY_API_KEY", "offline-test-key")
    return calls


def _patch_trace(monkeypatch):
    events = []

    def fake_append_trace_event(trace_id, stage, function, **kwargs):
        events.append(
            {"trace_id": trace_id, "stage": stage, "function": function, **kwargs}
        )

    monkeypatch.setattr(memory_service, "append_trace_event", fake_append_trace_event)
    return events


def test_call_chat_text_with_metadata_returns_finish_reason_without_network(monkeypatch):
    captured = _patch_model(
        monkeypatch,
        _FakeMessage(PARTIAL_SUMMARY, {"finish_reason": "length"}),
    )

    content, finish_reason = asyncio.run(
        llm.call_chat_text_with_metadata("system", "user", max_tokens=321)
    )

    assert content == PARTIAL_SUMMARY
    assert finish_reason == "length"
    assert captured == [{"streaming": False, "temperature": 0.1, "max_tokens": 321}]


def test_call_chat_text_with_metadata_returns_none_when_reason_is_absent(monkeypatch):
    _patch_model(monkeypatch, _FakeMessage("完整文本", {}))

    content, finish_reason = asyncio.run(
        llm.call_chat_text_with_metadata("system", "user", max_tokens=321)
    )

    assert content == "完整文本"
    assert finish_reason is None


def test_call_chat_text_stays_text_only_for_existing_callers(monkeypatch):
    _patch_model(monkeypatch, _FakeMessage("完整文本", {"finish_reason": "length"}))

    assert asyncio.run(llm.call_chat_text("system", "user", max_tokens=321)) == "完整文本"


def test_conversation_summary_rejects_length_truncation(monkeypatch, caplog):
    calls = _patch_summarizer(monkeypatch, PARTIAL_SUMMARY, "length")
    events = _patch_trace(monkeypatch)
    caplog.set_level(logging.WARNING)

    result = asyncio.run(
        memory_service._summarize_conversation_memory(
            "已有摘要", "待合并记忆", trace_id="trace-264"
        )
    )

    assert result == "", "截断的半句摘要不能作为长期记忆返回"
    assert calls[0]["max_tokens"] == config.MEMORY_SUMMARY_MAX_TOKENS
    assert f"{config.MEMORY_SUMMARY_MAX_TOKENS} token" in calls[0]["user_prompt"]
    assert f"{config.MEMORY_SUMMARY_MAX_CHARS} 字" not in calls[0]["user_prompt"]

    truncated = [event for event in events if event["stage"] == "memory_summary_truncated"]
    assert len(truncated) == 1
    assert truncated[0]["trace_id"] == "trace-264"
    assert truncated[0]["function"] == "_summarize_conversation_memory"
    assert truncated[0]["result"] == {
        "finish_reason": "length",
        "max_tokens": config.MEMORY_SUMMARY_MAX_TOKENS,
        "output_length": len(PARTIAL_SUMMARY),
    }
    assert PARTIAL_SUMMARY not in repr(events), "轨迹事件不能携带截断正文"
    assert PARTIAL_SUMMARY not in caplog.text, "日志不能携带截断正文"
    assert "max_tokens=" in caplog.text


def test_recent_memory_summary_rejects_length_truncation(monkeypatch):
    calls = _patch_summarizer(monkeypatch, PARTIAL_SUMMARY, "length")
    events = _patch_trace(monkeypatch)

    result = asyncio.run(
        memory_service._summarize_recent_memory("最近窗口原文", trace_id="trace-264")
    )

    assert result == ""
    assert calls[0]["max_tokens"] == config.MEMORY_RECENT_MAX_TOKENS
    assert f"{config.MEMORY_RECENT_MAX_TOKENS} token" in calls[0]["user_prompt"]
    assert f"{config.MEMORY_RECENT_MAX_CHARS} 字" not in calls[0]["user_prompt"]
    truncated = [event for event in events if event["stage"] == "recent_memory_compaction_truncated"]
    assert truncated[0]["result"] == {
        "finish_reason": "length",
        "max_tokens": config.MEMORY_RECENT_MAX_TOKENS,
        "output_length": len(PARTIAL_SUMMARY),
    }


def test_recent_memory_caller_uses_fallback_after_truncation(monkeypatch):
    _patch_summarizer(monkeypatch, PARTIAL_SUMMARY, "length")
    events = _patch_trace(monkeypatch)
    messages = []
    for index in range(8):
        messages.append(
            SimpleNamespace(
                role="user" if index % 2 == 0 else "assistant",
                content=f"第{index}条很长的近期消息" + "内容" * 800,
            )
        )

    result = asyncio.run(
        memory_service._format_recent_memory_messages(messages, trace_id="trace-264")
    )

    assert result.startswith("近期记忆压缩：\n第1轮对话：")
    assert PARTIAL_SUMMARY not in result
    stages = [event["stage"] for event in events]
    assert "recent_memory_compaction_truncated" in stages
    assert "recent_memory_compaction_fallback" in stages
    assert "recent_memory_compacted" not in stages


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_conversation_summary_keeps_text_for_normal_completion(monkeypatch, finish_reason):
    _patch_summarizer(monkeypatch, "完整摘要", finish_reason)
    _patch_trace(monkeypatch)

    result = asyncio.run(
        memory_service._summarize_conversation_memory("", "原文", trace_id="trace-264")
    )

    assert result == "完整摘要"


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_recent_memory_summary_keeps_text_for_normal_completion(monkeypatch, finish_reason):
    _patch_summarizer(monkeypatch, "完整近期记忆", finish_reason)
    _patch_trace(monkeypatch)

    result = asyncio.run(
        memory_service._summarize_recent_memory("原文", trace_id="trace-264")
    )

    assert result == "完整近期记忆"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MEMORY_SUMMARY_MAX_TOKENS", config.MEMORY_SUMMARY_MAX_TOKENS),
        ("MEMORY_RECENT_MAX_TOKENS", config.MEMORY_RECENT_MAX_TOKENS),
    ],
)
def test_token_config_defaults_match_env_example(name, value):
    env_example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(
        encoding="utf-8"
    )
    match = re.search(rf"^{name}=(\d+)$", env_example, flags=re.M)

    assert match is not None, f".env.example 缺少 {name}"
    assert int(match.group(1)) == value
    assert value == 900
