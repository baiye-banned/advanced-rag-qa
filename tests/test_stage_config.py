"""Issue #317：RAG 各阶段可以独立覆盖模型与密钥。

覆盖重点是「空值继承」：没有新增环境变量的部署必须完全沿用原有 getter，
只写一个字段的阶段只固定该字段，其余字段继续跟随来源在运行期的当前值。
全部用例离线执行，只替换客户端/模型工厂，不发真实 provider 请求。
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError

import rag.llm as llm


_STAGE_ENV_NAMES = [name for names in llm._STAGE_ENV.values() for name in names]

# config.py 在导入时解析的覆盖变量；router 单独测试 TEXT_FALLBACK_* 继承。
_CONFIG_OVERRIDE_STAGE_ENVS = tuple(
    f"{stage}_{suffix}"
    for stage in ("PLANNER", "ANSWER", "MEMORY", "RERANK_LLM", "ROUTER")
    for suffix in ("API_KEY", "BASE_URL", "MODEL")
)

_CONFIG_INHERITANCE_SCRIPT = """
import json

import dotenv

# 导入 config 就会读 .env；先变成 no-op，确保断言只受显式传入的 env 影响。
dotenv.load_dotenv = lambda *args, **kwargs: False

import config

print(json.dumps({
    "planner": [config.PLANNER_API_KEY, config.PLANNER_BASE_URL, config.PLANNER_MODEL],
    "answer": [config.ANSWER_API_KEY, config.ANSWER_BASE_URL, config.ANSWER_MODEL],
    "memory": [config.MEMORY_API_KEY, config.MEMORY_BASE_URL, config.MEMORY_MODEL],
    "rerank_llm": [
        config.RERANK_LLM_API_KEY,
        config.RERANK_LLM_BASE_URL,
        config.RERANK_LLM_MODEL,
    ],
    "router": [config.ROUTER_API_KEY, config.ROUTER_BASE_URL, config.ROUTER_MODEL],
}))
"""


@pytest.fixture(autouse=True)
def _clear_stage_env(monkeypatch):
    """每条用例都从「没有任何阶段覆盖」开始，避免宿主环境或 .env 影响断言。"""
    for name in _STAGE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class _FakeChatModel:
    def __init__(self, content='{"ok": true}'):
        self.content = content

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self.content, response_metadata={"finish_reason": "stop"})

    async def astream(self, messages):
        yield self.content


class _FailingStreamModel:
    async def astream(self, messages):
        raise RuntimeError("primary stream failed")
        yield  # pragma: no cover - 让调用点保持异步生成器语义


def _capture_chat_openai(monkeypatch):
    captured = []

    def fake_chat_openai(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(llm, "ChatOpenAI", fake_chat_openai)
    return captured


def _authentication_error():
    """构造真实的 openai 401；httpx 只在内存里建响应，不产生任何网络调用。"""
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    return AuthenticationError(
        "upstream rejected the request",
        response=httpx.Response(401, request=request),
        body=None,
    )


def test_unconfigured_stages_inherit_existing_getters(monkeypatch):
    calls = []

    def fake_deepseek(**kwargs):
        calls.append(("deepseek", kwargs))
        return _FakeChatModel()

    def fake_text_fallback(**kwargs):
        calls.append(("text_fallback", kwargs))
        return _FakeChatModel()

    monkeypatch.setattr(llm, "get_deepseek_model", fake_deepseek)
    monkeypatch.setattr(llm, "get_text_fallback_model", fake_text_fallback)

    for stage in ("planner", "answer", "memory", "rerank_llm"):
        model = llm.get_stage_model(stage, streaming=False, temperature=0.2, max_tokens=64)
        assert isinstance(model, _FakeChatModel)
    router_model = llm.get_stage_model("router", streaming=True, temperature=0.3)
    fallback_model = llm.get_stage_model("text_fallback", streaming=True)
    assert isinstance(router_model, _FakeChatModel)
    assert isinstance(fallback_model, _FakeChatModel)

    # 四个 DeepSeek 家族阶段全部走原 getter；router 与 text_fallback 走后备模型 getter。
    assert calls.count(
        ("deepseek", {"streaming": False, "temperature": 0.2, "max_tokens": 64})
    ) == 4
    assert calls.count(("text_fallback", {"streaming": True, "temperature": 0.3})) == 1
    assert calls.count(("text_fallback", {"streaming": True, "temperature": 0.1})) == 1


def test_stage_override_builds_client_from_stage_values(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setenv("PLANNER_API_KEY", "planner-stage-key")
    monkeypatch.setenv("PLANNER_BASE_URL", "https://planner.example.com/")
    monkeypatch.setenv("PLANNER_MODEL", "deepseekv4pro")

    llm.get_stage_model("planner", streaming=False, temperature=0.0, max_tokens=128)

    assert captured == [
        {
            "api_key": "planner-stage-key",
            "base_url": "https://planner.example.com/v1",
            "model": "deepseek-v4-pro",
            "temperature": 0.0,
            "streaming": False,
            "timeout": 60,
            "max_retries": 0,
            "max_completion_tokens": 128,
        }
    ]


@pytest.mark.parametrize(
    ("stage", "key_name"),
    [
        ("planner", "PLANNER_API_KEY"),
        ("router", "ROUTER_API_KEY"),
    ],
)
def test_stage_key_only_override_keeps_inherited_base_url_and_model(monkeypatch, stage, key_name):
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setenv(key_name, f"{stage}-only-key")

    llm.get_stage_model(stage)

    source_key, source_base_url, source_model = llm._source_values(llm._STAGE_SOURCE[stage])
    inherited_model = (
        llm.normalize_deepseek_model(source_model)
        if stage in llm._DEEPSEEK_FAMILY_STAGES
        else source_model
    )
    assert captured == [
        {
            "api_key": f"{stage}-only-key",
            "base_url": llm.openai_base_url(source_base_url),
            "model": inherited_model,
            "temperature": 0.1,
            "streaming": False,
            "timeout": 60,
            "max_retries": 0,
        }
    ]


def test_partial_override_keeps_own_field_and_follows_runtime_source(monkeypatch):
    """只固定 MODEL 的阶段不再跟随来源；未覆盖阶段随运行期替换的 DEEPSEEK_* 变化。"""
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setenv("PLANNER_MODEL", "planner-pinned-model")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "deepseek-key-v1")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-model-v1")

    llm.get_stage_model("planner")
    llm.get_stage_model("answer")

    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "deepseek-key-v2")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-model-v2")

    llm.get_stage_model("planner")
    llm.get_stage_model("answer")

    assert captured[0]["api_key"] == "deepseek-key-v1"
    assert captured[0]["model"] == "planner-pinned-model"
    assert captured[1]["api_key"] == "deepseek-key-v1"
    assert captured[1]["model"] == "deepseek-model-v1"
    assert captured[2]["api_key"] == "deepseek-key-v2"
    assert captured[2]["model"] == "planner-pinned-model"
    assert captured[3]["api_key"] == "deepseek-key-v2"
    assert captured[3]["model"] == "deepseek-model-v2"


def test_partial_stage_override_with_empty_source_key_reports_stage(monkeypatch):
    def fail_chat_openai(**kwargs):  # pragma: no cover - 走到这里说明没有拦截缺 Key
        raise AssertionError(f"不应构造客户端：{kwargs}")

    monkeypatch.setattr(llm, "ChatOpenAI", fail_chat_openai)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("PLANNER_MODEL", "planner-pinned-model")

    with pytest.raises(llm.LLMCredentialError) as excinfo:
        llm.get_stage_model("planner")

    assert excinfo.value.stage == "planner"
    assert str(excinfo.value) == "planner API Key 未配置"


def test_inherited_source_key_missing_without_stage_override(monkeypatch):
    """完全继承时缺的是来源变量，异常不能指向未被使用的阶段变量。"""
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")

    with pytest.raises(llm.LLMCredentialError) as excinfo:
        llm.get_stage_model("planner")

    assert excinfo.value.stage is None
    assert str(excinfo.value) == "planner API Key 未配置"


def test_answer_stage_missing_key_message_points_to_answer_env(monkeypatch):
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("ANSWER_MODEL", "answer-pinned-model")

    with pytest.raises(llm.LLMCredentialError) as excinfo:
        llm.get_stage_model("answer")

    kind, message = llm._classify_deepseek_failure(excinfo.value)

    assert kind == "missing_credentials"
    assert excinfo.value.stage == "answer"
    assert "ANSWER_API_KEY" in message
    assert "DEEPSEEK_API_KEY" not in message


def test_answer_stage_invalid_key_message_points_to_answer_env():
    kind, message = llm._classify_deepseek_failure(_authentication_error(), stage="answer")

    assert kind == "invalid_credentials"
    assert "ANSWER_API_KEY" in message
    assert "DEEPSEEK_API_KEY" not in message


def test_unknown_stage_keeps_deepseek_credential_wording():
    """完全继承的部署仍看到原有 DeepSeek 文案。"""
    kind, message = llm._classify_deepseek_failure(_authentication_error())

    assert kind == "invalid_credentials"
    assert message == llm.DEEPSEEK_API_KEY_INVALID_MESSAGE


def test_answer_stream_uses_stage_specific_credential_message_when_overridden(monkeypatch):
    async def failing_stream(model, messages):
        raise _authentication_error()
        yield ""  # pragma: no cover - 只验证异常分流

    _capture_chat_openai(monkeypatch)
    monkeypatch.setattr(llm, "_stream_model_chunks", failing_stream)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", False)
    monkeypatch.setenv("ANSWER_API_KEY", "answer-stage-key")

    async def collect():
        return [event async for event in llm.stream_answer_events("问题", "上下文")]

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "ANSWER_API_KEY" in events[0]["message"]
    assert "DEEPSEEK_API_KEY" not in events[0]["message"]


def test_planner_without_overrides_inherits_source_key(monkeypatch):
    """源 key 非空且 planner 未覆盖时，空的阶段配置继续沿用来源客户端。"""
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "deepseek-source-secret")

    llm.get_stage_model("planner")

    assert len(captured) == 1
    assert captured[0]["api_key"] == "deepseek-source-secret"
    assert captured[0]["base_url"] == llm.openai_base_url(llm.DEEPSEEK_BASE_URL)
    assert captured[0]["model"] == llm.normalize_deepseek_model(llm.DEEPSEEK_MODEL)


def test_call_helpers_forward_stage_to_model_selector(monkeypatch):
    captured = []

    def fake_get_stage_model(stage, **kwargs):
        captured.append((stage, kwargs))
        return _FakeChatModel('{"items": []}')

    monkeypatch.setattr(llm, "get_stage_model", fake_get_stage_model)

    assert asyncio.run(
        llm.call_chat_text("system", "user", max_tokens=123, stage="memory")
    ) == '{"items": []}'
    assert asyncio.run(
        llm.call_chat_json("system", "user", max_tokens=77, stage="rerank_llm")
    ) == {"items": []}

    assert captured == [
        (
            "memory",
            {"streaming": False, "temperature": 0.1, "max_tokens": 123},
        ),
        (
            "rerank_llm",
            {"streaming": False, "temperature": 0.1, "max_tokens": 77},
        ),
    ]


def test_router_and_answer_stream_use_dedicated_stages(monkeypatch):
    captured = []

    def fake_get_stage_model(stage, **kwargs):
        captured.append((stage, kwargs))
        if stage == "answer":
            return _FailingStreamModel()
        if stage == "text_fallback":
            return _FakeChatModel("后备回答")
        return _FakeChatModel('{"need_rag": true, "route": "rag"}')

    monkeypatch.setattr(llm, "get_stage_model", fake_get_stage_model)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_ENABLED", True)
    monkeypatch.setattr(llm, "TEXT_FALLBACK_API_KEY", "offline-fallback-key")

    assert asyncio.run(llm.call_router_json({"question": "制度在哪里"})) == {
        "need_rag": True,
        "route": "rag",
    }
    assert captured[0] == (
        "router",
        {"streaming": False, "temperature": 0, "max_tokens": 300},
    )

    async def collect():
        return [event async for event in llm.stream_answer_events("问题", "上下文")]

    events = asyncio.run(collect())

    assert events == [
        {
            "type": "reset",
            "reason": "text_fallback",
            "message": "DeepSeek 生成中断，已切换到文本后备模型重新生成本轮回答。",
        },
        "后备回答",
    ]
    assert captured[1] == ("answer", {"streaming": True, "temperature": 0.1})
    assert captured[2] == ("text_fallback", {"streaming": True, "temperature": 0.1})


def test_config_module_inherits_stage_values_at_import():
    """config.py 导入期的继承：无阶段变量时逐字跟随来源配置。"""
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in _CONFIG_OVERRIDE_STAGE_ENVS
    }
    env.update(
        {
            "DEEPSEEK_API_KEY": "deepseek-sentinel-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example.com",
            "DEEPSEEK_MODEL": "deepseek-sentinel-model",
            "TEXT_FALLBACK_API_KEY": "fallback-sentinel-key",
            "TEXT_FALLBACK_BASE_URL": "https://fallback.example.com",
            "TEXT_FALLBACK_MODEL": "fallback-sentinel-model",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _CONFIG_INHERITANCE_SCRIPT],
        cwd=Path(__file__).resolve().parents[1] / "backend",
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    stages = json.loads(completed.stdout)

    deepseek_values = [
        "deepseek-sentinel-key",
        "https://deepseek.example.com",
        "deepseek-sentinel-model",
    ]
    fallback_values = [
        "fallback-sentinel-key",
        "https://fallback.example.com",
        "fallback-sentinel-model",
    ]
    for stage in ("planner", "answer", "memory", "rerank_llm"):
        assert stages[stage] == deepseek_values
    assert stages["router"] == fallback_values
