"""Issue #317：RAG 各阶段可以独立覆盖模型与密钥。

覆盖重点是「空值继承」：没有新增环境变量的部署必须完全沿用原有 getter，
只写一个阶段的 API Key 时也必须继续继承该阶段来源的 BASE_URL 与 MODEL。
全部用例离线执行，只替换客户端/模型工厂，不发真实 provider 请求。
"""

import asyncio
from types import SimpleNamespace

import pytest

import rag.llm as llm


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

    for stage in ("planner", "answer", "memory", "rerank_llm", "ragas"):
        model = llm.get_stage_model(stage, streaming=False, temperature=0.2, max_tokens=64)
        assert isinstance(model, _FakeChatModel)
    router_model = llm.get_stage_model("router", streaming=True, temperature=0.3)
    fallback_model = llm.get_stage_model("text_fallback", streaming=True)
    assert isinstance(router_model, _FakeChatModel)
    assert isinstance(fallback_model, _FakeChatModel)

    expected_deepseek = [stage for stage, _ in calls if stage == "deepseek"]
    assert len(expected_deepseek) == 5
    assert calls.count(("deepseek", {"streaming": False, "temperature": 0.2, "max_tokens": 64})) == 5
    assert calls.count(("text_fallback", {"streaming": True, "temperature": 0.3})) == 1
    assert calls.count(("text_fallback", {"streaming": True, "temperature": 0.1})) == 1


def test_stage_override_builds_client_from_stage_values(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setattr(llm, "PLANNER_API_KEY", "planner-stage-key")
    monkeypatch.setattr(llm, "PLANNER_BASE_URL", "https://planner.example.com/")
    monkeypatch.setattr(llm, "PLANNER_MODEL", "deepseekv4pro")

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
    ("stage", "key_name", "base_url_name", "model_name", "expected_model"),
    [
        ("planner", "PLANNER_API_KEY", "PLANNER_BASE_URL", "PLANNER_MODEL", None),
        ("router", "ROUTER_API_KEY", "ROUTER_BASE_URL", "ROUTER_MODEL", None),
    ],
)
def test_stage_key_only_override_keeps_inherited_base_url_and_model(
    monkeypatch, stage, key_name, base_url_name, model_name, expected_model
):
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setattr(llm, key_name, f"{stage}-only-key")

    llm.get_stage_model(stage)

    inherited_model = getattr(llm, model_name)
    if stage in llm._DEEPSEEK_FAMILY_STAGES:
        inherited_model = llm.normalize_deepseek_model(inherited_model)
    assert captured == [
        {
            "api_key": f"{stage}-only-key",
            "base_url": llm.openai_base_url(getattr(llm, base_url_name)),
            "model": expected_model or inherited_model,
            "temperature": 0.1,
            "streaming": False,
            "timeout": 60,
            "max_retries": 0,
        }
    ]


def test_ragas_stage_keeps_ragas_llm_model_with_stage_key_and_url(monkeypatch):
    """RAGAS 只新增地址与密钥覆盖，模型名继续取既有 RAGAS_LLM_MODEL。"""
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setattr(llm, "RAGAS_API_KEY", "ragas-stage-key")
    monkeypatch.setattr(llm, "RAGAS_BASE_URL", "https://ragas.example.com")
    monkeypatch.setattr(llm, "RAGAS_LLM_MODEL", "deepseekv4flash")

    llm.get_stage_model("ragas")

    assert captured[0]["api_key"] == "ragas-stage-key"
    assert captured[0]["base_url"] == "https://ragas.example.com/v1"
    assert captured[0]["model"] == "deepseek-v4-flash"


@pytest.mark.parametrize("source_key", ["", "deepseek-source-secret"])
def test_missing_stage_key_raises_stage_specific_error_without_leaking(monkeypatch, source_key):
    def fail_chat_openai(**kwargs):  # pragma: no cover - 走到这里说明没有拦截缺 Key
        raise AssertionError(f"不应构造客户端：{kwargs}")

    monkeypatch.setattr(llm, "ChatOpenAI", fail_chat_openai)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", source_key)
    monkeypatch.setattr(llm, "PLANNER_API_KEY", "")

    with pytest.raises(RuntimeError) as excinfo:
        llm.get_stage_model("planner")

    assert isinstance(excinfo.value, llm.LLMCredentialError)
    assert str(excinfo.value) == "planner API Key 未配置"
    assert "deepseek-source-secret" not in str(excinfo.value)


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
