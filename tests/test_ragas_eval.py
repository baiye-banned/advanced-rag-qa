import asyncio
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag import ragas_eval


def test_truncate_text_preserves_falsy_values():
    assert ragas_eval._truncate_text(0, 10) == "0"
    assert ragas_eval._truncate_text(False, 10) == "False"
    assert ragas_eval._truncate_text(None, 10) == ""


def test_prepare_contexts_preserves_falsy_values(monkeypatch):
    monkeypatch.setattr(ragas_eval, "RAGAS_MAX_CONTEXTS", 4)
    monkeypatch.setattr(ragas_eval, "RAGAS_MAX_CONTEXT_CHARS", 20)

    assert ragas_eval._prepare_contexts([0, False, None, "", "制度"]) == ["0", "False", "制度"]


def test_ragas_import_compatibility():
    """固定 langchain-community==0.3.30 后，RAGAS 的导入链路可用（issue #291）。

    ragas 0.4.x 在 import 阶段导入 langchain_community.chat_models.vertexai，
    而 langchain-community 0.4.x 已移除该模块。CI 不安装可选评测依赖，缺少 ragas
    时跳过；ragas 已安装但导入失败必须让用例失败，不能用 importorskip 掩盖。
    """
    if importlib.util.find_spec("ragas") is None:
        pytest.skip("未安装可选评测依赖 ragas")

    import ragas
    from langchain_community.chat_models.vertexai import ChatVertexAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    assert ragas is not None
    assert ChatVertexAI is not None
    assert SingleTurnSample is not None
    assert OpenAIEmbeddings is not None
    assert llm_factory is not None
    assert Faithfulness is not None
    assert LLMContextPrecisionWithoutReference is not None
    assert ResponseRelevancy is not None


def test_disabled_path_does_not_import_ragas(monkeypatch):
    """RAGAS_ENABLED=false 时只写状态与轨迹事件，完全不加载 ragas。"""
    calls = []

    monkeypatch.setattr(ragas_eval, "RAGAS_ENABLED", False)
    monkeypatch.setattr(ragas_eval, "_mark_message", lambda *args: calls.append(("mark", args)))
    monkeypatch.setattr(
        ragas_eval,
        "append_trace_event",
        lambda *args, **kwargs: calls.append(("trace", args, kwargs)),
    )
    # 若禁用路径触发任何 ragas 导入，这里会直接抛 ImportError。
    monkeypatch.setitem(sys.modules, "ragas", None)

    asyncio.run(ragas_eval.evaluate_message_async(7, "问题", "回答", ["上下文"], "trace-ragas-off"))

    assert ("mark", (7, "failed", {}, "RAGAS 已关闭")) in calls
    trace_calls = [entry for entry in calls if entry[0] == "trace"]
    assert len(trace_calls) == 1
    _, args, kwargs = trace_calls[0]
    assert args == ("trace-ragas-off", "ragas_disabled", "evaluate_message_async")
    assert kwargs["result"] == {"message_id": 7}


def _install_fake_ragas_stack(monkeypatch):
    """离线替身：拦截 openai/ragas 导入，记录客户端参数与指标调用。"""
    openai_calls = []
    metric_calls = []
    trace_events = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            openai_calls.append(kwargs)

    class FakeSingleTurnSample:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMetric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def single_turn_score(self, sample):
            metric_calls.append(sample)
            return 0.5

    def fake_llm_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    ragas_package = ModuleType("ragas")
    dataset_schema = ModuleType("ragas.dataset_schema")
    dataset_schema.SingleTurnSample = FakeSingleTurnSample
    embeddings = ModuleType("ragas.embeddings")
    embeddings.OpenAIEmbeddings = FakeOpenAIEmbeddings
    llms = ModuleType("ragas.llms")
    llms.llm_factory = fake_llm_factory
    metrics = ModuleType("ragas.metrics")
    metrics.Faithfulness = FakeMetric
    metrics.LLMContextPrecisionWithoutReference = FakeMetric
    metrics.ResponseRelevancy = FakeMetric

    for name, module in {
        "openai": fake_openai,
        "ragas": ragas_package,
        "ragas.dataset_schema": dataset_schema,
        "ragas.embeddings": embeddings,
        "ragas.llms": llms,
        "ragas.metrics": metrics,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        ragas_eval,
        "append_trace_event",
        lambda *args, **kwargs: trace_events.append((args, kwargs)),
    )
    return openai_calls, metric_calls, trace_events


def test_ragas_llm_client_config_uses_ragas_key_and_url(monkeypatch):
    monkeypatch.setattr(ragas_eval, "RAGAS_API_KEY", "ragas-only-key")
    monkeypatch.setattr(ragas_eval, "RAGAS_BASE_URL", "https://ragas.example.com/")

    assert ragas_eval._ragas_llm_client_config() == {
        "api_key": "ragas-only-key",
        "base_url": "https://ragas.example.com/v1",
    }


def test_evaluate_message_sync_constructs_llm_client_from_ragas_config(monkeypatch):
    """真实 `_evaluate_message_sync` 路径必须用 RAGAS_API_KEY/RAGAS_BASE_URL。"""
    openai_calls, metric_calls, trace_events = _install_fake_ragas_stack(monkeypatch)
    monkeypatch.setattr(ragas_eval, "RAGAS_API_KEY", "ragas-stage-key")
    monkeypatch.setattr(ragas_eval, "RAGAS_BASE_URL", "https://ragas.example.com")
    monkeypatch.setattr(ragas_eval, "RAGAS_LLM_MODEL", "deepseekv4flash")
    monkeypatch.setattr(ragas_eval, "EMBEDDING_BASE_URL", "https://embedding.example.com")
    monkeypatch.setattr(ragas_eval, "EMBEDDING_API_KEY", "embedding-key")

    scores = ragas_eval._evaluate_message_sync("问题", "回答", ["上下文"], "trace-ragas-path")

    assert openai_calls[1] == {
        "api_key": "ragas-stage-key",
        "base_url": "https://ragas.example.com/v1",
    }
    assert openai_calls[0]["api_key"] == "embedding-key"
    assert openai_calls[0]["base_url"] == "https://embedding.example.com/v1"
    assert len(metric_calls) == 3
    assert scores == {
        "faithfulness": 0.5,
        "context_precision_without_reference": 0.5,
        "response_relevancy": 0.5,
    }
    assert [entry[0][1] for entry in trace_events] == ["ragas_metric_done"] * 3


def test_evaluate_message_sync_rejects_missing_ragas_key(monkeypatch):
    monkeypatch.setattr(ragas_eval, "RAGAS_API_KEY", "")
    monkeypatch.setattr(ragas_eval, "EMBEDDING_BASE_URL", "https://embedding.example.com")
    monkeypatch.setattr(ragas_eval, "EMBEDDING_API_KEY", "embedding-key")
    # 守卫必须在导入 ragas/openai 之前触发，避免未配置时仍加载可选依赖。
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(RuntimeError, match="RAGAS API Key 未配置"):
        ragas_eval._evaluate_message_sync("问题", "回答", ["上下文"])
