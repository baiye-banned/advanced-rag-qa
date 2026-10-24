import asyncio
import importlib.util
import sys

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
