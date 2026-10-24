"""回归（issue #263）：rerank trace 不得回吐原始异常。

rerank 的 trace 会随 assistant 消息落库，并由消息历史接口反复返回给客户端。旧实现把
``str(exc)`` 直接写进 ``error`` / ``fallback_reason``，而网络库异常可能带完整上游 URL、
查询参数与配置状态。本文件用带唯一哨兵的异常覆盖三条失败分支，断言 trace 里只剩固定
文案，同时原文仍以 ``exc_info`` 留在服务端日志里（可诊断性不降级）。
"""

import asyncio
import json
import logging

from rag import rerank


SENTINEL = "https://dashscope.example/?key=SECRET-SENTINEL"
LEAK_MARKERS = ("SECRET-SENTINEL", "dashscope.example")


def _chunks():
    return [
        {"id": "a", "content": "员工迟到一次罚款50元", "file_name": "制度.txt", "file_id": 1},
        {"id": "b", "content": "迟到超过30分钟视为旷工", "file_name": "制度.txt", "file_id": 1},
    ]


def _assert_trace_clean(trace: dict, where: str) -> None:
    serialized = json.dumps(trace, ensure_ascii=False, default=str)
    for marker in LEAK_MARKERS:
        assert marker not in serialized, f"{where} 回吐了异常原文片段 {marker!r}：{serialized}"


async def _boom(*_args, **_kwargs):
    raise RuntimeError(SENTINEL)


async def _fake_llm_success(question, chunks):
    return (
        [{**chunks[0], "rerank_score": 0.8}],
        {"status": "done", "provider": "deepseek", "items": [{"chunk_id": "a"}]},
    )


def test_primary_failure_masks_exception_in_trace(monkeypatch, caplog):
    """fallback 关闭：error 是固定文案，trace 不含哨兵，原文仍进日志。"""
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", False)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_qwen3", _boom)

    with caplog.at_level(logging.WARNING):
        reranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么罚", _chunks()))

    assert reranked == []
    assert trace == {
        "status": "failed",
        "provider": rerank.RERANK_PROVIDER,
        "model": rerank.RERANK_MODEL,
        "error": rerank.RERANK_FAILED_MESSAGE,
        "items": [],
    }
    _assert_trace_clean(trace, "rerank trace（fallback 关闭）")

    # 可诊断性不降级：原文与真实栈仍落服务端日志，并带 scope 便于检索。
    assert SENTINEL in caplog.text
    assert "RuntimeError" in caplog.text
    assert "scope=rerank_primary" in caplog.text


def test_fallback_success_masks_primary_exception(monkeypatch, caplog):
    """fallback 成功：fallback_reason 是固定文案，切换到备用的语义与字段不变。"""
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_qwen3", _boom)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_llm", _fake_llm_success)

    with caplog.at_level(logging.WARNING):
        reranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么罚", _chunks()))

    assert reranked[0]["id"] == "a"
    assert trace["status"] == "done"
    assert trace["provider"] == "deepseek_fallback"
    assert trace["fallback_from"] == rerank.RERANK_MODEL
    assert trace["fallback_reason"] == rerank.RERANK_FALLBACK_MESSAGE
    assert trace["items"] == [{"chunk_id": "a"}]
    _assert_trace_clean(trace, "rerank trace（fallback 成功）")

    assert SENTINEL in caplog.text
    assert "scope=rerank_fallback_success" in caplog.text


def test_fallback_failure_masks_both_exceptions(monkeypatch, caplog):
    """主、备都失败：fallback_reason 与 error 都是固定文案；这里走真实的备用实现，
    让 LLM fallback 自身的 ``str(exc)`` 出口（issue 点名的第 4 处）也被覆盖。"""
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_qwen3", _boom)
    monkeypatch.setattr(rerank, "call_chat_json", _boom)

    with caplog.at_level(logging.WARNING):
        reranked, trace = asyncio.run(rerank.rerank_chunks("迟到怎么罚", _chunks()))

    assert reranked == []
    assert trace["status"] == "failed"
    assert trace["provider"] == "deepseek"
    assert trace["fallback_from"] == rerank.RERANK_MODEL
    assert trace["fallback_reason"] == rerank.RERANK_ALL_FAILED_MESSAGE
    assert trace["error"] == rerank.RERANK_FALLBACK_FAILED_MESSAGE
    assert trace["items"] == []
    _assert_trace_clean(trace, "rerank trace（主备均失败）")

    # 两处异常各留一条带 scope 的日志。
    assert SENTINEL in caplog.text
    assert "scope=rerank_llm_fallback" in caplog.text
    assert "scope=rerank_fallback_failed" in caplog.text
