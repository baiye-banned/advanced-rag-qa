import asyncio

from rag import rerank


def _chunks():
    return [
        {"id": "a", "content": "员工迟到一次罚款50元", "file_name": "制度.txt", "file_id": 1},
        {"id": "b", "content": "迟到超过30分钟视为旷工", "file_name": "制度.txt", "file_id": 1},
    ]


def test_qwen3_rerank_success_reorders_chunks(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.92},
                        {"index": 0, "relevance_score": 0.51},
                    ]
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(rerank, "RERANK_API_KEY", "test-key")
    monkeypatch.setattr(rerank.httpx, "AsyncClient", FakeClient)

    reranked, trace = asyncio.run(rerank._rerank_chunks_with_qwen3("迟到三个小时怎么算", _chunks()))

    assert trace["status"] == "done"
    assert trace["provider"] == "dashscope"
    assert reranked[0]["id"] == "b"
    assert reranked[0]["rerank_score"] == 0.92
    assert reranked[0]["rerank_reason"] == "qwen3-rerank relevance score"


def test_rerank_falls_back_to_llm_when_qwen3_fails(monkeypatch):
    async def fake_qwen3(question, chunks):
        raise RuntimeError("qwen unavailable")

    async def fake_llm(question, chunks):
        return [{**chunks[0], "rerank_score": 0.8}], {"status": "done", "provider": "deepseek", "items": []}

    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_qwen3", fake_qwen3)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_llm", fake_llm)

    reranked, trace = asyncio.run(rerank.rerank_chunks("迟到三个小时怎么算", _chunks()))

    assert reranked[0]["id"] == "a"
    assert trace["provider"] == "deepseek_fallback"
    assert trace["fallback_from"] == "qwen3-rerank"
    assert trace["fallback_reason"] == rerank.RERANK_FALLBACK_MESSAGE


def test_rerank_returns_empty_when_qwen3_and_llm_fail(monkeypatch):
    async def fake_qwen3(question, chunks):
        raise RuntimeError("qwen unavailable")

    async def fake_llm(question, chunks):
        return [], {"status": "failed", "provider": "deepseek", "error": "llm unavailable", "items": []}

    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_qwen3", fake_qwen3)
    monkeypatch.setattr(rerank, "_rerank_chunks_with_llm", fake_llm)

    reranked, trace = asyncio.run(rerank.rerank_chunks("迟到三个小时怎么算", _chunks()))

    assert reranked == []
    assert trace["status"] == "failed"
    assert trace["provider"] == "deepseek"
    assert trace["fallback_from"] == "qwen3-rerank"


def test_llm_rerank_ignores_invalid_scores(monkeypatch):
    captured = {}

    async def fake_call_chat_json(*args, **kwargs):
        captured.update(kwargs)
        return {"results": [{"id": 1, "score": "bad-score", "reason": "invalid score"}]}

    monkeypatch.setattr(rerank, "call_chat_json", fake_call_chat_json)

    reranked, trace = asyncio.run(rerank._rerank_chunks_with_llm("question", _chunks()))

    assert trace["status"] == "done"
    assert captured["stage"] == "rerank_llm"
    assert reranked[0]["rerank_score"] == 0.0


def test_llm_rerank_skips_rows_with_invalid_ids(monkeypatch):
    async def fake_call_chat_json(*args, **kwargs):
        return {
            "results": [
                {"id": "bad-id", "score": 0.99, "reason": "invalid id"},
                {"id": 2, "score": 0.75, "reason": "valid id"},
            ]
        }

    monkeypatch.setattr(rerank, "call_chat_json", fake_call_chat_json)

    reranked, trace = asyncio.run(rerank._rerank_chunks_with_llm("question", _chunks()))

    assert trace["status"] == "done"
    assert [chunk["id"] for chunk in reranked] == ["b"]
    assert reranked[0]["rerank_score"] == 0.75


def test_llm_rerank_preserves_falsy_candidate_content(monkeypatch):
    captured = {}

    async def fake_call_chat_json(system_prompt, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return {"results": [{"id": 1, "score": 0.5, "reason": "kept"}]}

    monkeypatch.setattr(rerank, "call_chat_json", fake_call_chat_json)

    reranked, trace = asyncio.run(
        rerank._rerank_chunks_with_llm(
            "question",
            [
                {"id": "zero", "content": 0, "file_name": "zero.txt"},
                {"id": "false", "content": False, "file_name": "false.txt"},
            ],
        )
    )

    assert trace["status"] == "done"
    assert reranked[0]["id"] == "zero"
    assert '"content": "0"' in captured["user_prompt"]
    assert '"content": "False"' in captured["user_prompt"]


def test_select_final_chunks_ignores_invalid_keyword_score():
    ranked = [{"file_id": 1, "chunk_id": "a", "content": "ranked"}]
    keyword = [{"file_id": 2, "chunk_id": "b", "content": "keyword", "keyword_score": "bad-score"}]

    selected = rerank.select_final_chunks(ranked, keyword)

    assert selected == ranked


def test_select_final_chunks_skips_non_dict_keyword_candidates():
    ranked = [{"file_id": 1, "chunk_id": "a", "content": "ranked"}]

    selected = rerank.select_final_chunks(ranked, [None, {"file_id": 2, "chunk_id": "b", "content": "keyword", "keyword_score": 12}])

    assert selected[0]["chunk_id"] == "b"


def test_select_final_chunks_keeps_high_keyword_score_boost():
    ranked = [{"file_id": 1, "chunk_id": "a", "content": "ranked"}]
    keyword = [{"file_id": 2, "chunk_id": "b", "content": "keyword", "keyword_score": "12"}]

    selected = rerank.select_final_chunks(ranked, keyword)

    assert selected[0] == keyword[0]


def test_select_final_chunks_refuses_keyword_candidate_that_missed_the_query():
    """issue #56：绝对分值再高，没命中本次查询关键词的候选也不得插到首位。"""
    ranked = [
        {"file_id": 1, "chunk_id": f"r{index}", "content": f"报销流程第{index}条"}
        for index in range(5)
    ]
    off_topic = {
        "file_id": 2,
        "chunk_id": "b",
        "content": "迟到30分钟以内罚款50元",
        "keyword_score": 30.0,
        "keyword_hits": 0,
    }

    selected = rerank.select_final_chunks(ranked, [off_topic])

    assert selected == ranked
    assert off_topic not in selected


def test_select_final_chunks_injects_keyword_candidate_that_hit_the_query():
    ranked = [
        {"file_id": 1, "chunk_id": f"r{index}", "content": f"报销流程第{index}条"}
        for index in range(5)
    ]
    on_topic = {
        "file_id": 2,
        "chunk_id": "b",
        "content": "报销申请需提交原始发票",
        "keyword_score": 15.5,
        "keyword_hits": 2,
    }

    selected = rerank.select_final_chunks(ranked, [on_topic])

    assert selected[0] == on_topic
    assert len(selected) == rerank.RETRIEVAL_RERANK_TOP_N


def test_select_final_chunks_ignores_unparsable_keyword_hit_count():
    """命中数字段不可解析时按「未命中」处理：拒绝注入而不是回退到只看分值。"""
    ranked = [{"file_id": 1, "chunk_id": "a", "content": "ranked"}]
    keyword = [
        {"file_id": 2, "chunk_id": "b", "content": "keyword", "keyword_score": 12, "keyword_hits": "bad"}
    ]

    selected = rerank.select_final_chunks(ranked, keyword)

    assert selected == ranked
