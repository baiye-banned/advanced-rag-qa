"""Retrieval acceptance goal (docs/MAINTENANCE_GOAL_CLOSURE.md, goal 3).

tests/test_retrieval.py and tests/test_rerank.py cover the individual helpers. This
file drives the whole chain of ``retrieve_knowledge`` -- multi-route recall, RRF fusion,
rerank and final context selection -- with the real fusion/truncation code and only the
provider boundary stubbed:

* vector recall (``query_vectors``) is replaced, because it is asserted separately against
  a real Milvus Lite store in test_milvus_acceptance.py;
* keyword recall (``keyword_recall``) is replaced only where a case isolates a different
  route. It reads the relational store, not Milvus, so its real behavior is covered at the
  end of this file by driving the unpatched implementation against a real SQLAlchemy
  database holding real ``KnowledgeFile`` rows;
* the rerank HTTP call is replaced by an in-process transport, while the real
  ``rerank_chunks`` / ``select_final_chunks`` logic under test stays untouched;
* in the query-planning cases below, only the planner's model call (``call_chat_json``) is
  replaced, so ``build_query_plan`` normalization, the deterministic keyword fallback and
  the keyword terms handed to the keyword route all run for real.

No network access is required.
"""

import asyncio
import logging

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config import RETRIEVAL_RERANK_TOP_N, RETRIEVAL_ROUTE_TOP_K
from database.session import Base
from model.models import KnowledgeFile
from rag import rerank, retrieval


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    """The model declares MySQL LONGTEXT; SQLite needs it spelled as TEXT."""
    return "TEXT"

ROUTE_LABELS = [
    "planned",
    "simplified",
    "sub_question_1",
    "sub_question_2",
    "hyde",
    "rewrite_1",
    "rewrite_2",
]


def _chunk(key, content):
    return {
        "id": key,
        "chunk_id": key,
        "content": content,
        "file_name": "考勤制度.txt",
        "file_id": 1,
        "route": "vector",
    }


C1 = _chunk("c1", "迟到一次罚款50元")
C2 = _chunk("c2", "迟到超过30分钟视为旷工半天")
C3 = _chunk("c3", "当月迟到三次以上加倍处罚")
C4 = _chunk("c4", "迟到三个小时按旷工一天处理")
C5 = _chunk("c5", "请假需要提前一天申请")

QUERY_PLAN = {
    "original_question": "迟到三个小时怎么处理",
    "simplified_question": "考勤 迟到 处理",
    "sub_questions": ["迟到超过30分钟如何认定", "迟到三小时是否算旷工"],
    "hyde_document": "员工迟到超过30分钟按旷工处理，并处以罚款。",
    "rewrites": ["迟到扣多少钱", "迟到处罚标准"],
    "keywords": ["迟到", "旷工"],
    "required_evidence": ["迟到认定规则"],
}

# Fusion scores (k=60, rank counts from 1) are all distinct, so the expected fused
# order c4 > c5 > c3 > c1 > c2 is deterministic rather than a float coincidence.
ROUTE_RESULTS = {
    "planned": [C1, C2, C3],
    "simplified": [C3],
    "sub_question_1": [C4],
    "sub_question_2": [C4],
    "hyde": [C5, C4],
    "rewrite_1": [C5],
    "rewrite_2": [],
}
FUSED_ORDER = ["c4", "c5", "c3", "c1", "c2"]

# Deliberately different from the fused order: the final context must follow the rerank
# scores, so a chain that silently skipped or ignored rerank would fail these tests.
RERANK_SCORES = {
    C1["content"]: 0.4,
    C2["content"]: 0.95,
    C3["content"]: 0.35,
    C4["content"]: 0.2,
    C5["content"]: 0.1,
}
RERANKED_ORDER = ["c2", "c1", "c3", "c4", "c5"]


class _Recorder:
    def __init__(self):
        self.events = []

    async def add(self, event, *args, **payload):
        self.events.append({"event": event, "args": args, **payload})


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _install_rerank_transport(monkeypatch, handler):
    """Route the rerank HTTP call to ``handler(request)``; returns captured requests."""
    captured = []

    class _Client:
        def __init__(self, *args, **kwargs):
            captured.append({"args": args, "timeout": kwargs.get("timeout")})

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            request = captured[-1]
            request.update({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
            return _FakeResponse(handler(request))

    monkeypatch.setattr(rerank, "RERANK_API_KEY", "test-rerank-key")
    monkeypatch.setattr(rerank.httpx, "AsyncClient", _Client)
    return captured


def _score_by_content(scores):
    def handler(request):
        documents = request["json"]["documents"]
        return {
            "output": {
                "results": [
                    {"index": index, "relevance_score": scores[document]}
                    for index, document in enumerate(documents)
                ]
            }
        }

    return handler


def _raising_handler(exc):
    def handler(_request):
        raise exc

    return handler


def _install_recall(monkeypatch, route_results, keyword_chunks=()):
    """Replace marginal recall; records every recall call in order."""
    calls = []

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        calls.append({"route": route, "query": query, "top_k": top_k, "knowledge_base_id": knowledge_base_id})
        return [dict(chunk) for chunk in route_results.get(route, [])]

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        calls.append(
            {"route": "keyword", "query": list(keywords), "top_k": top_k, "knowledge_base_id": knowledge_base_id}
        )
        return [dict(chunk) for chunk in keyword_chunks]

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    return calls


def _retrieve(knowledge_base_id=7, trace_recorder=None, question="迟到三个小时怎么处理", plan=None, db=None):
    return asyncio.run(
        retrieval.retrieve_knowledge(
            question,
            knowledge_base_id=knowledge_base_id,
            db=object() if db is None else db,
            trace_recorder=trace_recorder,
            query_plan=dict(QUERY_PLAN if plan is None else plan),
        )
    )


def test_acceptance_multi_route_rrf_rerank_final_order(monkeypatch):
    calls = _install_recall(monkeypatch, ROUTE_RESULTS)
    captured = _install_rerank_transport(monkeypatch, _score_by_content(RERANK_SCORES))

    final, trace = _retrieve()

    assert [call["route"] for call in calls] == [*ROUTE_LABELS, "keyword"]
    assert all(call["top_k"] == RETRIEVAL_ROUTE_TOP_K for call in calls)
    assert all(call["knowledge_base_id"] == 7 for call in calls)

    assert [item["chunk_id"] for item in trace["rrf"]] == FUSED_ORDER
    assert trace["rrf"][0]["rrf_score"] == pytest.approx(2 / 61 + 1 / 62)
    assert trace["rrf"][0]["routes"] == [
        {"route": "sub_question_1", "rank": 1},
        {"route": "sub_question_2", "rank": 1},
        {"route": "hyde", "rank": 2},
    ]
    assert {item["route"]: item["count"] for item in trace["routes"]} == {
        "planned": 3,
        "simplified": 1,
        "sub_question_1": 1,
        "sub_question_2": 1,
        "hyde": 2,
        "rewrite_1": 1,
        "rewrite_2": 0,
        "keyword": 0,
    }

    # The provider received the fused candidates in fused order, once, with the plan's
    # original question and the configured top_n.
    assert len(captured) == 1
    request = captured[0]
    assert request["url"] == rerank.RERANK_BASE_URL
    assert request["timeout"] == rerank.RERANK_TIMEOUT_SECONDS
    assert request["headers"]["Authorization"] == "Bearer test-rerank-key"
    assert request["json"]["model"] == rerank.RERANK_MODEL
    assert request["json"]["query"] == QUERY_PLAN["original_question"]
    assert request["json"]["documents"] == [item["excerpt"] for item in trace["rrf"]]
    assert request["json"]["top_n"] == min(RETRIEVAL_RERANK_TOP_N, len(FUSED_ORDER))

    # Final context follows the rerank scores, not the fused order.
    assert [item["chunk_id"] for item in final] == RERANKED_ORDER
    assert final[0]["rerank_score"] == pytest.approx(0.95)
    assert final[0]["rerank_reason"] == "qwen3-rerank relevance score"
    assert trace["rerank"]["status"] == "done"
    assert trace["rerank"]["provider"] == rerank.RERANK_PROVIDER
    assert trace["rerank"]["model"] == rerank.RERANK_MODEL


def test_acceptance_rerank_failure_falls_back_to_fused_order(monkeypatch, caplog):
    _install_recall(monkeypatch, ROUTE_RESULTS)
    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", False)
    captured = _install_rerank_transport(monkeypatch, _raising_handler(httpx.ConnectError("connection refused")))

    with caplog.at_level(logging.WARNING, logger=rerank.__name__):
        final, trace = _retrieve()

    assert [item["chunk_id"] for item in final] == FUSED_ORDER
    assert all("rerank_score" not in item for item in final)
    assert len(captured) == 1
    assert trace["rerank"]["status"] == "failed"
    assert trace["rerank"]["provider"] == rerank.RERANK_PROVIDER
    # issue #263: the trace that is persisted with the message carries only the fixed
    # message, not the provider's raw exception text.
    assert trace["rerank"]["error"] == rerank.RERANK_FAILED_MESSAGE
    assert "connection refused" not in trace["rerank"]["error"]
    # The raw failure remains diagnosable from the server log.
    assert "connection refused" in caplog.text
    assert "scope=rerank_primary" in caplog.text


def test_acceptance_rerank_falls_back_to_llm_with_real_fused_candidates(monkeypatch, caplog):
    _install_recall(monkeypatch, ROUTE_RESULTS)
    prompts = {}

    async def fake_call_chat_json(system_prompt, user_prompt, **kwargs):
        prompts["system"] = system_prompt
        prompts["user"] = user_prompt
        return {
            "results": [
                {"id": 1, "score": 0.9, "reason": "直接命中迟到处理规则"},
                {"id": 2, "score": 0.2, "reason": "仅部分相关"},
            ]
        }

    monkeypatch.setattr(rerank, "RERANK_LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(rerank, "call_chat_json", fake_call_chat_json)
    _install_rerank_transport(monkeypatch, _raising_handler(httpx.ConnectError("qwen3 unreachable")))

    with caplog.at_level(logging.WARNING, logger=rerank.__name__):
        final, trace = _retrieve()

    # Explicit candidate ids are 1-based over the fused list.
    assert [item["chunk_id"] for item in final] == ["c4", "c5"]
    assert final[0]["rerank_score"] == pytest.approx(0.9)
    assert trace["rerank"]["provider"] == "deepseek_fallback"
    assert trace["rerank"]["fallback_from"] == rerank.RERANK_MODEL
    # issue #263: the fallback reason is a fixed message; the primary failure's raw text
    # stays in the server log.
    assert trace["rerank"]["fallback_reason"] == rerank.RERANK_FALLBACK_MESSAGE
    assert "qwen3 unreachable" not in trace["rerank"]["fallback_reason"]
    assert "qwen3 unreachable" in caplog.text
    assert "scope=rerank_fallback_success" in caplog.text
    for key in FUSED_ORDER:
        assert next(chunk["content"] for chunk in ROUTE_RESULTS["planned"] + ROUTE_RESULTS["hyde"] if chunk["chunk_id"] == key) in prompts["user"]


def test_acceptance_keyword_boost_prepends_best_keyword_chunk(monkeypatch):
    keyword_chunk = {**_chunk("k1", "员工迟到每月累计三次以上，公司可以解除劳动合同"), "keyword_score": 12.0}
    _install_recall(monkeypatch, ROUTE_RESULTS, keyword_chunks=[keyword_chunk])
    _install_rerank_transport(
        monkeypatch, _score_by_content({**RERANK_SCORES, keyword_chunk["content"]: 0.05})
    )

    final, trace = _retrieve()

    assert final[0]["chunk_id"] == "k1"
    assert final[0]["keyword_score"] == pytest.approx(12.0)
    assert [item["chunk_id"] for item in final[1:]] == RERANKED_ORDER[: RETRIEVAL_RERANK_TOP_N - 1]
    assert len(final) == RETRIEVAL_RERANK_TOP_N
    assert trace["routes"][-1]["route"] == "keyword"
    assert trace["routes"][-1]["count"] == 1


def test_acceptance_keyword_boost_does_not_duplicate_selected_chunk(monkeypatch):
    duplicate = {**C2, "keyword_score": 12.0, "route": "keyword"}
    _install_recall(monkeypatch, ROUTE_RESULTS, keyword_chunks=[duplicate])
    _install_rerank_transport(monkeypatch, _score_by_content(RERANK_SCORES))

    final, _ = _retrieve()

    assert [item["chunk_id"] for item in final] == RERANKED_ORDER
    assert len(final) == RETRIEVAL_RERANK_TOP_N
    keys = [rerank.chunk_key(item) for item in final]
    assert len(keys) == len(set(keys))


def test_acceptance_empty_recall_skips_rerank_without_provider_call(monkeypatch):
    calls = _install_recall(monkeypatch, {})
    captured = _install_rerank_transport(monkeypatch, _score_by_content({}))

    final, trace = _retrieve()

    assert final == []
    assert trace["rrf"] == []
    assert trace["rerank"] == {"status": "skipped", "items": []}
    assert [call["route"] for call in calls] == [*ROUTE_LABELS, "keyword"]
    assert [item["count"] for item in trace["routes"]] == [0] * len(ROUTE_LABELS + ["keyword"])
    assert captured == []


def test_acceptance_partial_empty_routes_still_produce_context(monkeypatch):
    _install_recall(monkeypatch, {"simplified": [C1]})
    captured = _install_rerank_transport(monkeypatch, _score_by_content({C1["content"]: 0.7}))

    final, trace = _retrieve()

    assert {item["route"]: item["count"] for item in trace["routes"]}["planned"] == 0
    assert [item["chunk_id"] for item in trace["rrf"]] == ["c1"]
    assert [item["chunk_id"] for item in final] == ["c1"]
    assert captured[0]["json"]["documents"] == [C1["content"]]
    assert trace["rerank"]["status"] == "done"


def test_acceptance_single_route_failure_is_not_swallowed(monkeypatch):
    recorder = _Recorder()

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        if route == "hyde":
            raise RuntimeError("milvus search failed")
        return [dict(chunk) for chunk in ROUTE_RESULTS.get(route, [])]

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="milvus search failed"):
        _retrieve(trace_recorder=recorder)

    # The failure aborts the retrieval instead of quietly returning a partial context.
    assert [event["event"] for event in recorder.events] == ["retriever_started"]


def test_acceptance_route_plan_is_deduplicated_and_capped(monkeypatch):
    plan = {
        "original_question": "迟到处理",
        "simplified_question": "迟到 处理",
        "sub_questions": [
            "迟到 处理",
            "迟到超过30分钟如何认定",
            "迟到三小时是否算旷工",
            "多余的子问题应被截断",
        ],
        "hyde_document": "迟到处理的相关制度说明",
        "rewrites": ["迟到扣多少钱", "迟到怎么处罚", "迟到申诉流程", "多余的改写应被截断"],
        "keywords": ["迟到"],
    }
    calls = _install_recall(monkeypatch, {})
    _install_rerank_transport(monkeypatch, _score_by_content({}))

    _retrieve(plan=plan)

    assert [call["route"] for call in calls] == [
        "planned",
        "simplified",
        "sub_question_2",
        "sub_question_3",
        "hyde",
        "rewrite_1",
        "rewrite_2",
        "rewrite_3",
        "keyword",
    ]


def test_acceptance_rerank_candidate_window_is_capped(monkeypatch):
    many = [_chunk(f"m{index}", f"候选段落{index}：员工迟到处理说明") for index in range(14)]
    _install_recall(monkeypatch, {"planned": many})
    captured = _install_rerank_transport(
        monkeypatch, _score_by_content({chunk["content"]: 0.5 for chunk in many})
    )

    final, trace = _retrieve()

    documents = captured[0]["json"]["documents"]
    assert len(documents) == 12
    assert documents == [chunk["content"] for chunk in many[:12]]
    assert captured[0]["json"]["top_n"] == RETRIEVAL_RERANK_TOP_N
    assert [item["chunk_id"] for item in trace["rrf"]] == [f"m{index}" for index in range(10)]
    assert [item["chunk_id"] for item in final] == [f"m{index}" for index in range(RETRIEVAL_RERANK_TOP_N)]


# ---------------------------------------------------------------------------
# Query planning: plan production, and the keyword terms it feeds the chain
#
# The cases above hand ``retrieve_knowledge`` a ready-made plan, so the planner itself is
# never invoked there. These cases drive ``build_query_plan`` against the planner's model
# boundary (``call_chat_json``) only, and then check that the keyword terms derived from a
# plan are the ones the keyword route actually searches for.
# ---------------------------------------------------------------------------


def test_acceptance_query_plan_normalizes_and_caps_the_planner_output(monkeypatch):
    seen = {}

    async def fake_call_chat_json(system_prompt, user_prompt, **kwargs):
        seen["user"] = user_prompt
        return {
            "hyde_document": "  员工迟到超过30分钟按旷工处理。  ",
            "rewrites": ["迟到扣多少钱", "   ", "迟到处罚标准", "迟到申诉流程", "多余的改写应被截断"],
            "keywords": ["迟到", "   ", "旷工", "迟到"],
        }

    monkeypatch.setattr(retrieval, "call_chat_json", fake_call_chat_json)

    plan = asyncio.run(retrieval.build_query_plan("迟到扣钱"))

    assert "迟到扣钱" in seen["user"]
    assert plan["error"] == ""
    # The hypothesis document is stripped, and the rewrite list is cleaned (blank entries
    # dropped) before the three-rewrite cap applies.
    assert plan["hyde_document"] == "员工迟到超过30分钟按旷工处理。"
    assert plan["rewrites"] == ["迟到扣多少钱", "迟到处罚标准", "迟到申诉流程"]
    # Plan keywords keep their order and lose the blanks and duplicates, then the
    # deterministic keywords of the question are merged in behind them.
    assert plan["keywords"] == ["迟到", "旷工", "迟到扣钱", "到扣", "扣钱"]

    # The merged keyword list is capped, and the cap keeps the earliest terms.
    async def many_keywords(*_args, **_kwargs):
        return {"keywords": [f"关键词{index}" for index in range(30)]}

    monkeypatch.setattr(retrieval, "call_chat_json", many_keywords)

    capped = asyncio.run(retrieval.build_query_plan("迟到扣钱"))

    assert capped["keywords"] == [f"关键词{index}" for index in range(24)]


def test_acceptance_query_plan_failure_still_yields_usable_keywords(monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("planner unreachable")

    monkeypatch.setattr(retrieval, "call_chat_json", boom)

    plan = asyncio.run(retrieval.build_query_plan("迟到扣钱"))

    assert plan["hyde_document"] == ""
    assert plan["rewrites"] == []
    assert plan["error"]
    # A failed plan must not take the keyword route down with it: the deterministic
    # extractor still produces query terms for the question.
    assert plan["keywords"] == ["迟到", "迟到扣钱", "到扣", "扣钱"]


def test_acceptance_fallback_keywords_extract_numeric_phrases_and_terms():
    keywords = retrieval._fallback_keywords("迟到30分钟以内罚款50元怎么处理")

    # Numeric policy phrases are matched as whole phrases, ahead of the single terms.
    assert keywords[:2] == ["30分钟以内", "50元"]
    assert "迟到" in keywords
    assert "罚款" in keywords


def test_acceptance_fallback_keywords_expand_short_chinese_tokens():
    keywords = retrieval._fallback_keywords("迟到扣钱")

    # A whole-token hit alone would miss the sub-phrases a user actually types, so short
    # Chinese tokens also contribute their adjacent bigrams.
    assert keywords == ["迟到", "迟到扣钱", "到扣", "扣钱"]


def test_acceptance_plan_keywords_reach_the_keyword_route(monkeypatch):
    calls = _install_recall(monkeypatch, {})
    _install_rerank_transport(monkeypatch, _score_by_content({}))
    plan = {
        "original_question": "迟到扣钱",
        "simplified_question": "迟到扣钱",
        "keywords": ["旷工"],
        "required_evidence": ["罚款"],
    }

    _retrieve(question="迟到扣钱", plan=plan)

    keyword_call = next(call for call in calls if call["route"] == "keyword")
    # The plan's keywords come first, then the deterministic keywords of the raw question
    # (merged in while the plan is normalized), then the plan's required evidence.
    # Duplicates collapse in place, which is why the question's terms are not repeated
    # when the chain merges them a second time.
    assert keyword_call["query"] == ["旷工", "迟到", "迟到扣钱", "到扣", "扣钱", "罚款"]
    assert keyword_call["knowledge_base_id"] == 7
    assert keyword_call["top_k"] == RETRIEVAL_ROUTE_TOP_K


# ---------------------------------------------------------------------------
# Real keyword recall (unpatched implementation, real relational store)
#
# ``keyword_recall`` reads the relational store, not Milvus, so a Milvus Lite fixture
# cannot cover it. These cases build the real ``KnowledgeFile`` table in a real SQLite
# database, insert real rows, and call the implementation itself -- no stub session and
# no patched ``keyword_recall``. Only the round trip through the model/session layer is
# shared with the other database-backed tests in this suite.
# ---------------------------------------------------------------------------

KB = 7
OTHER_KB = 8

ATTENDANCE = "员工迟到30分钟以内罚款50元；迟到超过30分钟按旷工处理。"
ASSESSMENT = "考勤制度：上下班迟到早退均计入月度考核。"
EXPENSE = "报销流程：发票需要在30天内提交。"
# Longer than the 900-character window, so the splitter must emit a second chunk. The
# filler has no keyword, and the tail starts exactly at the 720-character step (900 - 180).
LONG_CHUNK_STEP = 900 - 180
LONG_HEAD = "迟到超过30分钟按旷工处理。"
LONG_TAIL = "迟到罚款50元。"
LONG_FILE = LONG_HEAD + "附" * (LONG_CHUNK_STEP - len(LONG_HEAD)) + LONG_TAIL


def _real_knowledge_db(tmp_path):
    """Real SQLite database holding real ``KnowledgeFile`` rows."""
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'keyword-recall.db').as_posix()}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[KnowledgeFile.__table__])
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()
    db.add_all(
        [
            KnowledgeFile(knowledge_base_id=KB, name="考勤制度.txt", size=len(ATTENDANCE), content=ATTENDANCE),
            KnowledgeFile(knowledge_base_id=KB, name="考勤考核.txt", size=len(ASSESSMENT), content=ASSESSMENT),
            KnowledgeFile(knowledge_base_id=KB, name="报销制度.txt", size=len(EXPENSE), content=EXPENSE),
            # Same text as 考勤制度.txt, stored under another knowledge base: it must never leak.
            KnowledgeFile(knowledge_base_id=OTHER_KB, name="他库考勤.txt", size=len(ATTENDANCE), content=ATTENDANCE),
        ]
    )
    db.commit()
    return db


def test_acceptance_real_keyword_recall_reads_the_relational_store(tmp_path):
    db = _real_knowledge_db(tmp_path)

    chunks = retrieval.keyword_recall(db, KB, ["迟到", "旷工"], RETRIEVAL_ROUTE_TOP_K)

    # Only rows of the requested knowledge base, and only rows whose text matches a keyword.
    assert [chunk["file_name"] for chunk in chunks] == ["考勤制度.txt", "考勤考核.txt"]
    assert all(chunk["route"] == "keyword" for chunk in chunks)
    assert all("他库考勤.txt" != chunk["file_name"] for chunk in chunks)

    # The chunk carries the stored row's identity and its real offset in the stored text.
    assert chunks[0]["file_id"] > 0
    assert chunks[0]["chunk_id"] == "0"
    assert chunks[0]["id"] == f"{chunks[0]['file_id']}_0"
    assert chunks[0]["content"] == ATTENDANCE

    # Scoring and ordering are the implementation's own: the stronger match comes first.
    assert chunks[0]["keyword_score"] > chunks[1]["keyword_score"]
    # ... and it is strong enough to clear the final-selection boost threshold.
    assert chunks[0]["keyword_score"] >= 10


def test_acceptance_real_keyword_recall_chunks_long_text_and_caps_results(tmp_path):
    db = _real_knowledge_db(tmp_path)
    db.add(KnowledgeFile(knowledge_base_id=KB, name="长文件.txt", size=len(LONG_FILE), content=LONG_FILE))
    db.commit()

    chunks = retrieval.keyword_recall(db, KB, ["迟到", "旷工"], RETRIEVAL_ROUTE_TOP_K)

    long_chunks = [chunk for chunk in chunks if chunk["file_name"] == "长文件.txt"]
    # Chunk ids are the real character offsets produced by the splitter, and the second
    # chunk really starts at that offset in the stored text.
    assert [chunk["chunk_id"] for chunk in long_chunks] == ["0", str(LONG_CHUNK_STEP)]
    assert LONG_FILE[LONG_CHUNK_STEP:] == LONG_TAIL
    assert long_chunks[1]["content"] == LONG_TAIL
    # Chunks with no keyword hit are dropped, not returned with a zero score.
    assert all(chunk["keyword_score"] > 0 for chunk in chunks)

    # ``top_k`` caps the real result list, and blank keywords never reach the store.
    assert len(retrieval.keyword_recall(db, KB, ["迟到", "旷工"], 1)) == 1
    assert retrieval.keyword_recall(db, KB, [], RETRIEVAL_ROUTE_TOP_K) == []
    assert retrieval.keyword_recall(db, KB, ["   "], RETRIEVAL_ROUTE_TOP_K) == []
    assert retrieval.keyword_recall(db, OTHER_KB, ["旷工"], RETRIEVAL_ROUTE_TOP_K) != []


def test_acceptance_retrieve_knowledge_uses_the_real_keyword_recall(tmp_path, monkeypatch):
    """One path from plan to final context with the unpatched keyword recall.

    Only the vector routes and the rerank transport are stubbed; the keyword route runs
    the real implementation against the real database, and its strongest chunk has to
    survive fusion and rerank to reach the final context.
    """
    db = _real_knowledge_db(tmp_path)
    monkeypatch.setattr(retrieval, "query_vectors", lambda *args, **kwargs: [])

    def only_assessment(request):
        documents = request["json"]["documents"]
        assert documents == [ATTENDANCE, ASSESSMENT]
        return {"output": {"results": [{"index": 1, "relevance_score": 0.8}]}}

    captured = _install_rerank_transport(monkeypatch, only_assessment)

    final, trace = _retrieve(db=db)

    keyword_route = next(item for item in trace["routes"] if item["route"] == "keyword")
    assert keyword_route["count"] == 2
    assert [item["file_name"] for item in keyword_route["items"]] == ["考勤制度.txt", "考勤考核.txt"]

    # The real database content is what the reranker receives, already in the order the
    # real keyword scores gave the fused list.
    assert captured[0]["json"]["documents"] == [ATTENDANCE, ASSESSMENT]
    assert [item["chunk_id"] for item in trace["rrf"]] == ["0", "0"]

    # The reranker only kept the second document, so the prepended first element is the
    # real keyword chunk that the keyword route produced from the database.
    assert [item["file_name"] for item in final] == ["考勤制度.txt", "考勤考核.txt"]
    assert final[0]["content"] == ATTENDANCE
    assert final[0]["keyword_score"] >= 10
    assert final[1]["rerank_score"] == pytest.approx(0.8)
