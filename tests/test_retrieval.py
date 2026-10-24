import asyncio
import threading
import time

import pytest
from sqlalchemy.sql import operators

from rag import retrieval


def test_retrieve_knowledge_uses_sub_questions_and_rewrites(monkeypatch):
    vector_calls = []

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        vector_calls.append((route, query))
        return [
            {
                "id": route,
                "chunk_id": route,
                "content": query,
                "file_name": "制度.txt",
                "file_id": len(vector_calls),
                "route": route,
            }
        ]

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        return []

    async def fake_rerank_chunks(question, chunks):
        return chunks, {"status": "done", "items": []}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank_chunks)

    chunks, trace = asyncio.run(
        retrieval.retrieve_knowledge(
            "考勤 迟到 旷工 处罚",
            knowledge_base_id=1,
            db=object(),
            query_plan={
                "original_question": "迟到三个小时扣多少钱",
                "simplified_question": "考勤 迟到 旷工 处罚",
                "sub_questions": ["迟到超过30分钟如何认定"],
                "rewrites": ["迟到三个小时怎么处理"],
                "keywords": ["考勤", "旷工"],
                "required_evidence": ["迟到认定规则"],
            },
        )
    )

    # 多路召回并发执行后调用发生的先后不再有保证，trace 中的顺序才是对外语义
    # （它按 route_specs 的生成顺序固定），这里同样能验证“采纳子问题/改写 + 同义 query 去重”。
    assert {route for route, _ in vector_calls} == {"planned", "sub_question_1", "rewrite_1"}
    vector_routes_in_trace = [item["route"] for item in trace["routes"] if item["route"] != "keyword"]
    assert vector_routes_in_trace == ["planned", "sub_question_1", "rewrite_1"]
    assert chunks
    assert trace["query_plan"]["original_question"] == "迟到三个小时扣多少钱"


def test_retrieve_knowledge_keeps_event_loop_responsive_and_recalls_concurrently(monkeypatch):
    """一轮多路检索期间事件循环必须仍在被调度，且各路召回是并发而不是串行等待。"""
    route_count = 8
    route_delay = 0.2
    keyword_delay = 0.2
    lock = threading.Lock()
    inflight = {"now": 0, "max": 0}
    windows = {"vectors_end": 0.0, "keyword_start": None}
    calls = []

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        with lock:
            calls.append(route)
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        try:
            time.sleep(route_delay)
            return []
        finally:
            with lock:
                inflight["now"] -= 1
                windows["vectors_end"] = max(windows["vectors_end"], time.perf_counter())

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        with lock:
            windows["keyword_start"] = time.perf_counter()
        time.sleep(keyword_delay)
        return []

    async def fake_rerank_chunks(question, chunks):
        return chunks, {"status": "done", "items": []}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank_chunks)

    async def scenario():
        heartbeats = []

        async def ticker():
            while True:
                await asyncio.sleep(0.02)
                heartbeats.append(time.perf_counter())

        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)  # 先让心跳协程真正跑起来
        started_at = time.perf_counter()
        _, trace = await retrieval.retrieve_knowledge(
            "迟到怎么处理",
            knowledge_base_id=1,
            db=object(),
            query_plan={
                "original_question": "迟到怎么处理",
                "simplified_question": "考勤 迟到 处理",
                "sub_questions": ["迟到多久算旷工", "迟到罚款多少"],
                "hyde_document": "员工迟到30分钟以内罚款50元。",
                "rewrites": ["迟到怎么罚", "上班迟到扣多少钱", "迟到处罚规定"],
                "keywords": ["迟到"],
            },
        )
        finished_at = time.perf_counter()
        ticker_task.cancel()
        during = [beat for beat in heartbeats if started_at < beat < finished_at]
        return during, finished_at - started_at, trace

    heartbeats_during_retrieval, elapsed, trace = asyncio.run(scenario())

    assert [item["route"] for item in trace["routes"]] == [
        "planned",
        "simplified",
        "sub_question_1",
        "sub_question_2",
        "hyde",
        "rewrite_1",
        "rewrite_2",
        "rewrite_3",
        "keyword",
    ]
    assert len(calls) == route_count
    # 检索期间事件循环仍在调度心跳协程：修复前这里是 0（同步调用独占事件循环）。
    assert heartbeats_during_retrieval
    # 多路召回并发：任一时刻都有多路在途，而不是一路接一路。
    assert inflight["max"] >= 2
    # 关键字召回也在同一个 gather 里：它必须在最后一路向量召回结束前就启动。
    assert windows["keyword_start"] is not None
    assert windows["keyword_start"] < windows["vectors_end"]
    # 8 路各 0.2s：串行需要 1.6s，并发后应明显低于串行耗时。
    assert elapsed < route_count * route_delay * 0.6


def test_retrieve_knowledge_skips_vector_routes_when_embedding_backend_fails(monkeypatch):
    keyword_chunk = {
        "id": "kw-1",
        "chunk_id": "1",
        "content": "迟到30分钟以内罚款50元",
        "file_name": "考勤制度.txt",
        "file_id": 1,
        "route": "keyword",
    }
    status_calls = []

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        raise retrieval.EmbeddingBackendError("向量化接口调用失败（https://embedding.example/v1/embeddings）：429 Too Many Requests")

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        return [keyword_chunk]

    async def fake_rerank_chunks(question, chunks):
        return chunks, {"status": "done", "items": []}

    def fake_embedding_backend_status():
        status_calls.append(True)
        if len(status_calls) == 1:
            return {"mode": "openai-compatible", "last_error": ""}
        return {"mode": "unavailable", "last_error": "向量化接口调用失败：429 Too Many Requests"}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank_chunks)
    monkeypatch.setattr(retrieval, "embedding_backend_status", fake_embedding_backend_status)

    chunks, trace = asyncio.run(
        retrieval.retrieve_knowledge(
            "考勤 迟到 处罚",
            knowledge_base_id=1,
            db=object(),
            query_plan={
                "original_question": "迟到30分钟以内罚款多少钱",
                "simplified_question": "迟到罚款标准",
                "sub_questions": [],
                "rewrites": [],
                "keywords": ["考勤", "迟到"],
                "required_evidence": [],
            },
        )
    )

    assert [chunk["route"] for chunk in chunks] == ["keyword"]
    assert trace["embedding_error"]
    assert trace["embedding"]["mode"] == "unavailable"
    assert [route["count"] for route in trace["routes"]] == [0, 0, 1]
    assert len(status_calls) == 2


def test_retrieve_knowledge_degrades_only_the_failing_vector_route(monkeypatch):
    """单路向量化失败只降级该路：其余向量路与关键词路照常返回（并发 x 逐路隔离的并集语义）。"""
    healthy_chunk = {
        "id": "vec-2",
        "chunk_id": "2",
        "content": "迟到30分钟以内罚款50元",
        "file_name": "考勤制度.txt",
        "file_id": 1,
        "route": "simplified",
    }
    keyword_chunk = {
        "id": "kw-1",
        "chunk_id": "kw-1",
        "content": "迟到超过30分钟按旷工处理",
        "file_name": "考勤制度.txt",
        "file_id": 2,
        "route": "keyword",
    }
    vector_calls = []

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        vector_calls.append(route)
        if route == "planned":
            raise retrieval.EmbeddingBackendError("向量化接口调用失败：429 Too Many Requests")
        return [healthy_chunk]

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        return [keyword_chunk]

    async def fake_rerank_chunks(question, chunks):
        return chunks, {"status": "done", "items": []}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank_chunks)

    chunks, trace = asyncio.run(
        retrieval.retrieve_knowledge(
            "考勤 迟到 处罚",
            knowledge_base_id=1,
            db=object(),
            query_plan={
                "original_question": "迟到30分钟以内罚款多少钱",
                "simplified_question": "迟到罚款标准",
                "sub_questions": [],
                "rewrites": [],
                "keywords": ["考勤", "迟到"],
                "required_evidence": [],
            },
        )
    )

    # 失败路仍然被检索过（不是提前退出），只是该路 0 命中；其余路不受影响。
    assert set(vector_calls) == {"planned", "simplified"}
    assert [item["route"] for item in trace["routes"]] == ["planned", "simplified", "keyword"]
    assert [item["count"] for item in trace["routes"]] == [0, 1, 1]
    assert trace["embedding_error"]
    # 失败路没有退化成哈希向量召回（否则这里会混入该路的伪命中）。
    assert {chunk["id"] for chunk in chunks} >= {"vec-2", "kw-1"}
    assert not any(chunk.get("route") == "planned" for chunk in chunks)


def test_retrieve_knowledge_waits_for_all_recalls_before_propagating(monkeypatch):
    """非降级异常上抛前必须等全部召回结束：否则兄弟线程会在请求级 Session 关闭后继续跑。"""
    finished = []
    lock = threading.Lock()

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        if route == "planned":
            raise RuntimeError("boom")
        try:
            time.sleep(0.2)
            return []
        finally:
            with lock:
                finished.append(route)

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        try:
            time.sleep(0.2)
            return []
        finally:
            with lock:
                finished.append("keyword")

    async def fake_rerank_chunks(question, chunks):
        return chunks, {"status": "done", "items": []}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank_chunks)

    # 自建事件循环而不是 asyncio.run：asyncio.run 收尾会 join 线程池，
    # 把「异常上抛瞬间兄弟线程还在跑」这个差异掩盖掉，用不了作用例的判别力。
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            loop.run_until_complete(
                retrieval.retrieve_knowledge(
                    "考勤 迟到 处罚",
                    knowledge_base_id=1,
                    db=object(),
                    query_plan={
                        "original_question": "迟到30分钟以内罚款多少钱",
                        "simplified_question": "迟到罚款标准",
                        "sub_questions": [],
                        "rewrites": [],
                        "keywords": ["考勤", "迟到"],
                        "required_evidence": [],
                    },
                )
            )

        # 非 EmbeddingBackendError 异常原样上抛，但上抛那一刻其余召回必须已全部结束
        # （否则请求级 Session 已关闭、兄弟线程仍在使用它）。
        assert set(finished) == {"simplified", "keyword"}
    finally:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_build_route_specs_does_not_reintroduce_empty_question():
    assert retrieval._build_route_specs("   ", {}) == []


def test_build_route_specs_preserves_falsy_query_values():
    assert retrieval._build_route_specs(0, {"simplified_question": False}) == [
        ("planned", "0"),
        ("simplified", "False"),
    ]


def test_route_decision_treats_nan_confidence_as_fallback():
    decision = retrieval._normalize_decision(
        {
            "need_rag": False,
            "confidence": "nan",
            "reason": "bad confidence",
        }
    )

    assert decision["route"] == "rag"
    assert decision["source"] == "fallback"


def test_clip_preserves_falsy_values():
    assert retrieval._clip(0, 10) == "0"
    assert retrieval._clip(False, 10) == "False"
    assert retrieval._clip(None, 10) == ""


def test_normalize_for_match_preserves_falsy_values():
    assert retrieval._normalize_for_match(0) == "0"
    assert retrieval._normalize_for_match(False) == "false"
    assert retrieval._normalize_for_match(None) == ""


def test_keyword_helpers_only_treat_none_as_empty_text():
    assert retrieval._dedupe_keywords(["迟到", False, 0, None]) == ["迟到", "False"]
    assert retrieval._expand_keywords([False, 0, None]) == ["False"]
    assert retrieval._fallback_keywords(False) == ["False"]


def test_retrieve_knowledge_empty_question_does_not_add_empty_keyword_route(monkeypatch):
    vector_calls = []
    keyword_calls = []

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        vector_calls.append((route, query))
        return []

    def fake_keyword_recall(db, knowledge_base_id, keywords, top_k):
        keyword_calls.append(keywords)
        return []

    async def fake_rerank_chunks(question, chunks):
        return [], {"status": "skipped", "items": []}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", fake_keyword_recall)
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank_chunks)

    chunks, trace = asyncio.run(
        retrieval.retrieve_knowledge(
            "   ",
            knowledge_base_id=1,
            db=object(),
            query_plan={"keywords": [], "required_evidence": []},
        )
    )

    assert chunks == []
    assert trace["routes"] == []
    assert vector_calls == []
    assert keyword_calls == []


def test_rrf_fuse_skips_non_list_route_results():
    fused = retrieval.rrf_fuse([
        ("vector", None),
        ("keyword", [{"file_id": 1, "chunk_id": "a", "content": "hit"}]),
    ])

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "a"
    assert fused[0]["routes"] == [{"route": "keyword", "rank": 1}]


def test_rrf_fuse_skips_malformed_route_entries():
    fused = retrieval.rrf_fuse([
        None,
        ("too-short",),
        ("keyword", [{"file_id": 1, "chunk_id": "a", "content": "hit"}]),
    ])

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "a"


def test_rrf_fuse_skips_non_dict_chunks():
    fused = retrieval.rrf_fuse([
        ("keyword", [None, "bad", {"file_id": 1, "chunk_id": "a", "content": "hit"}]),
    ])

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "a"


class _FakeFileQuery:
    """只实现 keyword_recall 用到的那部分查询：列投影、filter/order_by/limit 和 all()。

    issue #59 之后取数层改成「按主键游标分批 + 带 LIMIT」，条件里既有等值也有 ``id >``，
    所以这里真的按条件过滤并切片，避免 fake 比真实查询宽松（宽松会让分页死循环）。
    """

    def __init__(self, files, columns):
        self._files = list(files)
        self._columns = columns or (retrieval.KnowledgeFile.id, retrieval.KnowledgeFile.name, retrieval.KnowledgeFile.content)
        self._criteria = []
        self._limit = None

    def filter(self, *criteria):
        self._criteria.extend(criteria)
        return self

    def order_by(self, *columns):
        keys = [column.key for column in columns]
        self._files.sort(key=lambda item: tuple(getattr(item, key) for key in keys))
        return self

    def limit(self, count):
        self._limit = count
        return self

    def all(self):
        rows = [item for item in self._files if all(_matches_file(item, criterion) for criterion in self._criteria)]
        if self._limit is not None:
            rows = rows[: self._limit]
        return [tuple(getattr(item, column.key) for column in self._columns) for item in rows]


def _matches_file(file, criterion):
    """只判定 ``列 == 值`` / ``列 > 值``（分页游标必须生效，否则会一直取到同一批）。

    LIKE 预筛等表达式在这里按「一律通过」处理：这个 fake 只关心「给一批文件，召回结果
    是什么」；SQL 谓词本身（预筛、知识库归属、LIMIT）由 tests/test_retrieval_acceptance.py
    的真实 SQLite 数据库和 test_keyword_recall_memory_59.py 的语句断言覆盖。
    """
    left = getattr(criterion, "left", None)
    if left is None or not hasattr(left, "key"):
        return True
    value = getattr(file, left.key)
    expected = getattr(criterion.right, "value", criterion.right)
    if criterion.operator is operators.eq:
        return value == expected
    if criterion.operator is operators.gt:
        return value > expected
    return True


class _FakeFileDb:
    """keyword_recall 只需要 db.query(KnowledgeFile.id, .name, .content) + 分批查询。"""

    def __init__(self, files):
        self.files = files

    def query(self, *columns):
        return _FakeFileQuery(self.files, columns)


class _FakeKnowledgeFile:
    def __init__(self, file_id, name, content, knowledge_base_id=1):
        self.id = file_id
        self.name = name
        self.content = content
        self.size = len(content)
        self.knowledge_base_id = knowledge_base_id


def test_keyword_recall_drops_window_that_misses_the_query(monkeypatch):
    """issue #56：文件级命中不代表每个窗口都相关，跑题窗口不得进入关键字候选。"""
    filler = "本制度由行政部负责解释。" * 85
    document = (
        "报销申请需提交原始发票，由财务部在五个工作日内完成审核并付款。"
        + filler
        + "迟到30分钟以内罚款50元，早退按同等标准处理，由人事部按月汇总。"
        + "员工如有疑问可向人事部咨询。" * 20
    )
    keywords = retrieval._expand_keywords(["报销", "发票", "财务"])

    hits = retrieval.keyword_recall(
        _FakeFileDb([_FakeKnowledgeFile(7, "员工手册.md", document)]), 1, keywords, 8
    )

    assert hits
    # 第 1 位必须是真正命中本次查询关键词的窗口，而不是另一段考勤条款。
    assert any(keyword in hits[0]["content"] for keyword in keywords)
    assert "迟到" not in hits[0]["content"]
    assert all(hit["keyword_hits"] > 0 for hit in hits)
    assert all(
        any(keyword in hit["content"] for keyword in keywords) for hit in hits
    )


def test_keyword_recall_drops_off_topic_window_that_only_hits_hardcoded_phrases():
    """非考勤语料同样成立：只命中写死场景短语（30分钟以内）的窗口必须落选。"""
    filler = "本制度由行政部负责解释。" * 80
    document = (
        "报销申请需提交原始发票，由财务部审核。"
        + filler
        + "付款审批将在30分钟以内完成，超时自动升级。"
    )
    keywords = retrieval._expand_keywords(["报销", "发票", "财务"])

    hits = retrieval.keyword_recall(
        _FakeFileDb([_FakeKnowledgeFile(7, "财务制度.md", document)]), 1, keywords, 8
    )

    assert hits
    assert all("30分钟以内" not in hit["content"] for hit in hits)
    assert all(
        any(keyword in hit["content"] for keyword in keywords) for hit in hits
    )


def test_keyword_score_ignores_scenario_phrases_absent_from_the_query():
    """issue #56：内容自身的场景特征词不再加分，零命中内容必须得 0 分。"""
    keywords = retrieval._expand_keywords(["报销", "发票", "财务"])
    on_topic = "报销申请需提交原始发票，由财务部在五个工作日内完成审核并付款。"

    for off_topic in (
        "迟到30分钟以内罚款50元，早退按同等标准处理，由人事部按月汇总。",
        "员工须按时上下班，考勤记录由人事部按月汇总，罚款200元。",
        "付款审批将在30分钟以内完成，超时自动升级。",
    ):
        assert retrieval._keyword_score(off_topic, keywords) == 0.0
        assert retrieval._keyword_score(off_topic, keywords) < retrieval._keyword_score(on_topic, keywords)
