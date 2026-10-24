"""issue #288: BM25 keyword ranking.

The keyword route pre-filters with SQL ``LIKE`` but used to rank windows with a
hand-tuned hit count. These tests pin the BM25 behaviour on top of that route:

* exact terms survive a wall of high-frequency generic words;
* document length is normalised;
* Chinese unigram/bigram and ASCII-code tokenisation is explicit;
* document frequency comes from **all** knowledge-base windows, not the ``LIKE``
  candidate set;
* ``BM25_ENABLED=false`` and every BM25 failure path fall back to the legacy order
  without raising into the chat chain;
* full BM25 scoring is bounded by ``BM25_MAX_SCORED_CANDIDATES``: thousands of LIKE
  matches still trigger at most ``budget`` ``score_text`` calls, and the trace records
  that the budget was exceeded instead of dropping candidates silently;
* the vocabulary cap, stable cache keys, single-flight rebuilds and prewarm;
* the 20-question precise-term fixture reports MRR@5 / Recall@5 for both rankings,
  now over realistic paragraphs plus near-miss negatives.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.session import Base
from model.models import KnowledgeFile
from rag import bm25, retrieval

KB = 288
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bm25_288_exact_terms.json"


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


@pytest.fixture(autouse=True)
def _clean_bm25_cache():
    retrieval.clear_bm25_cache()
    yield
    retrieval.clear_bm25_cache()


def _new_db(tmp_path, name):
    engine = create_engine(
        f"sqlite:///{(tmp_path / name).as_posix()}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[KnowledgeFile.__table__])
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()
    return engine, db


def _add_files(db, rows, knowledge_base_id=KB):
    db.add_all(
        [
            KnowledgeFile(
                knowledge_base_id=knowledge_base_id,
                name=name,
                size=len(content),
                content=content,
            )
            for name, content in rows
        ]
    )
    db.commit()
    db.expunge_all()


def _recall_with_diagnostics(db, keywords, top_k):
    diagnostics = retrieval._new_bm25_diagnostics()
    token = retrieval._BM25_DIAGNOSTICS.set(diagnostics)
    try:
        chunks = retrieval.keyword_recall(db, KB, keywords, top_k)
    finally:
        retrieval._BM25_DIAGNOSTICS.reset(token)
    return chunks, diagnostics


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenizer_emits_cjk_unigrams_bigrams_and_ascii_terms():
    tokens = bm25.tokenize("第 42 条 TS-999 5000 元")

    # CJK runs yield every unigram and adjacent bigram.
    assert {"第", "条", "元"} <= set(tokens)
    # Consecutive ASCII letters/digits are whole tokens, and code-like spellings keep
    # both the joined form and its parts.
    assert "42" in tokens
    assert "5000" in tokens
    assert "ts-999" in tokens
    assert {"ts", "999"} <= set(tokens)


def test_tokenizer_ignores_whitespace_inside_cjk_runs():
    assert "考假" in bm25.tokenize("考 假")
    assert "考假" in bm25.tokenize("考\n假")


# ---------------------------------------------------------------------------
# Length normalisation
# ---------------------------------------------------------------------------


def test_bm25_length_normalisation_prefers_the_short_window():
    short = "TS-999"
    long = "TS-999" + "填充" * 400
    index = bm25.BM25Index([short, long], k1=1.5, b=0.75)
    terms = bm25.query_terms("TS-999")

    # Same term frequency, but the long window must not outrank the short one.
    assert index.score_text(short, terms) > index.score_text(long, terms)


# ---------------------------------------------------------------------------
# Exact terms against high-frequency generic words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", ["第 42 条", "TS-999", "5000 元"])
def test_exact_terms_rank_first_despite_generic_word_noise(tmp_path, term):
    _, db = _new_db(tmp_path, f"exact-{term}.db")
    _add_files(
        db,
        [(f"目标-{term}.txt", f"精确凭据：{term}。仅此一处。")]
        + [(f"噪声{index}.txt", "制度" * 40 + "员工" * 40) for index in range(6)],
    )

    chunks, diagnostics = _recall_with_diagnostics(db, [term, "制度", "员工"], 5)

    assert chunks
    assert term in chunks[0]["content"]
    assert chunks[0]["bm25_rank"] == 1
    assert chunks[0]["bm25_score"] > 0
    # The legacy score is still carried for final-selection compatibility.
    assert chunks[0]["keyword_score"] > 0
    assert diagnostics["status"] == "ok"
    assert diagnostics["candidates"] > 0


# ---------------------------------------------------------------------------
# DF corpus scope
# ---------------------------------------------------------------------------


def test_df_comes_from_every_kb_window_not_the_like_candidates(tmp_path):
    """A token may appear outside the LIKE candidates; IDF must still see those rows.

    ``TS`` occurs in ten files that cannot match the query ``TS-999`` (no ``999``), so
    a candidate-only DF corpus would report df(ts)=1 and a much larger IDF. Building
    from all 11 windows must report df(ts)=11.
    """
    engine_db = _new_db(tmp_path, "df-corpus.db")
    _, db = engine_db
    _add_files(
        db,
        [("目标.txt", "TS-999 是错误码。")]
        + [(f"其他{index}.txt", "TS 部门公告。") for index in range(10)],
    )

    index, cache_hit, reason, observed, observed_terms = retrieval._bm25_index_for(db, KB)

    assert index is not None
    assert reason == ""
    assert cache_hit is False
    assert observed == 11
    assert observed_terms == index.term_count
    assert index.doc_count == 11
    assert index.df["ts"] == 11
    assert index.df["ts-999"] == 1

    candidate_only = bm25.BM25Index(["TS-999 是错误码。"])
    assert candidate_only.df["ts"] == 1
    # More documents containing "ts" => lower IDF. This is the observable difference
    # between the full-KB corpus and a candidate-set corpus.
    assert index.idf("ts") < candidate_only.idf("ts")

    chunks, diagnostics = _recall_with_diagnostics(db, ["TS-999"], 5)

    assert [chunk["file_name"] for chunk in chunks] == ["目标.txt"]
    assert diagnostics["status"] == "ok"
    assert diagnostics["corpus_docs"] == 11
    # Only one LIKE candidate was scored, but IDF saw all 11 windows.
    assert diagnostics["candidates"] == 1


# ---------------------------------------------------------------------------
# Index cache
# ---------------------------------------------------------------------------


def test_index_cache_is_reused_and_invalidated_by_kb_changes(tmp_path):
    _, db = _new_db(tmp_path, "cache.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])

    first, cache_hit, _reason, observed, _terms = retrieval._bm25_index_for(db, KB)
    assert first is not None
    assert cache_hit is False
    assert observed == 1

    second, cache_hit, _reason, observed, _terms = retrieval._bm25_index_for(db, KB)
    assert second is first
    assert cache_hit is True
    assert observed == 1

    # Adding a file changes the ``(file_id, size)`` signature, so the next lookup
    # rebuilds from the new full corpus.
    _add_files(db, [("报销.txt", "报销流程：发票需要在30天内提交。")])
    rebuilt, cache_hit, _reason, observed, _terms = retrieval._bm25_index_for(db, KB)
    assert rebuilt is not first
    assert cache_hit is False
    assert observed == 2


def test_delete_and_same_size_replace_each_rebuild_the_corpus(tmp_path):
    _, db = _new_db(tmp_path, "invalidate.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])
    first, *_ = retrieval._bm25_index_for(db, KB)
    assert first.doc_count == 1

    db.query(KnowledgeFile).delete()
    db.commit()
    emptied, cache_hit, reason, observed, terms = retrieval._bm25_index_for(db, KB)
    assert emptied is not first
    assert cache_hit is False and reason == ""
    assert observed == 0 and terms == 0

    # Same size, new id: add/delete/replace always changes the id set, which is
    # exactly what the cache signature records.
    _add_files(db, [("早退.txt", "早退30分钟以内罚款50元。")])
    replaced, cache_hit, _reason, observed, _terms = retrieval._bm25_index_for(db, KB)
    assert replaced is not first and replaced is not emptied
    assert cache_hit is False and observed == 1
    assert replaced.doc_count == 1


def test_cache_key_covers_every_scoring_parameter_and_k1_change_rebuilds(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "cache-params.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])

    key = retrieval._bm25_cache_key("db", KB)
    assert key[2:] == (
        float(retrieval.BM25_K1),
        float(retrieval.BM25_B),
        int(retrieval.BM25_MAX_DOCS),
        int(retrieval.BM25_MAX_TERMS),
    )

    first, cache_hit, *_ = retrieval._bm25_index_for(db, KB)
    assert cache_hit is False
    monkeypatch.setattr(retrieval, "BM25_K1", retrieval.BM25_K1 + 1.0)
    second, cache_hit, *_ = retrieval._bm25_index_for(db, KB)
    assert second is not first
    assert cache_hit is False
    assert second.k1 == pytest.approx(retrieval.BM25_K1)


def test_single_flight_builds_a_cache_key_once_for_concurrent_first_queries(monkeypatch):
    monkeypatch.setattr(retrieval, "_bm25_cache_scope", lambda _db: "single-flight-db")
    monkeypatch.setattr(retrieval, "_bm25_signature", lambda _db, _kb: ())
    builds = []

    def slow_build(_db, _kb):
        builds.append(threading.get_ident())
        time.sleep(0.05)
        return bm25.BM25Index(["迟到30分钟以内罚款50元。"]), "", 1, 1

    monkeypatch.setattr(retrieval, "_build_bm25_index", slow_build)

    barrier = threading.Barrier(8)
    errors = []

    def worker():
        try:
            barrier.wait()
            index, cache_hit, *_ = retrieval._bm25_index_for(object(), KB)
            assert index is not None
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert errors == []
    assert len(builds) == 1


def test_prewarm_populates_the_cache_so_the_first_query_is_a_hit(tmp_path, monkeypatch):
    engine, db = _new_db(tmp_path, "prewarm.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])
    maker = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    monkeypatch.setattr(retrieval, "SessionLocal", maker)

    retrieval._prewarm_bm25(KB)

    index, cache_hit, *_ = retrieval._bm25_index_for(db, KB)
    assert index is not None
    assert cache_hit is True


# ---------------------------------------------------------------------------
# Cost bounds
# ---------------------------------------------------------------------------


def _window_noise(blocks: int) -> str:
    """~one keyword window per 720 chars, each containing 「制度」.

    718 chars per block keeps tokenisation tiny (one CJK run + one ASCII run), so the
    test is about the scoring budget, not about how fast the tokenizer is.
    """
    return ("制度" + "x" * 716) * blocks


def test_full_bm25_scoring_is_bounded_by_the_candidate_budget(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "bounded.db")
    _add_files(
        db,
        [
            ("目标.txt", "精确凭据：TS-999。仅此一处。"),
            ("大文件.txt", _window_noise(3000)),
        ],
    )
    monkeypatch.setattr(retrieval, "BM25_MAX_SCORED_CANDIDATES", 32)
    scored_windows = []
    original = retrieval.BM25Index.score_text

    def counting_score_text(self, text, terms):
        scored_windows.append(text)
        return original(self, text, terms)

    monkeypatch.setattr(retrieval.BM25Index, "score_text", counting_score_text)

    chunks, diagnostics = _recall_with_diagnostics(db, ["TS-999", "制度"], 5)

    # Thousands of windows match; exactly the budget is fully scored. This is a
    # count assertion, not a wall-clock threshold, so it cannot go flaky on CI.
    assert diagnostics["candidates"] >= 2000
    assert diagnostics["candidate_budget"] == 32
    assert diagnostics["scored_candidates"] == 32
    assert diagnostics["scoring_budget_exceeded"] is True
    assert len(scored_windows) == 32
    # The rare exact term still wins: the cheap proxy is IDF-weighted, so a wall of
    # high-frequency 制度 does not crowd it out of the scoring budget. This is the
    # intended behaviour of the proxy, not a guarantee for every corpus.
    assert chunks[0]["bm25_rank"] == 1
    assert "TS-999" in chunks[0]["content"]


def test_bm25_disabled_never_builds_an_index(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "disabled.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("BM25 index must not be built while BM25_ENABLED is false")

    monkeypatch.setattr(retrieval, "BM25_ENABLED", False)
    monkeypatch.setattr(retrieval, "_bm25_index_for", forbidden)

    chunks, diagnostics = _recall_with_diagnostics(db, ["迟到"], 5)

    assert chunks
    assert diagnostics["status"] == "disabled"
    assert all("bm25_score" not in chunk for chunk in chunks)


def test_no_matching_candidates_skip_the_index_build(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "no-candidates.db")
    _add_files(db, [("无关.txt", "制度员工制度员工。")])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no LIKE candidate => the full-corpus index is not needed")

    monkeypatch.setattr(retrieval, "_bm25_index_for", forbidden)

    chunks, diagnostics = _recall_with_diagnostics(db, ["TS-999"], 5)

    assert chunks == []
    assert diagnostics["status"] == "skipped"
    assert diagnostics["reason"] == "no_candidates"
    assert diagnostics["candidates"] == 0


def test_vocabulary_cap_degrades_with_trace_and_operator_warning(tmp_path, monkeypatch, caplog):
    _, db = _new_db(tmp_path, "vocabulary.db")
    # 200 distinct CJK characters plus the query term: unigrams and bigrams blow past
    # the lowered cap immediately.
    content = "制度" + "".join(chr(0x4E00 + index) for index in range(200))
    _add_files(db, [("大词表.txt", content)])
    monkeypatch.setattr(retrieval, "BM25_MAX_TERMS", 40)

    with caplog.at_level(logging.WARNING, logger="rag.retrieval"):
        chunks, diagnostics = _recall_with_diagnostics(db, ["制度"], 5)

    assert chunks
    assert diagnostics["status"] == "degraded"
    assert diagnostics["degraded"] is True
    assert diagnostics["reason"] == "vocabulary_too_large"
    assert diagnostics["corpus_terms"] > 40
    assert all("bm25_score" not in chunk for chunk in chunks)
    assert any("BM25 corpus degraded" in record.message for record in caplog.records)


def test_bm25_index_raises_when_the_vocabulary_cap_is_exceeded():
    with pytest.raises(bm25.BM25VocabularyTooLarge):
        bm25.BM25Index(
            ["晚" + "".join(chr(0x4E00 + index) for index in range(100))],
            max_terms=10,
        )


def test_fullwidth_digits_tokenise_like_ascii():
    assert bm25.tokenize("ＴＳ－９９９") == bm25.tokenize("TS-999")


def test_bm25_parameters_are_validated():
    from config import normalize_bm25_b, normalize_bm25_k1

    assert normalize_bm25_k1(float("nan")) == pytest.approx(1.5)
    assert normalize_bm25_k1(float("inf")) == pytest.approx(1.5)
    assert normalize_bm25_k1(0) == pytest.approx(1.5)
    assert normalize_bm25_k1(2.0) == pytest.approx(2.0)
    assert normalize_bm25_b(float("nan")) == pytest.approx(0.75)
    assert normalize_bm25_b(1.5) == pytest.approx(0.75)
    assert normalize_bm25_b(0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


LONG_WINDOW_TERM = "迟到"
LONG_DOCUMENT = LONG_WINDOW_TERM + "填" * 5000


def test_bm25_enabled_changes_order_but_keeps_legacy_score(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "toggle.db")
    # Both windows have the same legacy score (one hit + one unique position); the
    # long one is first by file id. BM25 length normalisation must move the short one
    # ahead without changing keyword_score.
    _add_files(db, [("长文件.txt", LONG_DOCUMENT), ("短文件.txt", LONG_WINDOW_TERM)])

    monkeypatch.setattr(retrieval, "BM25_ENABLED", True)
    enabled, enabled_diagnostics = _recall_with_diagnostics(db, [LONG_WINDOW_TERM], 2)

    assert [chunk["file_name"] for chunk in enabled] == ["短文件.txt", "长文件.txt"]
    assert enabled[0]["keyword_score"] == enabled[1]["keyword_score"]
    assert [chunk["bm25_rank"] for chunk in enabled] == [1, 2]
    assert enabled[0]["bm25_score"] > enabled[1]["bm25_score"]
    assert enabled_diagnostics["status"] == "ok"

    monkeypatch.setattr(retrieval, "BM25_ENABLED", False)
    legacy, legacy_diagnostics = _recall_with_diagnostics(db, [LONG_WINDOW_TERM], 2)

    assert [chunk["file_name"] for chunk in legacy] == ["长文件.txt", "短文件.txt"]
    assert all("bm25_score" not in chunk for chunk in legacy)
    assert all("bm25_rank" not in chunk for chunk in legacy)
    assert legacy_diagnostics["status"] == "disabled"


def test_corpus_limit_degrades_to_legacy_ordering_with_trace(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "too-large.db")
    _add_files(db, [(f"文件{index}.txt", "迟到30分钟以内罚款50元。") for index in range(3)])
    monkeypatch.setattr(retrieval, "BM25_MAX_DOCS", 2)

    chunks, diagnostics = _recall_with_diagnostics(db, ["迟到"], 5)

    assert chunks
    assert diagnostics["status"] == "degraded"
    assert diagnostics["degraded"] is True
    assert diagnostics["reason"] == "corpus_too_large"
    assert diagnostics["corpus_docs"] == 3
    assert all("bm25_score" not in chunk for chunk in chunks)


class _BoomIndex:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("SECRET_BUILD_DETAIL")


def test_index_build_failure_falls_back_without_raising(tmp_path, monkeypatch, caplog):
    _, db = _new_db(tmp_path, "build-failure.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])
    monkeypatch.setattr(retrieval, "BM25Index", _BoomIndex)

    with caplog.at_level(logging.WARNING, logger="rag.retrieval"):
        chunks, diagnostics = _recall_with_diagnostics(db, ["迟到"], 5)

    assert chunks
    assert all("bm25_score" not in chunk for chunk in chunks)
    assert diagnostics["status"] == "fallback"
    assert diagnostics["degraded"] is True
    assert diagnostics["reason"] == "index_build_failed"
    # The exception text goes to the log, never to the trace/diagnostics payload.
    assert "SECRET_BUILD_DETAIL" not in json.dumps(diagnostics, ensure_ascii=False)
    assert any("BM25 index build failed" in record.message for record in caplog.records)


def test_scoring_failure_falls_back_without_raising(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "score-failure.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])

    def _raise(_self, _text, _terms):
        raise RuntimeError("SECRET_SCORE_DETAIL")

    monkeypatch.setattr(retrieval.BM25Index, "score_text", _raise)

    chunks, diagnostics = _recall_with_diagnostics(db, ["迟到"], 5)

    assert chunks
    assert all("bm25_score" not in chunk for chunk in chunks)
    assert diagnostics["status"] == "fallback"
    assert diagnostics["degraded"] is True
    assert diagnostics["reason"] == "scoring_failed"
    assert "SECRET_SCORE_DETAIL" not in json.dumps(diagnostics, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Trace integration
# ---------------------------------------------------------------------------


def _retrieve_for_trace(db, monkeypatch):
    monkeypatch.setattr(retrieval, "query_vectors", lambda *_args, **_kwargs: [])

    async def fake_rerank(_question, chunks):
        return chunks, {"status": "done", "items": []}

    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank)
    return asyncio.run(
        retrieval.retrieve_knowledge(
            "迟到怎么处理",
            knowledge_base_id=KB,
            db=db,
            query_plan={
                "original_question": "迟到怎么处理",
                "simplified_question": "迟到 处理",
                "keywords": ["迟到"],
                "required_evidence": [],
            },
        )
    )


def test_retrieval_trace_records_bm25_sizes_and_parameters(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "trace.db")
    _add_files(
        db,
        [("考勤.txt", "迟到30分钟以内罚款50元。"), ("无关.txt", "制度员工制度员工。")],
    )

    _, trace = _retrieve_for_trace(db, monkeypatch)

    info = trace["bm25"]
    assert info["enabled"] is True
    assert info["k1"] == pytest.approx(1.5)
    assert info["b"] == pytest.approx(0.75)
    assert info["status"] == "ok"
    assert info["degraded"] is False
    assert info["corpus_docs"] == 2
    assert info["candidates"] > 0
    assert info["reason"] == ""


def test_retrieval_trace_does_not_echo_bm25_exception_text(tmp_path, monkeypatch):
    _, db = _new_db(tmp_path, "trace-failure.db")
    _add_files(db, [("考勤.txt", "迟到30分钟以内罚款50元。")])
    monkeypatch.setattr(retrieval, "BM25Index", _BoomIndex)

    _, trace = _retrieve_for_trace(db, monkeypatch)

    assert trace["bm25"]["status"] == "fallback"
    assert trace["bm25"]["reason"] == "index_build_failed"
    assert "SECRET_BUILD_DETAIL" not in json.dumps(trace, ensure_ascii=False)
    assert "SECRET_BUILD_DETAIL" not in json.dumps(
        [item for item in trace["routes"]], ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# 20-question precise-term fixture: legacy vs BM25 ranking evidence
# ---------------------------------------------------------------------------


def _load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_corpus(cases):
    """Real paragraphs for the targets plus a near-miss paragraph per case.

    The near misses share the surrounding wording (「第 … 条」, 「… 元」, the same
    error-code prefix) but differ in the exact number/code, so a ranker that only
    counts generic words cannot tell them apart from the target. The generic-noise
    files remain: they are what the legacy hit-count score prefers.
    """
    rows = []
    for case in cases:
        rows.append((f"目标-{case['id']}.txt", case["document"]))
        rows.append((f"近似-{case['id']}.txt", case["near_miss"]))
    rows.extend(
        (f"噪声{index}.txt", "制度" * 40 + "员工" * 40) for index in range(8)
    )
    return rows


def _rank_of(chunks, term):
    for rank, chunk in enumerate(chunks, start=1):
        if term in chunk["content"]:
            return rank
    return None


def _ranking_metrics(ranks, k=5):
    hits = [rank for rank in ranks if rank is not None and rank <= k]
    return {
        "mrr_at_5": sum(1.0 / rank for rank in hits) / len(ranks),
        "recall_at_5": len(hits) / len(ranks),
        "hits_at_5": len(hits),
        "ranked": len(ranks),
    }


def test_precise_term_fixture_bm25_is_not_worse_and_improves_at_least_one(tmp_path, monkeypatch):
    cases = _load_fixture()
    assert len(cases) == 20
    assert all(case["category"] == "exact_term" for case in cases)
    _, db = _new_db(tmp_path, "precise-terms.db")
    _add_files(db, _fixture_corpus(cases))

    legacy_ranks = []
    monkeypatch.setattr(retrieval, "BM25_ENABLED", False)
    for case in cases:
        term = case["exact_terms"][0]
        chunks = retrieval.keyword_recall(db, KB, [term, "制度", "员工"], 5)
        legacy_ranks.append(_rank_of(chunks, term))

    monkeypatch.setattr(retrieval, "BM25_ENABLED", True)
    retrieval.clear_bm25_cache()
    bm25_ranks = []
    for case in cases:
        term = case["exact_terms"][0]
        chunks = retrieval.keyword_recall(db, KB, [term, "制度", "员工"], 5)
        bm25_ranks.append(_rank_of(chunks, term))

    legacy = _ranking_metrics(legacy_ranks)
    ranked = _ranking_metrics(bm25_ranks)
    print(f"PRECISE_TERM_METRICS legacy={legacy} bm25={ranked}")

    assert ranked["mrr_at_5"] >= legacy["mrr_at_5"]
    assert ranked["recall_at_5"] >= legacy["recall_at_5"]
    improvements = sum(
        1
        for old, new in zip(legacy_ranks, bm25_ranks, strict=True)
        if new is not None and new <= 5 and (old is None or old > 5 or new < old)
    )
    assert improvements >= 1
    # The fixture is adversarial by construction: every target has the exact term and
    # the legacy score is dominated by repeated generic words, so BM25 should find all
    # 20 within the first five results.
    assert ranked["recall_at_5"] == 1.0
