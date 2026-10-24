#!/usr/bin/env python
"""Offline, reproducible RAG evaluation harness (phase 1, issue #293).

Phase 1 is deliberately offline: ``--dry-run`` never imports the backend, never reads
credentials and never opens a socket. It validates a fixed question set and writes a
JSON + Markdown report whose metric slots are explicitly ``null`` / ``not_run``. No
score is fabricated; the report only proves that the dataset and the report structure
are sound, so phase 2 can plug real retrieval and generation runs into the same shape.

Question-set format (JSON array, UTF-8). Every entry has these fields:

    id                stable string, unique inside the file
    question          question text
    expected_answer   reference answer (for an unanswerable entry: the refusal criteria)
    evidence          human-readable section/clause labels; [] only when answerable=false
    expected_doc      production ``file_name`` of the source file; one string for all
                      evidence units or a list parallel to ``expected_sections``
    expected_sections section heading/clause text that must appear in a chunk's
                      ``content``; a list with one entry per evidence unit
    category          single_fact | numeric_deadline | cross_section | table |
                      negation_exception | unanswerable | exact_term | ...
    answerable        boolean

``expected_doc`` + ``expected_sections`` make evidence matching executable against the
real retrieval shape (``file_name``/``chunk_id``/``content``). When they are absent the
harness falls back to the legacy ``evidence`` label-in-content rule, but the bundled
fixture always supplies them.

``exact_term`` entries also carry ``exact_terms`` (a non-empty list of literal terms) and
the question must contain at least one of those terms verbatim; they are the 20 exact-term
fixtures reserved for #288 (BM25). Any extra category name is accepted, but the six
required categories are reported when missing.

An empty question array is a validation error (exit code 1), not an accepted no-op.

Usage:

    python scripts/eval_rag.py --dry-run \
        --questions tests/fixtures/rag_eval_questions.json \
        --out ../RAG_ops/reports/eval-dry-run.json

``--out`` is the JSON path; the Markdown report defaults to the same stem with ``.md``
(override with ``--md-out``). Reports are intentionally written outside the repository.

Metric definitions live in ``docs/RAG_EVAL_BASELINE.md``; the same formulas are summarized
in ``METRIC_DEFINITIONS`` below. Exit codes: 0 = validation passed, 1 = validation errors
(report still written), 2 = unreadable input or unsupported mode.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "tests" / "fixtures" / "rag_eval_questions.json"

SCHEMA_VERSION = 1
REQUIRED_FIELDS = ("id", "question", "expected_answer", "evidence", "category", "answerable")
REQUIRED_CATEGORIES = (
    "single_fact",
    "numeric_deadline",
    "cross_section",
    "table",
    "negation_exception",
    "unanswerable",
)
EXACT_TERM_CATEGORY = "exact_term"

# Default difflib ratio used to flag near-duplicate question/answer pairs. The value is
# configurable from the CLI (``--duplicate-threshold``) and from ``validate_questions``.
DEFAULT_DUPLICATE_THRESHOLD = 0.9
# An answer pair is only flagged when the questions are at least this similar; this keeps
# short generic answers ("应明确拒答") from being reported across unrelated questions.
ANSWER_PAIR_QUESTION_FLOOR = 0.5

METRIC_KEYS = (
    "recall_at_5",
    "recall_at_20",
    "mrr_at_5",
    "citation_hit",
    "answer_correctness",
    "refusal_correctness",
    "ingestion_latency_ms",
    "ingestion_failure_rate",
)

METRIC_DEFINITIONS = {
    "recall_at_5": "检索前 5 条中包含的标准依据数 / 该题标准依据总数，可答题按题取平均。",
    "recall_at_20": "检索前 20 条中包含的标准依据数 / 该题标准依据总数，可答题按题取平均。",
    "mrr_at_5": "首个相关 chunk 排在检索前 5 位时取 1/rank，否则取 0；可答题按题取平均。",
    "citation_hit": "答案引用的来源命中标准依据的可答题数 / 参与判定的可答题数。",
    "answer_correctness": "答案判定正确的可答题数 / 参与判定的可答题数。",
    "refusal_correctness": "正确拒答的不可答题数 / 参与判定的不可答题数。",
    "ingestion_latency_ms": "单文件入库耗时（毫秒）的平均值。",
    "ingestion_failure_rate": "入库失败文件数 / 入库文件总数。",
}

# Legacy baseline: heading-aware *character* chunking. Since issue #287 chunk_text()
# slices by a token budget unless chunk_size/chunk_overlap are passed explicitly, so
# this is kept only for historical comparison -- it is NOT the current production path.
# Phase 1 records the configuration only; tests/test_eval_harness_293.py pins these
# values against the function's AST defaults.
CHARACTER_BASELINE = {
    "name": "legacy-character",
    "chunk_method": "chunk_text(chunk_size=..., chunk_overlap=...)",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "source": "backend/crud/knowledge_file.py:chunk_text",
}

# Current production baseline (issue #287): token-budget chunking selected by the
# default chunk_text() call path. Kept in sync with backend/config.py defaults; this
# script stays offline and does not import the backend. Reports label it separately so
# the legacy character numbers can never be mistaken for the current production baseline.
CURRENT_TOKEN_BASELINE = {
    "name": "current-token",
    "chunk_method": "chunk_text(file_id=...)",
    "budget_unit": "tokens",
    "token_limit": 500,
    "token_target": 400,
    "token_overlap": 50,
    "token_safety_ratio": 0.85,
    "source": "backend/crud/knowledge_file.py:chunk_text",
}


class QuestionSetError(ValueError):
    """The question file is unreadable or is not a JSON array."""


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_label(value) -> str:
    return " ".join(str(value or "").split()).casefold()


def _normalize_doc_name(value) -> str:
    """Normalize a production ``file_name`` for comparison (case/space/extension)."""
    text = _normalize_label(value)
    for suffix in (".markdown", ".md", ".txt", ".pdf", ".docx", ".doc"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def _doc_matches(chunk_file_name, expected_doc) -> bool:
    actual = _normalize_doc_name(chunk_file_name)
    expected = _normalize_doc_name(expected_doc)
    return bool(actual) and bool(expected) and actual == expected


def _section_in_content(chunk_content, section) -> bool:
    needle = _normalize_label(section)
    return bool(needle) and needle in _normalize_label(chunk_content)


def expected_evidence_units(question: dict) -> list:
    """Return executable expected-evidence units for a question.

    A unit is ``{"doc", "section", "label"}``. The structured ``expected_doc`` /
    ``expected_sections`` fields are preferred; when absent (legacy external sets) each
    ``evidence`` label becomes a unit with ``doc=None`` and ``section=None``.
    """
    sections = question.get("expected_sections")
    docs = question.get("expected_doc")
    labels = [label for label in question.get("evidence") or [] if _is_nonempty_str(label)]

    if isinstance(sections, list) and sections and all(_is_nonempty_str(item) for item in sections):
        if _is_nonempty_str(docs):
            doc_list = [docs] * len(sections)
        elif (
            isinstance(docs, list)
            and len(docs) == len(sections)
            and all(_is_nonempty_str(item) for item in docs)
        ):
            doc_list = docs
        else:
            doc_list = [None] * len(sections)
        return [
            {
                "doc": doc_list[index],
                "section": section,
                "label": labels[index] if index < len(labels) else section,
            }
            for index, section in enumerate(sections)
        ]

    return [{"doc": None, "section": None, "label": label} for label in labels]


def _chunk_matches_evidence_unit(chunk: dict, unit: dict) -> bool:
    """Match a production chunk against one structured evidence unit.

    The primary rule is ``file_name`` equality (extension-insensitive) plus the section
    heading/clause appearing in ``content``. Content-only chunks (no ``file_name``) and
    legacy units (no doc/section) fall back to label/section text matching so callers
    that only expose ``content`` still get an executable, if weaker, mapping.
    """
    content = chunk.get("content")
    doc = unit.get("doc")
    section = unit.get("section")
    label = unit.get("label")

    if doc:
        file_name = chunk.get("file_name")
        if _is_nonempty_str(file_name) and not _doc_matches(file_name, doc):
            return False
        # Content-only chunks (no file_name) fall back to the section heading or the full
        # evidence label, which itself names the source document.
        return _section_in_content(content, section) or (
            _is_nonempty_str(label) and _normalize_label(label) in _normalize_label(content)
        )

    if section:
        return _section_in_content(content, section)
    return _chunk_matches_evidence(chunk, label)


def _chunk_matches_evidence(chunk: dict, expected_label: str) -> bool:
    normalized_expected = _normalize_label(expected_label)
    if not normalized_expected:
        return False
    labels = chunk.get("evidence")
    if isinstance(labels, str):
        labels = [labels]
    elif not isinstance(labels, list):
        labels = []
    if any(_normalize_label(label) == normalized_expected for label in labels or []):
        return True
    # Fallback for callers that only expose chunk text: the clause label appears in it.
    return normalized_expected in _normalize_label(chunk.get("content"))


def _find_near_duplicates(questions: list, threshold: float = DEFAULT_DUPLICATE_THRESHOLD) -> list:
    """Return near-duplicate question/answer pairs using difflib ratios."""
    candidates = [
        (index, entry)
        for index, entry in enumerate(questions)
        if isinstance(entry, dict) and _is_nonempty_str(entry.get("id"))
    ]
    duplicates = []
    for left in range(len(candidates)):
        left_index, left_entry = candidates[left]
        for right in range(left + 1, len(candidates)):
            right_index, right_entry = candidates[right]
            question_similarity = difflib.SequenceMatcher(
                None,
                _normalize_label(left_entry.get("question")),
                _normalize_label(right_entry.get("question")),
            ).ratio()
            answer_similarity = difflib.SequenceMatcher(
                None,
                _normalize_label(left_entry.get("expected_answer")),
                _normalize_label(right_entry.get("expected_answer")),
            ).ratio()
            kind = None
            if question_similarity >= threshold:
                kind = "question"
            elif (
                answer_similarity >= threshold
                and question_similarity >= ANSWER_PAIR_QUESTION_FLOOR
            ):
                kind = "answer"
            if kind is None:
                continue
            duplicates.append(
                {
                    "kind": kind,
                    "ids": [left_entry.get("id"), right_entry.get("id")],
                    "indexes": [left_index, right_index],
                    "question_similarity": round(question_similarity, 6),
                    "answer_similarity": round(answer_similarity, 6),
                }
            )
    return duplicates


def _display_question_set(path: Path) -> dict:
    """Describe the input without leaking an absolute path from the host."""
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return {"name": path.name, "source": "external"}
    return {"name": path.name, "source": "repository", "relative_path": relative.as_posix()}


def load_questions(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QuestionSetError(f"无法读取问题集 {path.name}：{exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QuestionSetError(f"问题集 {path.name} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, list):
        raise QuestionSetError(f"问题集 {path.name} 的根节点必须是 JSON 数组")
    return payload


def _validate_evidence(entry: dict, entry_id, index: int, errors: list) -> None:
    evidence = entry.get("evidence")
    if not isinstance(evidence, list):
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "evidence",
                "reason": "必须是字符串数组",
            }
        )
        return
    if any(not _is_nonempty_str(item) for item in evidence):
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "evidence",
                "reason": "数组元素必须是非空字符串",
            }
        )
    if entry.get("answerable") is True and not evidence:
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "evidence",
                "reason": "可回答问题至少需要一个依据章节/条款",
            }
        )


def _validate_exact_terms(entry: dict, entry_id, index: int, errors: list, missing_fields: list) -> None:
    category = entry.get("category")
    exact_terms = entry.get("exact_terms")
    if category != EXACT_TERM_CATEGORY and exact_terms is None:
        return
    if category == EXACT_TERM_CATEGORY and exact_terms is None:
        missing_fields.append({"index": index, "id": entry_id, "fields": ["exact_terms"]})
        errors.append(
            {
                "code": "missing_fields",
                "index": index,
                "id": entry_id,
                "fields": ["exact_terms"],
            }
        )
        return
    if not isinstance(exact_terms, list) or not exact_terms:
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "exact_terms",
                "reason": "必须是非空字符串数组",
            }
        )
        return
    if any(not _is_nonempty_str(item) for item in exact_terms):
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "exact_terms",
                "reason": "数组元素必须是非空字符串",
            }
        )
        return
    question = entry.get("question")
    if category == EXACT_TERM_CATEGORY and _is_nonempty_str(question):
        normalized_question = _normalize_label(question)
        if not any(
            _normalize_label(term) in normalized_question
            for term in exact_terms
            if _is_nonempty_str(term)
        ):
            errors.append(
                {
                    "code": "exact_term_not_in_question",
                    "index": index,
                    "id": entry_id,
                    "field": "exact_terms",
                    "reason": "question 必须至少包含一个 exact_terms 字面词项",
                    "terms": list(exact_terms),
                }
            )


def _validate_expected_evidence(entry: dict, entry_id, index: int, errors: list, warnings: list) -> None:
    """Validate the optional structured evidence mapping."""
    docs = entry.get("expected_doc")
    sections = entry.get("expected_sections")
    if docs is None and sections is None:
        if entry.get("answerable") is True and entry.get("evidence"):
            warnings.append(
                {
                    "code": "missing_evidence_mapping",
                    "index": index,
                    "id": entry_id,
                    "reason": "建议提供 expected_doc + expected_sections，以便按 file_name 与章节匹配",
                }
            )
        return
    if docs is None or sections is None:
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "expected_doc" if docs is None else "expected_sections",
                "reason": "expected_doc 与 expected_sections 必须同时提供",
            }
        )
        return
    if _is_nonempty_str(docs):
        docs_valid = True
    elif isinstance(docs, list):
        docs_valid = bool(docs) and all(_is_nonempty_str(item) for item in docs)
    else:
        docs_valid = False
    if not docs_valid:
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "expected_doc",
                "reason": "必须是非空字符串或非空字符串数组",
            }
        )
    if not isinstance(sections, list) or not sections or any(
        not _is_nonempty_str(item) for item in sections
    ):
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "expected_sections",
                "reason": "必须是非空字符串数组",
            }
        )
        return
    if isinstance(docs, list) and len(docs) != len(sections):
        errors.append(
            {
                "code": "invalid_field",
                "index": index,
                "id": entry_id,
                "field": "expected_doc",
                "reason": "expected_doc 数组长度必须与 expected_sections 一致",
            }
        )
    evidence = entry.get("evidence")
    if isinstance(evidence, list) and evidence and len(evidence) != len(sections):
        warnings.append(
            {
                "code": "evidence_mapping_length_mismatch",
                "index": index,
                "id": entry_id,
                "evidence_count": len(evidence),
                "expected_sections_count": len(sections),
            }
        )


def validate_questions(
    questions: list, *, duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD
) -> dict:
    """Validate the dataset and return a stable, report-ready summary."""
    errors = []
    warnings = []
    missing_fields = []
    duplicates = []
    seen_ids = set()
    category_counts = {}
    answerable_count = 0
    unanswerable_count = 0

    if not questions:
        errors.append(
            {
                "code": "empty_question_set",
                "reason": "问题集至少需要一条问题；空数组视为校验失败",
            }
        )

    for index, entry in enumerate(questions):
        if not isinstance(entry, dict):
            errors.append({"code": "invalid_entry", "index": index, "reason": "条目必须是 JSON 对象"})
            continue

        entry_id = entry.get("id")
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            missing_fields.append({"index": index, "id": entry_id, "fields": missing})
            errors.append(
                {"code": "missing_fields", "index": index, "id": entry_id, "fields": missing}
            )

        for field in ("id", "question", "expected_answer", "category"):
            if field in entry and not _is_nonempty_str(entry.get(field)):
                errors.append(
                    {
                        "code": "invalid_field",
                        "index": index,
                        "id": entry_id,
                        "field": field,
                        "reason": "必须是非空字符串",
                    }
                )

        if _is_nonempty_str(entry_id):
            if entry_id in seen_ids:
                if entry_id not in duplicates:
                    duplicates.append(entry_id)
                    errors.append({"code": "duplicate_id", "id": entry_id})
            else:
                seen_ids.add(entry_id)

        if "answerable" in entry and not isinstance(entry.get("answerable"), bool):
            errors.append(
                {
                    "code": "invalid_field",
                    "index": index,
                    "id": entry_id,
                    "field": "answerable",
                    "reason": "必须是布尔值",
                }
            )

        if "evidence" in entry:
            _validate_evidence(entry, entry_id, index, errors)
        if "exact_terms" in entry or entry.get("category") == EXACT_TERM_CATEGORY:
            _validate_exact_terms(entry, entry_id, index, errors, missing_fields)
        _validate_expected_evidence(entry, entry_id, index, errors, warnings)

        category = entry.get("category")
        if _is_nonempty_str(category):
            category_counts[category] = category_counts.get(category, 0) + 1

        answerable = entry.get("answerable")
        if answerable is True:
            answerable_count += 1
        elif answerable is False:
            unanswerable_count += 1

        if category == "unanswerable" and answerable is True:
            warnings.append(
                {
                    "code": "answerable_category_mismatch",
                    "id": entry_id,
                    "category": category,
                    "answerable": answerable,
                }
            )
        elif _is_nonempty_str(category) and category != "unanswerable" and answerable is False:
            warnings.append(
                {
                    "code": "answerable_category_mismatch",
                    "id": entry_id,
                    "category": category,
                    "answerable": answerable,
                }
            )

    missing_required_categories = [
        category for category in REQUIRED_CATEGORIES if category_counts.get(category, 0) == 0
    ]
    for category in missing_required_categories:
        warnings.append({"code": "missing_required_category", "category": category})

    near_duplicates = _find_near_duplicates(questions, duplicate_threshold)
    for duplicate in near_duplicates:
        warnings.append(
            {
                "code": f"near_duplicate_{duplicate['kind']}",
                "ids": duplicate["ids"],
                "question_similarity": duplicate["question_similarity"],
                "answer_similarity": duplicate["answer_similarity"],
            }
        )

    return {
        "ok": not errors,
        "question_count": len(questions),
        "errors": errors,
        "warnings": warnings,
        "missing_fields": missing_fields,
        "duplicate_ids": duplicates,
        "near_duplicates": near_duplicates,
        "duplicate_threshold": duplicate_threshold,
        "category_counts": dict(sorted(category_counts.items())),
        "missing_required_categories": missing_required_categories,
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
    }


def question_set_sha256(questions: list) -> str:
    canonical = json.dumps(questions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_retrieval_metrics(
    questions: list, retrieval_results: dict, k_values=(5, 20), mrr_k: int = 5
) -> dict:
    """Compute Recall@k, MRR@k and unmatched evidence from retrieval results.

    ``retrieval_results`` maps a question id to an ordered list of production-shaped
    chunks (``file_name``/``chunk_id``/``content``). Relevance is decided by
    :func:`_chunk_matches_evidence_unit`: ``file_name`` plus section heading/text for
    structured fixtures, with a content-only fallback. Only answerable questions with at
    least one expected evidence unit and a supplied result row are evaluated.
    """
    answerable = [q for q in questions if isinstance(q, dict) and q.get("answerable") is True]
    per_question = {}
    recall_rows = {k: [] for k in k_values}
    reciprocal_ranks = []
    unmatched_total = 0

    for question in answerable:
        question_id = question.get("id")
        if not _is_nonempty_str(question_id) or question_id not in retrieval_results:
            continue
        units = expected_evidence_units(question)
        if not units:
            continue

        chunks = retrieval_results.get(question_id) or []
        if not isinstance(chunks, list):
            chunks = []
        chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]

        row = {
            "expected_evidence": len(units),
            "evidence_units": [unit["label"] for unit in units],
        }
        for k in k_values:
            top_k = chunks[:k]
            matched = sum(
                1
                for unit in units
                if any(_chunk_matches_evidence_unit(chunk, unit) for chunk in top_k)
            )
            ratio = matched / len(units)
            recall_rows[k].append(ratio)
            row[f"recall_at_{k}"] = round(ratio, 6)

        first_relevant_rank = None
        for rank, chunk in enumerate(chunks, start=1):
            if any(_chunk_matches_evidence_unit(chunk, unit) for unit in units):
                first_relevant_rank = rank
                break
        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None and first_relevant_rank <= mrr_k
            else 0.0
        )
        reciprocal_ranks.append(reciprocal_rank)
        row["first_relevant_rank"] = first_relevant_rank
        row[f"reciprocal_rank_at_{mrr_k}"] = round(reciprocal_rank, 6)

        unmatched = [
            unit["label"]
            for unit in units
            if not any(_chunk_matches_evidence_unit(chunk, unit) for chunk in chunks)
        ]
        row["unmatched_evidence"] = unmatched
        unmatched_total += len(unmatched)
        per_question[question_id] = row

    values = {
        f"recall_at_{k}": (round(sum(rows) / len(rows), 6) if rows else None)
        for k, rows in recall_rows.items()
    }
    values["mrr_at_5"] = (
        round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6)
        if reciprocal_ranks
        else None
    )
    return {
        "answerable_questions": len(answerable),
        "evaluated_questions": len(per_question),
        "unmatched_evidence_count": unmatched_total,
        "values": values,
        "per_question": per_question,
    }


def build_report(
    questions: list,
    *,
    questions_path: Path,
    mode: str = "dry-run",
    retrieval_results: dict | None = None,
    generated_at: str | None = None,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> dict:
    validation = validate_questions(questions, duplicate_threshold=duplicate_threshold)
    values = {key: None for key in METRIC_KEYS}
    per_question = {}
    retrieval_summary = None

    if retrieval_results is not None:
        computed = compute_retrieval_metrics(questions, retrieval_results)
        values["recall_at_5"] = computed["values"]["recall_at_5"]
        values["recall_at_20"] = computed["values"]["recall_at_20"]
        values["mrr_at_5"] = computed["values"]["mrr_at_5"]
        per_question = computed["per_question"]
        retrieval_summary = {
            "answerable_questions": computed["answerable_questions"],
            "evaluated_questions": computed["evaluated_questions"],
            "unmatched_evidence_count": computed["unmatched_evidence_count"],
        }
        metric_status = "partial_offline_retrieval"
        metric_reason = "仅由调用方提供的离线检索结果计算 Recall@5/20 与 MRR@5；其余指标未运行。"
    else:
        metric_status = "not_run"
        metric_reason = "dry-run 不调用 embedding/LLM，也不伪造检索、答案或拒答分数。"

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "generated_at": generated_at or _utc_now(),
        "question_set": {
            **_display_question_set(questions_path),
            "sha256": question_set_sha256(questions),
            "count": len(questions),
        },
        "validation": validation,
        "metrics": {
            "status": metric_status,
            "reason": metric_reason,
            "values": values,
            "retrieval": retrieval_summary,
            "per_question": per_question,
            "definitions": METRIC_DEFINITIONS,
        },
        "baseline": {
            "status": "configuration_only",
            "note": (
                "legacy-character 是历史字符切片口径，不是当前生产默认值；"
                "当前生产默认口径见 current，token 预算路径（issue #287）。"
                "一阶段只记录配置，未运行真实 embedding/检索。"
            ),
            **CHARACTER_BASELINE,
            "current": CURRENT_TOKEN_BASELINE,
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    validation = report["validation"]
    metrics = report["metrics"]
    question_set = report["question_set"]
    source_label = "仓库内" if question_set["source"] == "repository" else "外部"
    question_label = question_set.get("relative_path", question_set["name"])

    lines = [
        "# RAG 评测 dry-run 报告",
        "",
        f"- 生成时间（UTC）：`{report['generated_at']}`",
        f"- 模式：`{report['mode']}`",
        f"- 问题集：`{question_label}`（{source_label}）",
        f"- 问题集 SHA-256：`{question_set['sha256']}`",
        f"- 校验结果：**{'通过' if validation['ok'] else '未通过'}**",
        f"- 问题总数：{validation['question_count']}",
        "",
        "## 分类计数",
        "",
        "| 分类 | 数量 |",
        "| --- | ---: |",
    ]
    for category, count in validation["category_counts"].items():
        lines.append(f"| {_md_cell(category)} | {count} |")
    lines.append(f"| 合计 | {validation['question_count']} |")

    lines += [
        "",
        "## 可回答分布",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| 可回答 | {validation['answerable']} |",
        f"| 不可回答 | {validation['unanswerable']} |",
        "",
        "## 校验明细",
        "",
    ]
    if validation["missing_fields"]:
        for item in validation["missing_fields"]:
            fields = "、".join(item["fields"])
            lines.append(f"- 缺失字段：`{item.get('id') or item['index']}` -> {fields}")
    else:
        lines.append("- 缺失字段：无")
    if validation["duplicate_ids"]:
        lines.append(f"- 重复 ID：{'、'.join(validation['duplicate_ids'])}")
    else:
        lines.append("- 重复 ID：无")
    if validation["warnings"]:
        for warning in validation["warnings"]:
            code = warning["code"]
            if code == "missing_required_category":
                lines.append(f"- 警告：缺少必备分类 `{warning['category']}`")
            elif code == "answerable_category_mismatch":
                lines.append(
                    "- 警告：`{}` 的 category={} 与 answerable={} 不一致".format(
                        warning.get("id"), warning.get("category"), warning.get("answerable")
                    )
                )
            elif code == "missing_evidence_mapping":
                lines.append(
                    f"- 警告：`{warning.get('id')}` 缺少 expected_doc/expected_sections，"
                    "Recall 将回退到 evidence 标签文本匹配"
                )
            elif code == "evidence_mapping_length_mismatch":
                lines.append(
                    f"- 警告：`{warning.get('id')}` 的 evidence 数量 "
                    f"({warning['evidence_count']}) 与 expected_sections 数量 "
                    f"({warning['expected_sections_count']}) 不一致"
                )
            elif code in ("near_duplicate_question", "near_duplicate_answer"):
                kind = "问题" if code == "near_duplicate_question" else "答案"
                lines.append(
                    f"- 警告：`{warning['ids'][0]}` 与 `{warning['ids'][1]}` 的{kind}近似重复"
                    f"（问题相似度 {warning['question_similarity']:.3f}，"
                    f"答案相似度 {warning['answer_similarity']:.3f}）"
                )
            else:
                lines.append(f"- 警告：`{warning.get('id')}` {warning.get('reason', code)}")
    else:
        lines.append("- 警告：无")
    if validation["errors"]:
        lines.append("")
        lines.append("### 错误")
        lines.append("")
        for error in validation["errors"]:
            if error["code"] == "missing_fields":
                lines.append(f"- `{error.get('id')}`：缺少字段 {', '.join(error['fields'])}")
            elif error["code"] == "duplicate_id":
                lines.append(f"- 重复 ID：`{error['id']}`")
            elif error["code"] == "invalid_entry":
                lines.append(f"- 第 {error['index']} 条：{error['reason']}")
            elif error["code"] == "empty_question_set":
                lines.append(f"- {error['reason']}")
            elif error["code"] == "exact_term_not_in_question":
                terms = "、".join(error.get("terms") or [])
                lines.append(
                    f"- `{error.get('id')}` 的问题未包含 exact_terms 字面词项：{terms}"
                )
            else:
                lines.append(
                    f"- `{error.get('id')}` 的 {error.get('field')}：{error.get('reason')}"
                )

    lines += [
        "",
        "## 指标",
        "",
        f"状态：`{metrics['status']}`。{metrics['reason']}",
        "",
        "| 指标 | 值 | 定义 |",
        "| --- | --- | --- |",
    ]
    for key in METRIC_KEYS:
        value = metrics["values"].get(key)
        rendered = "未运行" if value is None else value
        lines.append(
            f"| {_md_cell(key)} | {_md_cell(rendered)} | "
            f"{_md_cell(metrics['definitions'].get(key, ''))} |"
        )
    if metrics.get("retrieval") is not None:
        lines.append("")
        lines.append(
            "- 参与离线 Recall/MRR 计算：可答题 {answerable_questions} 道，实际评估 "
            "{evaluated_questions} 道".format(**metrics["retrieval"])
        )
        lines.append(
            "- 未匹配标准依据：{unmatched_evidence_count} 条".format(**metrics["retrieval"])
        )

    baseline = report["baseline"]
    current = baseline.get("current") or {}
    lines += [
        "",
        "## 基线口径",
        "",
        "### legacy-character（历史字符基线，非当前生产口径）",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| 名称 | `{_md_cell(baseline['name'])}` |",
        f"| 切片方法 | `{_md_cell(baseline['chunk_method'])}` |",
        f"| chunk_size | {baseline['chunk_size']} |",
        f"| chunk_overlap | {baseline['chunk_overlap']} |",
        f"| 源码位置 | `{_md_cell(baseline['source'])}` |",
        f"| 状态 | `{_md_cell(baseline['status'])}` |",
        "",
    ]
    if current:
        lines += [
            "### current-token（当前生产默认口径）",
            "",
            "| 字段 | 值 |",
            "| --- | --- |",
            f"| 名称 | `{_md_cell(current['name'])}` |",
            f"| 切片方法 | `{_md_cell(current['chunk_method'])}` |",
            f"| 计量单位 | `{_md_cell(current['budget_unit'])}` |",
            f"| CHUNK_TOKEN_LIMIT | {current['token_limit']} |",
            f"| CHUNK_TOKEN_TARGET | {current['token_target']} |",
            f"| CHUNK_TOKEN_OVERLAP | {current['token_overlap']} |",
            f"| CHUNK_TOKEN_SAFETY_RATIO | {current['token_safety_ratio']} |",
            f"| 源码位置 | `{_md_cell(current['source'])}` |",
            "",
        ]
    lines += [
        f"说明：{baseline['note']}",
        "",
        "指标公式详见 `docs/RAG_EVAL_BASELINE.md`。",
        "",
    ]
    return "\n".join(lines)


def _write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线、可复现的 RAG 评测基线（一阶段）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="离线校验问题集并输出报告；一阶段必须显式传入",
    )
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS),
        help="问题集 JSON 路径（默认仓库内小型夹具）",
    )
    parser.add_argument("--out", required=True, help="JSON 报告输出路径")
    parser.add_argument("--md-out", default=None, help="Markdown 报告路径（默认为 --out 同名 .md）")
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=DEFAULT_DUPLICATE_THRESHOLD,
        help=f"difflib 近似重复阈值（0-1，默认 {DEFAULT_DUPLICATE_THRESHOLD}）",
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        parser.error("一阶段只支持 --dry-run；真实 embedding/LLM 评测尚未接入")
    if not 0.0 < args.duplicate_threshold <= 1.0:
        parser.error("--duplicate-threshold 必须在 (0, 1] 范围内")

    questions_path = Path(args.questions)
    try:
        questions = load_questions(questions_path)
    except QuestionSetError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    report = build_report(
        questions,
        questions_path=questions_path,
        duplicate_threshold=args.duplicate_threshold,
    )
    out_path = Path(args.out)
    md_path = Path(args.md_out) if args.md_out else out_path.with_suffix(".md")
    try:
        _write_report(
            out_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        _write_report(md_path, render_markdown(report))
    except OSError as exc:
        print(f"错误：写入报告失败：{exc}", file=sys.stderr)
        return 2

    print(f"JSON 报告：{out_path}")
    print(f"Markdown 报告：{md_path}")
    if not report["validation"]["ok"]:
        print("问题集校验未通过，详见报告。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
