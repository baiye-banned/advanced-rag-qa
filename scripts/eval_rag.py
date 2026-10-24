#!/usr/bin/env python
"""Offline, reproducible RAG evaluation harness (phase 1, issue #293).

Phase 1 is deliberately offline: ``--dry-run`` never imports the backend, never reads
credentials and never opens a socket. It validates a fixed question set and writes a
JSON + Markdown report whose metric slots are explicitly ``null`` / ``not_run``. No
score is fabricated; the report only proves that the dataset and the report structure
are sound, so phase 2 can plug real retrieval and generation runs into the same shape.

Question-set format (JSON array, UTF-8). Every entry has these fields:

    id               stable string, unique inside the file
    question         question text
    expected_answer  reference answer (for an unanswerable entry: the refusal criteria)
    evidence         section/clause labels that prove the answer; [] only when answerable=false
    category         single_fact | numeric_deadline | cross_section | table |
                     negation_exception | unanswerable | exact_term | ...
    answerable       boolean

``exact_term`` entries also carry ``exact_terms`` (a non-empty list of literal terms);
they are the 20 exact-term fixtures reserved for #288 (BM25). Any extra category name is
accepted, but the six required categories are reported when missing.

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

METRIC_KEYS = (
    "recall_at_5",
    "recall_at_20",
    "citation_hit",
    "answer_correctness",
    "refusal_correctness",
    "ingestion_latency_ms",
    "ingestion_failure_rate",
)

METRIC_DEFINITIONS = {
    "recall_at_5": "检索前 5 条中包含的标准依据数 / 该题标准依据总数，可答题按题取平均。",
    "recall_at_20": "检索前 20 条中包含的标准依据数 / 该题标准依据总数，可答题按题取平均。",
    "citation_hit": "答案引用的来源命中标准依据的可答题数 / 参与判定的可答题数。",
    "answer_correctness": "答案判定正确的可答题数 / 参与判定的可答题数。",
    "refusal_correctness": "正确拒答的不可答题数 / 参与判定的不可答题数。",
    "ingestion_latency_ms": "单文件入库耗时（毫秒）的平均值。",
    "ingestion_failure_rate": "入库失败文件数 / 入库文件总数。",
}

# Current production baseline: heading-aware character chunking in
# backend/crud/knowledge_file.py::chunk_text. Phase 1 records the configuration only;
# tests/test_eval_harness_293.py pins these values against the function's AST defaults.
CHARACTER_BASELINE = {
    "name": "character",
    "chunk_method": "chunk_text",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "source": "backend/crud/knowledge_file.py:chunk_text",
}


class QuestionSetError(ValueError):
    """The question file is unreadable or is not a JSON array."""


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_label(value) -> str:
    return " ".join(str(value or "").split()).casefold()


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


def validate_questions(questions: list) -> dict:
    """Validate the dataset and return a stable, report-ready summary."""
    errors = []
    warnings = []
    missing_fields = []
    duplicates = []
    seen_ids = set()
    category_counts = {}
    answerable_count = 0
    unanswerable_count = 0

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

    return {
        "ok": not errors,
        "question_count": len(questions),
        "errors": errors,
        "warnings": warnings,
        "missing_fields": missing_fields,
        "duplicate_ids": duplicates,
        "category_counts": dict(sorted(category_counts.items())),
        "missing_required_categories": missing_required_categories,
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
    }


def question_set_sha256(questions: list) -> str:
    canonical = json.dumps(questions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def compute_retrieval_metrics(questions: list, retrieval_results: dict, k_values=(5, 20)) -> dict:
    """Compute Recall@k from per-question retrieval results.

    ``retrieval_results`` maps a question id to an ordered list of chunks. A chunk is
    relevant when its ``evidence`` metadata equals one of the question's labels, or when
    the label appears in its ``content``. Only answerable questions with at least one
    expected label and a supplied result row are evaluated.
    """
    answerable = [q for q in questions if isinstance(q, dict) and q.get("answerable") is True]
    per_question = {}
    recall_rows = {k: [] for k in k_values}

    for question in answerable:
        question_id = question.get("id")
        if not _is_nonempty_str(question_id) or question_id not in retrieval_results:
            continue
        expected = [label for label in question.get("evidence") or [] if _is_nonempty_str(label)]
        if not expected:
            continue

        chunks = retrieval_results.get(question_id) or []
        if not isinstance(chunks, list):
            chunks = []

        row = {"expected_evidence": len(expected)}
        for k in k_values:
            top_k = [chunk for chunk in chunks[:k] if isinstance(chunk, dict)]
            matched = sum(
                1
                for label in expected
                if any(_chunk_matches_evidence(chunk, label) for chunk in top_k)
            )
            ratio = matched / len(expected)
            recall_rows[k].append(ratio)
            row[f"recall_at_{k}"] = round(ratio, 6)
        per_question[question_id] = row

    evaluated = len(recall_rows[k_values[0]]) if k_values else 0
    values = {
        f"recall_at_{k}": (round(sum(rows) / len(rows), 6) if rows else None)
        for k, rows in recall_rows.items()
    }
    return {
        "answerable_questions": len(answerable),
        "evaluated_questions": evaluated,
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
) -> dict:
    validation = validate_questions(questions)
    values = {key: None for key in METRIC_KEYS}
    per_question = {}
    retrieval_summary = None

    if retrieval_results is not None:
        computed = compute_retrieval_metrics(questions, retrieval_results)
        values["recall_at_5"] = computed["values"]["recall_at_5"]
        values["recall_at_20"] = computed["values"]["recall_at_20"]
        per_question = computed["per_question"]
        retrieval_summary = {
            "answerable_questions": computed["answerable_questions"],
            "evaluated_questions": computed["evaluated_questions"],
        }
        metric_status = "partial_offline_retrieval"
        metric_reason = "仅由调用方提供的离线检索结果计算 Recall@5/20；其余指标未运行。"
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
            "note": "一阶段只记录当前字符切片配置，未运行真实 embedding/检索。",
            **CHARACTER_BASELINE,
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
            if warning["code"] == "missing_required_category":
                lines.append(f"- 警告：缺少必备分类 `{warning['category']}`")
            else:
                lines.append(
                    "- 警告：`{}` 的 category={} 与 answerable={} 不一致".format(
                        warning.get("id"), warning.get("category"), warning.get("answerable")
                    )
                )
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
            "- 参与离线 Recall 计算：可答题 {answerable_questions} 道，实际评估 "
            "{evaluated_questions} 道".format(**metrics["retrieval"])
        )

    baseline = report["baseline"]
    lines += [
        "",
        "## 字符切片基线",
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
    args = parser.parse_args(argv)

    if not args.dry_run:
        parser.error("一阶段只支持 --dry-run；真实 embedding/LLM 评测尚未接入")

    questions_path = Path(args.questions)
    try:
        questions = load_questions(questions_path)
    except QuestionSetError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    report = build_report(questions, questions_path=questions_path)
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
