"""Offline RAG evaluation harness tests (issue #293)."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_rag.py"
FIXTURE = ROOT / "tests" / "fixtures" / "rag_eval_questions.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location("eval_rag_293", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_rag = _load_harness()


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _valid_question(question_id, evidence, category="single_fact", answerable=True):
    return {
        "id": question_id,
        "question": f"问题 {question_id}",
        "expected_answer": f"答案 {question_id}",
        "evidence": evidence,
        "category": category,
        "answerable": answerable,
    }


def test_fixture_covers_required_categories_and_exact_terms():
    questions = eval_rag.load_questions(FIXTURE)
    validation = eval_rag.validate_questions(questions)

    assert validation["ok"] is True
    assert validation["question_count"] == 32
    for category in eval_rag.REQUIRED_CATEGORIES:
        assert validation["category_counts"][category] == 2
    assert validation["category_counts"]["exact_term"] == 20
    assert validation["answerable"] == 30
    assert validation["unanswerable"] == 2
    assert validation["missing_required_categories"] == []
    assert validation["duplicate_ids"] == []


def test_missing_fields_and_duplicate_ids_are_reported():
    questions = [
        _valid_question("dup", ["S1"]),
        _valid_question("dup", ["S2"]),
        {
            "id": "bad",
            "question": "缺少标准答案",
            "evidence": [],
            "category": "single_fact",
            "answerable": "yes",
        },
    ]

    validation = eval_rag.validate_questions(questions)

    assert validation["ok"] is False
    assert {"index": 2, "id": "bad", "fields": ["expected_answer"]} in validation["missing_fields"]
    assert validation["duplicate_ids"] == ["dup"]
    codes = {error["code"] for error in validation["errors"]}
    assert {"missing_fields", "duplicate_id", "invalid_field"} <= codes


def test_exact_term_entry_requires_exact_terms_field():
    questions = [
        {
            "id": "e1",
            "question": "缺少 exact_terms 的精确词项题",
            "expected_answer": "应报告缺失字段",
            "evidence": ["S1"],
            "category": "exact_term",
            "answerable": True,
        }
    ]

    validation = eval_rag.validate_questions(questions)

    assert validation["ok"] is False
    assert {"index": 0, "id": "e1", "fields": ["exact_terms"]} in validation["missing_fields"]
    assert {
        "code": "missing_fields",
        "index": 0,
        "id": "e1",
        "fields": ["exact_terms"],
    } in validation["errors"]


def test_dry_run_writes_json_and_markdown_report(tmp_path):
    out = tmp_path / "eval-dry-run.json"

    code = eval_rag.main(
        ["--dry-run", "--questions", str(FIXTURE), "--out", str(out)]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    markdown = out.with_suffix(".md")
    assert markdown.exists()
    assert report["mode"] == "dry-run"
    assert report["validation"]["ok"] is True
    assert report["question_set"]["source"] == "repository"
    assert report["question_set"]["relative_path"] == "tests/fixtures/rag_eval_questions.json"
    assert report["metrics"]["status"] == "not_run"
    assert all(value is None for value in report["metrics"]["values"].values())
    assert report["baseline"]["chunk_size"] == 1200
    assert report["baseline"]["chunk_overlap"] == 150

    report_text = out.read_text(encoding="utf-8")
    markdown_text = markdown.read_text(encoding="utf-8")
    assert str(ROOT) not in report_text
    assert str(ROOT) not in markdown_text
    assert "RAG 评测 dry-run 报告" in markdown_text
    assert "未运行" in markdown_text


def test_dry_run_is_deterministic_except_timestamp(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert eval_rag.main(["--dry-run", "--questions", str(FIXTURE), "--out", str(first)]) == 0
    assert eval_rag.main(["--dry-run", "--questions", str(FIXTURE), "--out", str(second)]) == 0

    first_report = json.loads(first.read_text(encoding="utf-8"))
    second_report = json.loads(second.read_text(encoding="utf-8"))
    assert first_report.pop("generated_at")
    assert second_report.pop("generated_at")
    assert first_report == second_report

    def without_timestamp(text):
        return "\n".join(
            line for line in text.splitlines() if not line.startswith("- 生成时间")
        )

    assert without_timestamp(first.with_suffix(".md").read_text(encoding="utf-8")) == (
        without_timestamp(second.with_suffix(".md").read_text(encoding="utf-8"))
    )


def test_external_question_set_is_accepted_and_warns_about_missing_categories(tmp_path):
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps([_valid_question("x1", ["S1"])], ensure_ascii=False),
        encoding="utf-8",
    )
    out = tmp_path / "external-report.json"

    code = eval_rag.main(["--dry-run", "--questions", str(external), "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["validation"]["ok"] is True
    assert report["question_set"] == {
        "name": "external.json",
        "source": "external",
        "sha256": report["question_set"]["sha256"],
        "count": 1,
    }
    expected_missing = [
        category
        for category in eval_rag.REQUIRED_CATEGORIES
        if category != "single_fact"
    ]
    assert report["validation"]["missing_required_categories"] == expected_missing
    assert any(
        warning["code"] == "missing_required_category"
        for warning in report["validation"]["warnings"]
    )
    assert all(str(tmp_path) not in text for text in _walk_strings(report))


def test_invalid_question_set_exits_nonzero_and_still_writes_report(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "only-id"}]), encoding="utf-8")
    out = tmp_path / "bad-report.json"

    code = eval_rag.main(["--dry-run", "--questions", str(bad), "--out", str(out)])

    assert code == 1
    assert out.exists()
    assert out.with_suffix(".md").exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["validation"]["ok"] is False
    assert report["validation"]["missing_fields"]


def test_unreadable_question_set_exits_two(tmp_path):
    out = tmp_path / "missing-report.json"

    code = eval_rag.main(
        [
            "--dry-run",
            "--questions",
            str(tmp_path / "does-not-exist.json"),
            "--out",
            str(out),
        ]
    )

    assert code == 2
    assert not out.exists()


def test_requires_dry_run(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        eval_rag.main(["--questions", str(FIXTURE), "--out", str(tmp_path / "x.json")])
    assert excinfo.value.code == 2


def test_recall_metrics_from_supplied_results():
    questions = [
        _valid_question("a", ["S1", "S2"]),
        _valid_question("b", ["S3"]),
        _valid_question("c", [], category="unanswerable", answerable=False),
    ]
    retrieval_results = {
        "a": [{"evidence": ["S1"], "content": ""}]
        + [{"evidence": [], "content": "无关内容"} for _ in range(5)]
        + [{"evidence": ["S2"], "content": ""}],
        "b": [],
    }

    metrics = eval_rag.compute_retrieval_metrics(questions, retrieval_results)

    assert metrics["answerable_questions"] == 2
    assert metrics["evaluated_questions"] == 2
    assert metrics["values"]["recall_at_5"] == 0.25
    assert metrics["values"]["recall_at_20"] == 0.5
    assert metrics["per_question"]["a"]["recall_at_5"] == 0.5
    assert metrics["per_question"]["a"]["recall_at_20"] == 1.0
    assert metrics["per_question"]["b"]["recall_at_5"] == 0.0
    assert "c" not in metrics["per_question"]

    report = eval_rag.build_report(
        questions,
        questions_path=FIXTURE,
        retrieval_results=retrieval_results,
        generated_at="2026-01-01T00:00:00Z",
    )
    assert report["metrics"]["status"] == "partial_offline_retrieval"
    assert report["metrics"]["values"]["recall_at_5"] == 0.25
    assert report["metrics"]["values"]["recall_at_20"] == 0.5
    assert report["metrics"]["values"]["citation_hit"] is None


def test_character_baseline_matches_production_defaults():
    source = (ROOT / "backend" / "crud" / "knowledge_file.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    defaults = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "chunk_text":
            positional = node.args.args[-len(node.args.defaults):]
            defaults = {
                argument.arg: ast.literal_eval(default)
                for argument, default in zip(positional, node.args.defaults)
            }
            break

    assert defaults is not None
    assert defaults["chunk_size"] == eval_rag.CHARACTER_BASELINE["chunk_size"]
    assert defaults["chunk_overlap"] == eval_rag.CHARACTER_BASELINE["chunk_overlap"]
