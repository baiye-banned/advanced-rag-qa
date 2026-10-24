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


def _valid_question(
    question_id,
    evidence,
    category="single_fact",
    answerable=True,
    *,
    question=None,
    expected_answer=None,
    expected_doc=None,
    expected_sections=None,
):
    entry = {
        "id": question_id,
        "question": question if question is not None else f"问题 {question_id}",
        "expected_answer": (
            expected_answer if expected_answer is not None else f"答案 {question_id}"
        ),
        "evidence": evidence,
        "category": category,
        "answerable": answerable,
    }
    if expected_doc is not None:
        entry["expected_doc"] = expected_doc
    if expected_sections is not None:
        entry["expected_sections"] = expected_sections
    return entry


def _mapped_question(question_id, *, doc="员工休假管理办法.md", section="第2章 第1条"):
    return _valid_question(
        question_id,
        [f"《{doc[:-3]}》{section}"],
        expected_doc=doc,
        expected_sections=[section],
    )


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
    assert validation["near_duplicates"] == []
    assert validation["warnings"] == []


def test_fixture_exact_term_questions_literally_contain_a_term():
    questions = eval_rag.load_questions(FIXTURE)
    exact_term_questions = [q for q in questions if q["category"] == "exact_term"]

    assert len(exact_term_questions) == 20
    for question in exact_term_questions:
        normalized_question = eval_rag._normalize_label(question["question"])
        assert any(
            eval_rag._normalize_label(term) in normalized_question
            for term in question["exact_terms"]
        ), question["id"]


def test_fixture_answerable_questions_have_executable_evidence_mapping():
    questions = eval_rag.load_questions(FIXTURE)

    for question in questions:
        if question.get("answerable") is not True:
            continue
        assert question.get("expected_doc"), question["id"]
        assert question.get("expected_sections"), question["id"]
        docs = question["expected_doc"]
        if isinstance(docs, list):
            assert len(docs) == len(question["expected_sections"])
        assert len(question["expected_sections"]) == len(question["evidence"])


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


def test_exact_term_question_must_contain_literal_term():
    questions = [
        _valid_question(
            "e2",
            ["S1"],
            category="exact_term",
            question="这条问题不含任何编号",
            expected_doc="doc.md",
            expected_sections=["S1"],
        )
    ]
    questions[0]["exact_terms"] = ["ABC-1"]

    validation = eval_rag.validate_questions(questions)

    assert validation["ok"] is False
    assert any(
        error["code"] == "exact_term_not_in_question" and error["id"] == "e2"
        for error in validation["errors"]
    )


def test_empty_question_set_exits_nonzero_and_reports_error(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    out = tmp_path / "empty-report.json"

    code = eval_rag.main(["--dry-run", "--questions", str(empty), "--out", str(out)])

    assert code == 1
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["validation"]["ok"] is False
    assert report["validation"]["question_count"] == 0
    assert any(
        error["code"] == "empty_question_set" for error in report["validation"]["errors"]
    )
    assert all(value is None for value in report["metrics"]["values"].values())
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "问题集至少需要一条问题" in markdown


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
    assert report["metrics"]["values"]["mrr_at_5"] is None
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
    warning_codes = {warning["code"] for warning in report["validation"]["warnings"]}
    assert "missing_required_category" in warning_codes
    assert "missing_evidence_mapping" in warning_codes
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


def test_markdown_reports_missing_field_details(tmp_path):
    bad = tmp_path / "bad-fields.json"
    bad.write_text(
        json.dumps(
            [
                {
                    "id": "bad",
                    "question": "缺少标准答案",
                    "evidence": [],
                    "category": "single_fact",
                    "answerable": True,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "bad-fields-report.json"

    assert eval_rag.main(["--dry-run", "--questions", str(bad), "--out", str(out)]) == 1
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "- 缺失字段：`bad` -> expected_answer" in markdown
    assert "缺少字段 expected_answer" in markdown


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


def test_invalid_duplicate_threshold_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        eval_rag.main(
            [
                "--dry-run",
                "--questions",
                str(FIXTURE),
                "--out",
                str(tmp_path / "x.json"),
                "--duplicate-threshold",
                "1.5",
            ]
        )
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
    assert metrics["values"]["mrr_at_5"] == 0.5
    assert metrics["per_question"]["a"]["recall_at_5"] == 0.5
    assert metrics["per_question"]["a"]["recall_at_20"] == 1.0
    assert metrics["per_question"]["a"]["first_relevant_rank"] == 1
    assert metrics["per_question"]["b"]["recall_at_5"] == 0.0
    assert metrics["per_question"]["b"]["first_relevant_rank"] is None
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
    assert report["metrics"]["values"]["mrr_at_5"] == 0.5
    assert report["metrics"]["values"]["citation_hit"] is None


def test_recall_uses_file_name_and_section_with_production_chunk_shape():
    questions = [_mapped_question("p1")]
    retrieval_results = {
        "p1": [
            {
                "chunk_id": "0",
                "file_name": "费用报销管理办法.md",
                "content": "第2章 第1条 这是另一份文档里的同号章节",
            },
            {
                "chunk_id": "1",
                "file_name": "员工休假管理办法.md",
                "content": "第2章 第1条 入职满一年后每年享有 5 天年假",
            },
        ]
    }

    metrics = eval_rag.compute_retrieval_metrics(questions, retrieval_results)

    assert all(
        set(chunk) == {"chunk_id", "file_name", "content"}
        for chunk in retrieval_results["p1"]
    )
    assert metrics["values"]["recall_at_5"] == 1.0
    assert metrics["values"]["mrr_at_5"] == 0.5
    assert metrics["per_question"]["p1"]["first_relevant_rank"] == 2
    assert metrics["per_question"]["p1"]["reciprocal_rank_at_5"] == 0.5
    assert metrics["unmatched_evidence_count"] == 0


def test_recall_falls_back_to_content_only_chunks_without_evidence_metadata():
    questions = [_mapped_question("p1")]
    retrieval_results = {
        "p1": [
            {
                "chunk_id": "0",
                "content": "第2章 第1条 入职满一年后每年享有 5 天年假",
            }
        ]
    }

    metrics = eval_rag.compute_retrieval_metrics(questions, retrieval_results)

    assert "evidence" not in retrieval_results["p1"][0]
    assert "file_name" not in retrieval_results["p1"][0]
    assert metrics["values"]["recall_at_5"] == 1.0
    assert metrics["values"]["mrr_at_5"] == 1.0
    assert metrics["per_question"]["p1"]["first_relevant_rank"] == 1
    assert metrics["unmatched_evidence_count"] == 0


def _chunks_with_relevant_at(rank, total=6):
    chunks = [
        {"chunk_id": str(index), "file_name": "doc.md", "content": "无关内容"}
        for index in range(total)
    ]
    if rank is not None:
        chunks[rank - 1]["content"] = "第1条 相关内容"
    return chunks


def test_mrr_at_5_with_ranks_one_three_six_and_missing():
    questions = [
        _mapped_question("r1", doc="doc.md", section="第1条"),
        _mapped_question("r2", doc="doc.md", section="第1条"),
        _mapped_question("r3", doc="doc.md", section="第1条"),
        _mapped_question("r4", doc="doc.md", section="第1条"),
    ]
    retrieval_results = {
        "r1": _chunks_with_relevant_at(1),
        "r2": _chunks_with_relevant_at(3),
        "r3": _chunks_with_relevant_at(6),
        "r4": _chunks_with_relevant_at(None),
    }

    metrics = eval_rag.compute_retrieval_metrics(questions, retrieval_results)

    assert metrics["values"]["recall_at_5"] == 0.5
    assert metrics["values"]["recall_at_20"] == 0.75
    assert metrics["values"]["mrr_at_5"] == 0.333333
    assert metrics["per_question"]["r1"]["first_relevant_rank"] == 1
    assert metrics["per_question"]["r2"]["first_relevant_rank"] == 3
    assert metrics["per_question"]["r3"]["first_relevant_rank"] == 6
    assert metrics["per_question"]["r3"]["reciprocal_rank_at_5"] == 0.0
    assert metrics["per_question"]["r4"]["first_relevant_rank"] is None
    assert metrics["per_question"]["r4"]["unmatched_evidence"] == ["《doc》第1条"]
    assert metrics["unmatched_evidence_count"] == 1


def test_retrieval_report_reports_unmatched_evidence_count():
    questions = [_mapped_question("u1")]
    retrieval_results = {
        "u1": [
            {
                "chunk_id": "0",
                "file_name": "other.md",
                "content": "第2章 第1条 错误文档",
            }
        ]
    }

    report = eval_rag.build_report(
        questions,
        questions_path=FIXTURE,
        retrieval_results=retrieval_results,
        generated_at="2026-01-01T00:00:00Z",
    )

    assert report["metrics"]["retrieval"]["unmatched_evidence_count"] == 1
    assert report["metrics"]["per_question"]["u1"]["unmatched_evidence"] == [
        "《员工休假管理办法》第2章 第1条"
    ]
    markdown = eval_rag.render_markdown(report)
    assert "未匹配标准依据：1 条" in markdown


def test_near_duplicate_questions_and_answers_are_reported_with_configurable_threshold():
    questions = [
        _valid_question(
            "n1",
            ["S1"],
            question="员工年假有几天？",
            expected_answer="每年 5 天。",
        ),
        _valid_question(
            "n2",
            ["S2"],
            question="员工年假有几天啊？",
            expected_answer="每年 5 天。",
        ),
    ]

    strict = eval_rag.validate_questions(questions, duplicate_threshold=0.99)
    loose = eval_rag.validate_questions(questions, duplicate_threshold=0.9)

    assert [pair["kind"] for pair in strict["near_duplicates"]] == ["answer"]
    assert [pair["kind"] for pair in loose["near_duplicates"]] == ["question"]
    assert strict["near_duplicates"][0]["ids"] == ["n1", "n2"]
    assert strict["duplicate_threshold"] == 0.99
    warning_codes = {warning["code"] for warning in strict["warnings"]}
    assert "near_duplicate_answer" in warning_codes


def test_character_baseline_is_an_explicit_legacy_path():
    """issue #287 之后生产默认走 token 路径，字符基线改为显式传参。

    ``CHARACTER_BASELINE`` 仍用于对比历史口径，但它不再等于 ``chunk_text`` 的默认值：
    只有显式传入 ``chunk_size`` / ``chunk_overlap`` 时才走字符窗口。这里钉住两件事：
    默认值保持 None（否则服务端会静默退回字符路径），以及基线配置仍是显式的 1200/150。
    """
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
    assert defaults["chunk_size"] is None
    assert defaults["chunk_overlap"] is None
    assert eval_rag.CHARACTER_BASELINE["chunk_size"] == 1200
    assert eval_rag.CHARACTER_BASELINE["chunk_overlap"] == 150


def test_current_token_baseline_matches_config_defaults():
    """``CURRENT_TOKEN_BASELINE`` 必须与 config 的默认值一致，不能各写各的。

    评估脚本刻意不 import backend（离线运行），所以这里用 AST 读取 config.py 的
    ``DEFAULT_CHUNK_TOKEN_*`` 常量与 ``_env_int`` / ``_env_float`` 的字面量默认值；
    任何一处漂移都会让报告里的「current-token」基线不再是生产默认口径。
    """
    tree = ast.parse((ROOT / "backend" / "config.py").read_text(encoding="utf-8"))

    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("DEFAULT_CHUNK_TOKEN_")
        ):
            constants[node.targets[0].id] = ast.literal_eval(node.value)

    assert constants == {
        "DEFAULT_CHUNK_TOKEN_LIMIT": 500,
        "DEFAULT_CHUNK_TOKEN_TARGET": 400,
        "DEFAULT_CHUNK_TOKEN_OVERLAP": 50,
    }

    env_defaults = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_env_int", "_env_float"}
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            continue
        env_defaults[node.args[0].value] = node.args[1].value

    assert env_defaults["CHUNK_TOKEN_LIMIT"] == constants["DEFAULT_CHUNK_TOKEN_LIMIT"]
    assert env_defaults["CHUNK_TOKEN_TARGET"] == constants["DEFAULT_CHUNK_TOKEN_TARGET"]
    assert env_defaults["CHUNK_TOKEN_OVERLAP"] == constants["DEFAULT_CHUNK_TOKEN_OVERLAP"]

    baseline = eval_rag.CURRENT_TOKEN_BASELINE
    assert baseline["token_limit"] == env_defaults["CHUNK_TOKEN_LIMIT"]
    assert baseline["token_target"] == env_defaults["CHUNK_TOKEN_TARGET"]
    assert baseline["token_overlap"] == env_defaults["CHUNK_TOKEN_OVERLAP"]
    assert baseline["token_safety_ratio"] == env_defaults["CHUNK_TOKEN_SAFETY_RATIO"]
