"""issue #289：父子切片与上下文前缀。

覆盖：
- child 召回后按 ``parent_id`` 聚合，同一 parent 只占一个最终上下文槽位；
- 跨章节不误聚合，长条款的多个 child 能合并回同一 parent；
- 旧数据（无元数据行 / 无 parent_id）回退 child 内容并留下 trace；
- 删除文件时 child 与 parent 元数据一起清理；
- ``PARENT_CHILD_ENABLED=false`` 回到 #287 的单层切片；
- 上下文前缀是确定性的，并计入与 #287 相同的 tiktoken token 预算。
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config import CHUNK_TOKEN_LIMIT, CHUNK_TOKEN_SAFETY_RATIO
from crud import knowledge_file as crud_knowledge_file
from crud.knowledge_file import (
    MIN_CHILD_TOKEN_COUNT,
    _context_prefix,
    _parent_text_for_group,
    build_parent_child_chunks,
    chunk_text,
    count_chunk_tokens,
    resolve_chunk_token_budget,
)
from database.session import Base
from model.models import KnowledgeChunkMetadata, KnowledgeFile, User
from rag import retrieval
from service import knowledge_service
from service.utils_service import _build_sources


INSTITUTION_TEXT = """第一章 总则
第一条
为规范公司考勤管理，特制定本制度。
第二条
本制度适用于全体员工，由人事部负责解释。

第二章 考勤管理
第三条
员工应按时上下班，不得迟到早退。
第四条
员工迟到三十分钟以内记口头提醒；超过三十分钟按旷工半天处理。
"""

# 单个长条款：同一条款被切成多个 base chunk，父级必须把它们聚合回一个 parent。
LONG_ARTICLE_TEXT = "第五条\n" + "".join(
    f"第{index}项规定：员工应当遵守考勤、请假和信息安全要求，由人事部负责解释。"
    for index in range(1, 61)
)


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


@pytest.fixture(autouse=True)
def _enable_parent_child(monkeypatch):
    """本文件验证父子切片本身；#289 的开关默认关闭由 config/部署决定。"""
    monkeypatch.setattr(retrieval, "PARENT_CHILD_ENABLED", True)
    monkeypatch.setattr(knowledge_service, "PARENT_CHILD_ENABLED", True)


@pytest.fixture()
def metadata_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        bind=engine,
        tables=[User.__table__, KnowledgeFile.__table__, KnowledgeChunkMetadata.__table__],
    )
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()
    try:
        yield SimpleNamespace(db=db, engine=engine)
    finally:
        db.close()


def _hard_limit() -> int:
    return max(1, int(CHUNK_TOKEN_LIMIT * CHUNK_TOKEN_SAFETY_RATIO))


def _create_file(db, *, name="考勤制度.txt", content=INSTITUTION_TEXT, user_id=7) -> KnowledgeFile:
    entry = KnowledgeFile(
        knowledge_base_id=5,
        name=name,
        size=len(content),
        content=content,
        user_id=user_id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def _stored_rows(db, file_id):
    return (
        db.query(KnowledgeChunkMetadata)
        .filter_by(file_id=file_id)
        .order_by(KnowledgeChunkMetadata.id.asc())
        .all()
    )


def _chunk_payload(child: dict, *, file_id: int) -> dict:
    """把入库 child 变成检索侧候选的形状（Milvus 的 output fields）。"""
    return {
        "id": child["id"],
        "chunk_id": child["id"],
        "file_id": file_id,
        "file_name": "考勤制度.txt",
        "content": child["text"],
        "route": "planned",
    }


def test_same_parent_children_collapse_into_one_context_slot(metadata_db):
    db = metadata_db.db
    entry = _create_file(db, name="长条款.txt", content=LONG_ARTICLE_TEXT)
    children, parents = build_parent_child_chunks(
        chunk_text(LONG_ARTICLE_TEXT, entry.id), file_name=entry.name
    )
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, [*children, *parents])

    same_parent = [child for child in children if child["metadata"]["parent_id"] == "p:0"]
    assert len(same_parent) > 1
    candidates = [_chunk_payload(child, file_id=entry.id) for child in same_parent[:2]]

    expanded, stats = retrieval._expand_parent_chunks_sync(db, candidates)

    assert len(expanded) == 1
    assert stats["expanded"] == 1
    assert stats["merged"] == 1
    assert expanded[0]["parent_id"] == "p:0"
    assert expanded[0]["parent_text"] == parents[0]["text"]
    assert expanded[0]["content"] == parents[0]["text"]
    assert expanded[0]["parent_expanded"] is True
    assert expanded[0]["chunk_id"] == candidates[0]["chunk_id"]


def test_retrieve_knowledge_returns_the_parent_once_for_multiple_child_hits(
    metadata_db, monkeypatch
):
    db = metadata_db.db
    entry = _create_file(db, name="长条款.txt", content=LONG_ARTICLE_TEXT)
    children, parents = build_parent_child_chunks(
        chunk_text(LONG_ARTICLE_TEXT, entry.id), file_name=entry.name
    )
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, [*children, *parents])
    hits = [_chunk_payload(child, file_id=entry.id) for child in children[:2]]

    def fake_query_vectors(query, top_k, knowledge_base_id, route):
        return hits if route == "planned" else []

    async def fake_rerank(question, chunks):
        return chunks, {"status": "skipped", "items": []}

    monkeypatch.setattr(retrieval, "query_vectors", fake_query_vectors)
    monkeypatch.setattr(retrieval, "keyword_recall", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "rerank_chunks", fake_rerank)

    final, trace = asyncio.run(
        retrieval.retrieve_knowledge(
            "长条款怎么规定",
            knowledge_base_id=5,
            db=db,
            query_plan={
                "original_question": "长条款怎么规定",
                "simplified_question": "长条款",
                "sub_questions": [],
                "rewrites": [],
                "keywords": [],
            },
        )
    )

    assert len(final) == 1
    assert final[0]["content"] == parents[0]["text"]
    assert final[0]["parent_id"] == "p:0"
    assert trace["parent_child"]["status"] == "done"
    assert trace["parent_child"]["expanded"] == 1
    assert trace["parent_child"]["merged"] == 1


def test_cross_chapter_chunks_are_not_merged_and_long_article_is():
    children, parents = build_parent_child_chunks(
        chunk_text(INSTITUTION_TEXT, 1), file_name="考勤制度.txt"
    )

    parents_by_id = {parent["id"]: parent for parent in parents}
    chapter_one = next(
        child for child in children if "第一章" in child["metadata"]["section_path"]
    )
    chapter_two = next(
        child for child in children if "第二章" in child["metadata"]["section_path"]
    )
    assert chapter_one["metadata"]["parent_id"] != chapter_two["metadata"]["parent_id"]
    assert "第二章" not in parents_by_id[chapter_one["metadata"]["parent_id"]]["text"]
    assert "第一章" not in parents_by_id[chapter_two["metadata"]["parent_id"]]["text"]

    long_children, long_parents = build_parent_child_chunks(
        chunk_text(LONG_ARTICLE_TEXT, 1), file_name="长条款.txt"
    )
    assert len(long_parents) == 1
    assert {child["metadata"]["parent_id"] for child in long_children} == {"p:0"}
    assert len(long_children) > len(long_parents)


def test_unstructured_long_text_gets_bounded_parents():
    """没有章节/条款边界的纯文本不能整篇聚成一个巨型 parent。"""
    target, hard_limit, _overlap = resolve_chunk_token_budget()
    parent_cap = max(hard_limit, target * 8)

    children, parents = build_parent_child_chunks(
        chunk_text("a" * 200_000, 1), file_name="plain.txt"
    )

    assert len(parents) > 1
    assert len(children) > len(parents)
    assert all(parent["metadata"]["token_count"] <= parent_cap for parent in parents)
    assert {child["metadata"]["parent_id"] for child in children} == {
        parent["id"] for parent in parents
    }


def test_legacy_data_falls_back_to_child_content(metadata_db):
    db = metadata_db.db
    entry = _create_file(db)
    child = {
        "id": "0",
        "chunk_id": "0",
        "file_id": entry.id,
        "file_name": entry.name,
        "content": "旧文件的单层切片正文",
        "route": "planned",
    }

    # 完全没有元数据行：模拟 #287 之前上传、没有重建的旧文件。
    expanded, stats = retrieval._expand_parent_chunks_sync(db, [child])
    assert expanded == [child]
    assert stats["fallback"] == 1
    assert stats["expanded"] == 0

    # 有 #287 行但 parent_id 为空：同样按 child 内容降级。
    crud_knowledge_file.replace_chunk_metadata(
        db,
        entry.id,
        5,
        [{"id": "0", "text": child["content"], "metadata": {"token_count": 10}}],
    )
    expanded, stats = retrieval._expand_parent_chunks_sync(db, [child])
    assert expanded == [child]
    assert stats["mapped"] == 1
    assert stats["fallback"] == 1
    assert stats["expanded"] == 0


def test_asyncio_expansion_never_raises_when_metadata_is_unavailable(monkeypatch):
    child = {
        "id": "0",
        "chunk_id": "0",
        "file_id": 1,
        "file_name": "旧文件.txt",
        "content": "旧正文",
        "route": "planned",
    }

    def unavailable_session():
        raise RuntimeError("database is unavailable")

    monkeypatch.setattr(retrieval, "SessionLocal", unavailable_session)

    expanded, trace = asyncio.run(retrieval.expand_parent_chunks([child], db=None))

    assert expanded == [child]
    assert trace["status"] == "fallback"
    assert trace["reason"] == "metadata_unavailable"


def test_deleting_a_file_cleans_child_and_parent_metadata(metadata_db):
    db = metadata_db.db
    entry = _create_file(db, name="长条款.txt", content=LONG_ARTICLE_TEXT)
    children, parents = build_parent_child_chunks(
        chunk_text(LONG_ARTICLE_TEXT, entry.id), file_name=entry.name
    )
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, [*children, *parents])

    rows = _stored_rows(db, entry.id)
    assert {row.chunk_id for row in rows} == {
        *(child["id"] for child in children),
        *(parent["id"] for parent in parents),
    }
    assert any(row.chunk_id.startswith("p:") for row in rows)

    assert crud_knowledge_file.delete_knowledge_file(db, entry.id, 7) is not None
    assert db.query(KnowledgeFile).filter_by(id=entry.id).count() == 0
    assert _stored_rows(db, entry.id) == []


def test_parent_child_disabled_keeps_single_layer_semantics(metadata_db, monkeypatch):
    db = metadata_db.db
    entry = _create_file(db, name="长条款.txt", content=LONG_ARTICLE_TEXT)
    base_chunks = chunk_text(LONG_ARTICLE_TEXT, entry.id)

    monkeypatch.setattr(knowledge_service, "PARENT_CHILD_ENABLED", False)
    index_chunks, metadata_chunks = knowledge_service._chunk_for_index(
        LONG_ARTICLE_TEXT, entry.id, entry.name
    )

    assert index_chunks == base_chunks
    assert metadata_chunks == base_chunks
    assert all(chunk["metadata"]["parent_id"] is None for chunk in index_chunks)

    # 检索侧的开关关闭时同样原样返回 child 内容。
    monkeypatch.setattr(retrieval, "PARENT_CHILD_ENABLED", False)
    candidates = [_chunk_payload(chunk, file_id=entry.id) for chunk in base_chunks[:2]]
    expanded, trace = asyncio.run(retrieval.expand_parent_chunks(candidates, db=db))
    assert expanded == candidates
    assert trace["status"] == "disabled"


def test_context_prefix_is_deterministic_and_budgeted():
    target, hard_limit, _overlap = resolve_chunk_token_budget()
    assert target <= hard_limit == _hard_limit()

    base_chunks = chunk_text(LONG_ARTICLE_TEXT, 1)
    first_children, _ = build_parent_child_chunks(base_chunks, file_name="考勤制度.txt")
    second_children, _ = build_parent_child_chunks(base_chunks, file_name="考勤制度.txt")

    assert [child["text"] for child in first_children] == [
        child["text"] for child in second_children
    ]
    for child in first_children:
        prefix = child["metadata"]["context_prefix"]
        assert prefix.startswith("【文档：考勤制度.txt")
        assert child["text"].startswith(prefix)
        assert count_chunk_tokens(prefix) > 0
        # 前缀与正文一起受 #287 的同一硬上限约束（不是目标值）。
        assert count_chunk_tokens(child["text"]) <= hard_limit
        assert child["metadata"]["token_count"] == count_chunk_tokens(child["text"])

    # 病态超长文件名不能让前缀吃掉正文预算。
    long_name = "超长文件名" * 40 + ".txt"
    long_children, _ = build_parent_child_chunks(base_chunks, file_name=long_name)
    assert long_children
    assert max(count_chunk_tokens(child["text"]) for child in long_children) <= _hard_limit()
    assert all(child["text"].strip() for child in long_children)
    # 前缀被文件名截断、章/条标签没进前缀时，正文里的标题必须保留，不能两边都丢。
    long_child = next(
        child for child in long_children if child["metadata"]["section_path"] == "第五条"
    )
    body_after_prefix = long_child["text"][
        len(long_child["metadata"]["context_prefix"]):
    ].lstrip("\n")
    assert body_after_prefix.startswith("第五条")


def test_base_chunk_near_target_with_prefix_stays_one_child():
    """接近 target 的切片加上前缀后仍在硬上限内，不能被误切成两片。"""
    target, hard_limit, _overlap = resolve_chunk_token_budget()
    metadata = {
        "section_path": "第一章 总则 > 第一条",
        "article_no": "第一条",
    }
    prefix = _context_prefix("考勤制度.txt", metadata, max_tokens=hard_limit // 4)
    prefix_tokens = count_chunk_tokens(prefix)
    # 短句让正文能贴近上限，覆盖「旧 target 预算会切、新硬上限不切」的区间。
    sentence = "员工应当遵守考勤要求。"
    body_lines: list[str] = []
    # 留出前缀和换行后仍不超过硬上限，但已经超过「target - 前缀」的旧预算。
    while count_chunk_tokens("\n".join([*body_lines, sentence])) <= hard_limit - prefix_tokens - 8:
        body_lines.append(sentence)
    body = "\n".join(body_lines)
    assert count_chunk_tokens(body) > target - prefix_tokens
    metadata["token_count"] = count_chunk_tokens(body)
    base = {
        "id": "0",
        "text": f"第一章 总则\n第一条\n{body}",
        "metadata": metadata,
    }

    children, _ = build_parent_child_chunks([base], file_name="考勤制度.txt")

    assert len(children) == 1
    assert children[0]["id"] == "0"
    assert count_chunk_tokens(children[0]["text"]) <= hard_limit
    # 旧实现用 target 做正文预算，正文超过 target - 前缀 就会被切成两片；现在保持
    # 1 个 child，因为完整文本仍在硬上限内。
    assert count_chunk_tokens(body) > target - prefix_tokens
    # 正文开头的重复章节/条款标题已被剥掉，只剩前缀里那一份。
    assert children[0]["text"].count("第一章 总则") == 1
    body_after_prefix = children[0]["text"][len(prefix):].lstrip("\n")
    assert not body_after_prefix.startswith("第一章 总则")
    assert not body_after_prefix.startswith("第一条")
    assert children[0]["metadata"]["context_prefix"].endswith("】")


def test_prefix_deduplication_does_not_double_child_count():
    base_chunks = chunk_text(INSTITUTION_TEXT, 1)
    children, parents = build_parent_child_chunks(
        base_chunks, file_name="考勤制度.txt"
    )

    assert len(children) == len(base_chunks)
    assert len(parents) >= 1
    for base, child in zip(base_chunks, children):
        assert child["id"] == base["id"]
        prefix = child["metadata"]["context_prefix"]
        body_after_prefix = child["text"][len(prefix):].lstrip("\n")
        section = str(base["metadata"].get("section_path") or "")
        for heading in section.split(" > "):
            if heading.strip():
                assert not body_after_prefix.startswith(heading.strip())


def test_split_children_never_produce_tiny_fragments():
    target, hard_limit, _overlap = resolve_chunk_token_budget()
    sentence = "员工应当遵守考勤、请假和信息安全要求，由人事部负责解释。"
    # 刚好超过硬上限：按行/句贪心切分很容易留下十几 token 的尾巴。
    body_lines: list[str] = []
    while count_chunk_tokens("\n".join(body_lines)) <= hard_limit + 20:
        body_lines.append(sentence)
    body = "\n".join(body_lines)
    assert count_chunk_tokens(body) > hard_limit

    base = {
        "id": "0",
        "text": body,
        "metadata": {"section_path": "", "article_no": "", "token_count": count_chunk_tokens(body)},
    }
    children, _ = build_parent_child_chunks([base], file_name="制度.txt")

    assert len(children) > 1
    assert all(
        count_chunk_tokens(child["text"]) >= MIN_CHILD_TOKEN_COUNT for child in children
    )
    assert all(count_chunk_tokens(child["text"]) <= hard_limit for child in children)

    # 更长的正文同样不允许出现碎片。
    long_base = {
        "id": "1",
        "text": "\n".join([sentence] * 120),
        "metadata": {"section_path": "", "article_no": "", "token_count": 0},
    }
    long_children, _ = build_parent_child_chunks([long_base], file_name="制度.txt")
    assert len(long_children) > 1
    assert all(
        count_chunk_tokens(child["text"]) >= MIN_CHILD_TOKEN_COUNT
        for child in long_children
    )


def _parent_lines(*, source: str, file_name: str = "制度.txt") -> tuple[list[dict], str]:
    children, parents = build_parent_child_chunks(
        chunk_text(source, 1), file_name=file_name
    )
    return children, "\n".join(parent["text"] for parent in parents)


def test_parent_text_strips_chunk_overlap_and_keeps_sentence_count():
    children, parents = build_parent_child_chunks(
        chunk_text(LONG_ARTICLE_TEXT, 1), file_name="长条款.txt"
    )

    assert len(parents) == 1
    parent_text = parents[0]["text"]
    sentences = [item for item in re.split(r"(?<=。)", parent_text) if item.strip()]
    assert len(sentences) == 60
    # 每项规定在 parent 里恰好出现一次；#287 的重叠若未剥掉，切片边界会重复。
    for index in range(1, 61):
        assert parent_text.count(f"第{index}项规定") == 1
    # parent 仍保留一次标题，且 child 数量大于 parent（长条款被切成多片）。
    assert parent_text.count("第五条") == 1
    assert len(children) > len(parents)


def test_metadata_records_only_the_overlap_actually_carried_forward():
    """``overlap_text`` 必须精确对应本片开头真正来自上一片的正文。

    旧的「上一片末尾 == 当前片开头」比较没有记录来源，只能靠位置猜；这里要求
    metadata 里的字段与本片正文严格一致，parent 拼接才有可信依据。
    """
    chunks = chunk_text(LONG_ARTICLE_TEXT, 1)

    assert all("overlap_text" in chunk["metadata"] for chunk in chunks)
    assert chunks[0]["metadata"]["overlap_text"] == ""
    assert any(chunk["metadata"]["overlap_text"] for chunk in chunks)
    for chunk in chunks:
        overlap_text = chunk["metadata"]["overlap_text"]
        if not overlap_text:
            continue
        body = chunk["text"]
        heading_prefix = "\n".join(
            part.strip()
            for part in str(chunk["metadata"].get("section_path") or "").split(" > ")
            if part.strip()
        )
        if heading_prefix and body.startswith(heading_prefix):
            body = body[len(heading_prefix):].lstrip("\n")
        assert body.startswith(overlap_text)


def test_parent_keeps_every_identical_short_line():
    """60 行相同短句：原文本来就重复，parent 必须保留 60 句（旧实现只剩 55）。"""
    sentence = "员工应当按时上下班，不得迟到早退，违反考勤制度者按旷工处理。"
    source = "\n".join([sentence] * 60)

    _children, parent_text = _parent_lines(source=source, file_name="同句.txt")

    sentences = [item for item in re.split(r"(?<=。)", parent_text) if item.strip()]
    assert len(sentences) == 60
    assert parent_text.count(sentence) == 60


def test_parent_keeps_every_identical_44_token_sentence():
    """44-token 同句重复 60 行：真实重叠仍要剥，正文重复一行都不能少。"""
    sentence = "员工应当遵守考勤管理、请假审批和信息安全要求，由人事部负责解释并定期接受检查。"
    assert count_chunk_tokens(sentence) == 44
    source = "\n".join([sentence] * 60)

    _children, parent_text = _parent_lines(source=source, file_name="同句.txt")

    sentences = [item for item in re.split(r"(?<=。)", parent_text) if item.strip()]
    assert len(sentences) == 60
    assert parent_text.count(sentence) == 60


def test_parent_keeps_every_identical_85_token_paragraph():
    """12 段相同的 85-token 段落：parent 段落数必须等于原文（旧实现只剩 8 段）。"""
    paragraph = (
        "员工应当遵守考勤管理、请假审批和信息安全要求，由人事部负责解释并归档保存，"
        "违反规定的按照公司制度处理，各部门负责人承担管理责任并定期接受公司监督检查相关。"
    )
    assert count_chunk_tokens(paragraph) == 85
    source = "\n\n".join([paragraph] * 12)

    _children, parent_text = _parent_lines(source=source, file_name="同段落.txt")

    paragraphs = [item for item in re.split(r"(?<=。)", parent_text) if item.strip()]
    assert len(paragraphs) == 12
    assert parent_text.count(paragraph) == 12


def test_parent_keeps_every_identical_table_row():
    """200 行相同表格行：表格行不参与重叠，parent 行数必须等于原文（旧实现 185 行）。"""
    header = "姓名 | 事项 | 次数 | 处理"
    row = "张三 | 迟到 | 一次 | 口头提醒"
    source_lines = [header, *([row] * 200)]
    source = "\n".join(source_lines)

    _children, parent_text = _parent_lines(source=source, file_name="表格.txt")

    assert parent_text.split("\n") == source_lines
    assert parent_text.count(row) == 200
    assert all(
        chunk["metadata"]["overlap_text"] == ""
        for chunk in chunk_text(source, 1)
    )


def test_token_flush_multi_piece_records_overlap_only_on_first(monkeypatch):
    """一次 flush 因硬上限拆成多片时，只有第一片带 overlap_text。

    ``(target, hard, overlap) = (10, 12, 100)`` 会让重叠单元与新正文各自成片：
    重叠片记录文本，紧随其后的新片必须是空串，否则 parent 拼接会把相邻片误判成重叠。
    """
    monkeypatch.setattr(
        crud_knowledge_file, "resolve_chunk_token_budget", lambda: (10, 12, 100)
    )
    lines = ["甲甲甲甲甲。", "乙乙乙乙乙。", "丙丙丙丙丙。", "丁丁丁丁丁。", "戊戊戊戊戊。"]

    chunks = chunk_text("\n".join(lines), 1)

    overlaps = [chunk["metadata"]["overlap_text"] for chunk in chunks]
    assert overlaps[0] == ""
    assert any(overlaps)
    for index, overlap in enumerate(overlaps[:-1]):
        if overlap:
            # 重叠片之后紧跟的新正文片，其 overlap_text 必须为空。
            assert overlaps[index + 1] == ""
    # 拼接回 parent 时每行恰好出现一次。
    assert _parent_text_for_group(chunks).split("\n") == lines


# ---------------------------------------------------------------------------
# 上传 / 重建链路写入 parent 与 child 元数据
# ---------------------------------------------------------------------------


class _UploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content_type = "text/plain"
        self._content = content

    async def read(self, size=-1):
        chunk, self._content = self._content[:size], self._content[size:]
        return chunk


class _Request:
    headers: dict = {}


class _User:
    id = 7


def _run_upload(db, monkeypatch, indexed):
    monkeypatch.setattr(
        knowledge_service,
        "resolve_knowledge_base",
        lambda _db, _kid, _user_id: SimpleNamespace(id=5),
    )
    monkeypatch.setattr(
        knowledge_service.crud_knowledge_file,
        "extract_file_text",
        lambda _name, _content: LONG_ARTICLE_TEXT,
    )
    monkeypatch.setattr(
        knowledge_service,
        "add_chunks",
        lambda chunks, *_args, **_kwargs: indexed.append(chunks),
    )
    monkeypatch.setattr(knowledge_service, "delete_file_chunks", lambda _file_id: None)
    return asyncio.run(
        knowledge_service.upload_knowledge(
            request=_Request(),
            file=_UploadFile("长条款.txt", b"content"),
            knowledge_base_id=5,
            user=_User(),
            db=db,
        )
    )


def test_upload_writes_child_and_parent_metadata(metadata_db, monkeypatch):
    db = metadata_db.db
    indexed = []
    response = _run_upload(db, monkeypatch, indexed)

    assert set(response) == {"id", "knowledge_base_id", "name", "size", "created_at"}
    assert indexed and indexed[0]
    children = indexed[0]
    rows = _stored_rows(db, response["id"])
    child_rows = [row for row in rows if not row.chunk_id.startswith("p:")]
    parent_rows = [row for row in rows if row.chunk_id.startswith("p:")]

    assert {row.chunk_id for row in child_rows} == {child["id"] for child in children}
    assert child_rows and parent_rows
    parent_ids = {row.chunk_id for row in parent_rows}
    parent_text_by_id = {row.chunk_id: row.parent_text for row in parent_rows}
    assert all(parent_text_by_id.values())
    for row in child_rows:
        assert row.parent_id in parent_ids
        # parent 正文只存一份在 parent 行；child 行只存关系，避免逐行复制全文。
        assert row.parent_text is None
        assert row.token_count == count_chunk_tokens(
            next(child["text"] for child in children if child["id"] == row.chunk_id)
        )
    assert all(child["metadata"]["parent_id"] for child in children)
    assert all(child["metadata"]["parent_text"] is None for child in children)


def test_rebuild_indexes_children_and_refreshes_parent_metadata(monkeypatch):
    entry = SimpleNamespace(
        id=3,
        content=LONG_ARTICLE_TEXT,
        name="长条款.txt",
        knowledge_base_id=5,
    )

    class _RebuildQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def all(self):
            return [entry]

    class _RebuildSession:
        def query(self, *_args, **_kwargs):
            return _RebuildQuery()

        def close(self):
            pass

    indexed = []
    replaced = []
    monkeypatch.setattr(knowledge_service, "SessionLocal", lambda: _RebuildSession())
    monkeypatch.setattr(
        knowledge_service, "add_chunks", lambda chunks, *_a, **_k: indexed.append(chunks)
    )
    monkeypatch.setattr(
        knowledge_service.crud_knowledge_file,
        "replace_chunk_metadata",
        lambda _db, file_id, kid, chunks: replaced.append((file_id, kid, chunks)),
    )
    monkeypatch.setattr(
        knowledge_service.crud_knowledge_file, "delete_chunk_metadata", lambda *_a, **_k: None
    )

    knowledge_service.rebuild_existing_knowledge_index()

    assert indexed and replaced
    embedded, metadata = indexed[0], replaced[0][2]
    assert {child["id"] for child in embedded} == {
        chunk["id"] for chunk in metadata if not chunk["id"].startswith("p:")
    }
    assert any(chunk["id"].startswith("p:") for chunk in metadata)
    assert all(chunk["metadata"]["parent_id"] for chunk in embedded)


# ---------------------------------------------------------------------------
# 上下文总量预算、关键字窗口隔离与 sources 可选字段
# ---------------------------------------------------------------------------


def _text_with_at_least_tokens(min_tokens: int) -> str:
    sentence = "该条款规定员工应当遵守考勤、请假和信息安全要求，由人事部负责解释。"
    text = ""
    while count_chunk_tokens(text) < min_tokens:
        text += sentence
    return text


def test_parent_context_is_truncated_at_parent_boundaries(metadata_db, monkeypatch):
    db = metadata_db.db
    entry = _create_file(db, name="五个长条款.txt")
    parent_texts = [_text_with_at_least_tokens(2000) for _ in range(5)]
    candidates = []
    for index, parent_text in enumerate(parent_texts):
        child_id = f"c{index}"
        parent_id = f"p:{index}"
        db.add(
            KnowledgeChunkMetadata(
                file_id=entry.id,
                knowledge_base_id=5,
                chunk_id=child_id,
                parent_id=parent_id,
                section_path=f"第{index}条",
                article_no=f"第{index}条",
                token_count=count_chunk_tokens(parent_text),
                metadata_json="{}",
            )
        )
        db.add(
            KnowledgeChunkMetadata(
                file_id=entry.id,
                knowledge_base_id=5,
                chunk_id=parent_id,
                section_path=f"第{index}条",
                article_no=f"第{index}条",
                token_count=count_chunk_tokens(parent_text),
                parent_text=parent_text,
                metadata_json="{}",
            )
        )
        candidates.append(
            {
                "id": child_id,
                "chunk_id": child_id,
                "file_id": entry.id,
                "file_name": entry.name,
                "content": f"child {index}",
                "route": "planned",
            }
        )
    db.commit()

    monkeypatch.setattr(retrieval, "PARENT_CONTEXT_MAX_TOKENS", 6000)
    expanded, trace = asyncio.run(retrieval.expand_parent_chunks(candidates, db=db))

    assert trace["status"] == "done"
    assert trace["truncated"] is True
    assert trace["expanded_tokens"] <= 6000
    assert 0 < len(expanded) < 5
    # 被保留的是前 N 个完整 parent，不是被截成半条的文本。
    assert [chunk["content"] for chunk in expanded] == parent_texts[: len(expanded)]
    assert trace["expanded_tokens"] == sum(
        count_chunk_tokens(chunk["content"]) for chunk in expanded
    )
    assert trace["dropped"] == 5 - len(expanded)


def test_keyword_window_candidate_is_not_mapped_or_aggregated(metadata_db):
    db = metadata_db.db
    entry = _create_file(db, name="长条款.txt", content=LONG_ARTICLE_TEXT)
    children, parents = build_parent_child_chunks(
        chunk_text(LONG_ARTICLE_TEXT, entry.id), file_name=entry.name
    )
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, [*children, *parents])
    child_payload = _chunk_payload(children[0], file_id=entry.id)
    keyword_content = "关键词窗口的正文，绝不进入 trace，也不能被聚合成 parent。"
    keyword_payload = {
        "id": children[0]["id"],
        "chunk_id": children[0]["id"],
        "file_id": entry.id,
        "file_name": entry.name,
        "content": keyword_content,
        "route": "keyword",
    }

    expanded, trace = asyncio.run(
        retrieval.expand_parent_chunks([child_payload, keyword_payload], db=db)
    )

    assert trace["keyword_unmapped"] == 1
    assert trace["status"] == "partial"
    assert trace["expanded"] == 1
    assert len(expanded) == 2
    returned_keyword = next(chunk for chunk in expanded if chunk["route"] == "keyword")
    assert returned_keyword == keyword_payload
    serialized_trace = json.dumps(trace, ensure_ascii=False)
    assert keyword_content not in serialized_trace
    assert parents[0]["text"] not in serialized_trace


def test_build_sources_parent_fields_are_optional():
    plain = {
        "file_id": 3,
        "file_name": "制度.txt",
        "chunk_id": "0",
        "route": "planned",
        "content": "普通单层切片正文。",
    }
    plain_source = _build_sources([plain])[0]
    assert "parent_id" not in plain_source
    assert "parent_section_path" not in plain_source
    assert "parent_article_no" not in plain_source
    assert "parent_expanded" not in plain_source

    expanded = {
        **plain,
        "parent_id": "p:0",
        "parent_section_path": "第一章 总则 > 第一条",
        "parent_article_no": "第一条",
        "parent_expanded": True,
    }
    expanded_source = _build_sources([expanded])[0]
    assert expanded_source["parent_id"] == "p:0"
    assert expanded_source["parent_section_path"] == "第一章 总则 > 第一条"
    assert expanded_source["parent_article_no"] == "第一条"
    assert expanded_source["parent_expanded"] is True
