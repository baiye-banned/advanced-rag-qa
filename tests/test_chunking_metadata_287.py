"""issue #287：token 预算切片与结构元数据。

覆盖：
- 中文制度文档的章节路径、条款编号与 token 上限（token 数一律用 ``count_chunk_tokens``
  断言，不拿字符数冒充）；
- 超长条款按句子回退切分、不丢正文；
- 表格行原子化、表头/行元数据；
- 空文本、重复标题；
- ``knowledge_chunk_metadata`` 的写入/替换/删除、旧数据降级与上传失败清理（SQLite）。
"""

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config import (
    CHUNK_TOKEN_LIMIT,
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SAFETY_RATIO,
    CHUNK_TOKEN_TARGET,
    normalize_chunk_token_budget,
)
from crud import knowledge_file as crud_knowledge_file
from crud.knowledge_file import (
    _hard_split_by_tokens,
    chunk_coverage_ratio,
    chunk_text,
    count_chunk_tokens,
    extract_docx_text,
    get_knowledge_content,
    resolve_chunk_token_budget,
)
from database.session import Base
from model.models import KnowledgeChunkMetadata, KnowledgeFile, User
from rag import retrieval
from service import knowledge_service


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

TABLE_TEXT = """第一章 处罚标准
第五条
姓名 | 部门 | 迟到次数
张三 | 研发部 | 2
李四 | 市场部 | 1
"""


def _hard_limit() -> int:
    return max(1, int(CHUNK_TOKEN_LIMIT * CHUNK_TOKEN_SAFETY_RATIO))


def test_chinese_institution_chunks_carry_section_and_article_metadata():
    target, hard_limit, overlap = resolve_chunk_token_budget()

    chunks = chunk_text(INSTITUTION_TEXT, file_id=1)

    assert chunks
    assert 0 < target <= hard_limit
    assert hard_limit == _hard_limit()
    assert target == min(CHUNK_TOKEN_TARGET, hard_limit)
    assert overlap == min(CHUNK_TOKEN_OVERLAP, target)

    article_nos = {chunk["metadata"]["article_no"] for chunk in chunks}
    assert article_nos == {"第一条", "第二条", "第三条", "第四条"}

    section_paths = {chunk["metadata"]["section_path"] for chunk in chunks}
    assert "第一章 总则 > 第一条" in section_paths
    assert "第二章 考勤管理 > 第四条" in section_paths

    for chunk in chunks:
        metadata = chunk["metadata"]
        # token_count 必须来自同一个计数器；拿字符数冒充在这里会直接对不上。
        assert metadata["token_count"] == count_chunk_tokens(chunk["text"])
        assert metadata["token_count"] <= hard_limit
        assert metadata["table_id"] is None
        assert metadata["row_no"] is None
        assert metadata["parent_id"] is None
        assert metadata["parent_text"] is None

    joined = "\n".join(chunk["text"] for chunk in chunks)
    for sentence in (
        "为规范公司考勤管理，特制定本制度。",
        "本制度适用于全体员工，由人事部负责解释。",
        "员工应按时上下班，不得迟到早退。",
        "员工迟到三十分钟以内记口头提醒；超过三十分钟按旷工半天处理。",
    ):
        assert sentence in joined
    assert chunk_coverage_ratio(INSTITUTION_TEXT, chunks) >= 0.9


def test_long_clause_falls_back_to_sentence_boundaries_without_losing_text():
    sentences = [
        f"第{index}项规定：员工应当遵守考勤、请假和信息安全要求，由人事部负责解释。"
        for index in range(1, 41)
    ]
    source = "第五条\n" + "".join(sentences)

    chunks = chunk_text(source, file_id=1)

    assert len(chunks) > 1
    assert all(chunk["metadata"]["token_count"] <= _hard_limit() for chunk in chunks)
    assert all(chunk["metadata"]["article_no"] == "第五条" for chunk in chunks)

    joined = "".join(chunk["text"] for chunk in chunks)
    for sentence in sentences:
        # 按句子回退：每一句都必须完整出现，不能被从中间切开或丢掉。
        assert sentence in joined
    assert chunk_coverage_ratio(source, chunks) >= 0.9


def test_table_rows_are_atomic_and_report_table_metadata():
    chunks = chunk_text(TABLE_TEXT, file_id=1)

    rows = [
        "姓名 | 部门 | 迟到次数",
        "张三 | 研发部 | 2",
        "李四 | 市场部 | 1",
    ]
    for row in rows:
        assert any(row in chunk["text"] for chunk in chunks)
    # 单元格不能被拆散：提到某个单元格的切片必须包含完整的整行。
    for chunk in chunks:
        if "张三" in chunk["text"]:
            assert "张三 | 研发部 | 2" in chunk["text"]
        if "李四" in chunk["text"]:
            assert "李四 | 市场部 | 1" in chunk["text"]

    table_chunks = [chunk for chunk in chunks if chunk["metadata"]["table_id"] == 1]
    assert table_chunks
    assert all(chunk["metadata"]["table_id"] == 1 for chunk in table_chunks)
    covered_rows = {
        row_no
        for chunk in table_chunks
        for row_no in range(
            chunk["metadata"]["row_no"],
            chunk["metadata"]["row_no_end"] + 1,
        )
    }
    assert covered_rows == {1, 2, 3}
    header_chunks = [chunk for chunk in table_chunks if chunk["metadata"]["is_table_header"]]
    assert header_chunks
    assert all(chunk["metadata"]["row_no"] == 1 for chunk in header_chunks)
    assert any("姓名 | 部门 | 迟到次数" in chunk["text"] for chunk in header_chunks)


def test_oversized_table_row_is_kept_atomic_with_explicit_warnings(caplog):
    # 单行远超硬上限：宁可超出预算也不能拆单元格或丢掉正文，并且必须留下 warning。
    oversized_row = " | ".join(f"单元格{index}" + "内容" * 200 for index in range(3))
    source = "第五条\n姓名 | 部门 | 备注\n" + oversized_row

    with caplog.at_level("WARNING"):
        chunks = chunk_text(source, file_id=1)

    assert any(oversized_row in chunk["text"] for chunk in chunks)
    assert "keeping it atomic" in caplog.text
    assert "exceeds the token hard limit" in caplog.text
    # 表格行整体仍被登记为第 2 行，且不是表头。
    row_chunks = [chunk for chunk in chunks if oversized_row in chunk["text"]]
    assert row_chunks[0]["metadata"]["row_no"] == 2
    assert row_chunks[0]["metadata"]["is_table_header"] is False


def test_docx_multi_table_keeps_document_order_and_per_table_metadata():
    """两张表必须各自编号、各自表头，并挂在原文所在的章节下（issue #287）。

    修复前 ``extract_docx_text`` 先写完全部段落、再把所有表格堆到文末：两张表的
    行在文本里连续，``_assign_table_positions`` 把它们合成 ``table_id=1``，
    ``row_no`` 从 1 连到 5，第二张表的表头被当成普通行，``section_path`` 也统一
    落到最后一个章节。这里用「3 行表 + 2 行表 + 中间隔一个章节标题」钉住修复。
    """
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("第一章 考勤管理")
    document.add_paragraph("第一条")
    document.add_paragraph("员工应按时上下班。")

    first = document.add_table(rows=3, cols=3)
    for row_index, cells in enumerate(
        [
            ("姓名", "部门", "迟到次数"),
            ("张三", "研发部", "2"),
            ("李四", "市场部", "1"),
        ]
    ):
        for column_index, value in enumerate(cells):
            first.rows[row_index].cells[column_index].text = value

    document.add_paragraph("第二章 处罚标准")
    second = document.add_table(rows=2, cols=2)
    for row_index, cells in enumerate(
        [
            ("违规行为", "处罚"),
            ("迟到超过三十分钟", "旷工半天"),
        ]
    ):
        for column_index, value in enumerate(cells):
            second.rows[row_index].cells[column_index].text = value

    document.add_paragraph("第三章 附则")
    document.add_paragraph("本制度自发布之日起施行。")

    buffer = BytesIO()
    document.save(buffer)
    extracted = extract_docx_text(buffer.getvalue())

    chunks = chunk_text(extracted, file_id=1)

    table_chunks = {
        table_id: [
            chunk
            for chunk in chunks
            if chunk["metadata"]["table_id"] == table_id
        ]
        for table_id in (1, 2)
    }
    assert table_chunks[1] and table_chunks[2]

    expected_sections = {1: "第一章 考勤管理", 2: "第二章 处罚标准"}
    expected_rows = {1: {1, 2, 3}, 2: {1, 2}}
    header_rows = {1: "姓名 | 部门 | 迟到次数", 2: "违规行为 | 处罚"}

    for table_id, table_rows in expected_rows.items():
        covered_rows = set()
        header_chunks = []
        for chunk in table_chunks[table_id]:
            metadata = chunk["metadata"]
            covered_rows.update(range(metadata["row_no"], metadata["row_no_end"] + 1))
            assert expected_sections[table_id] in metadata["section_path"]
            if metadata["is_table_header"]:
                header_chunks.append(chunk)
        # 每张表都从 row_no=1 重新开始，第二张表不会被并进 table_id=1。
        assert covered_rows == table_rows
        assert header_chunks
        assert all(chunk["metadata"]["row_no"] == 1 for chunk in header_chunks)
        assert any(header_rows[table_id] in chunk["text"] for chunk in header_chunks)

    # 边界哨兵只服务于分表，绝不能进入切片正文。
    assert all("\x00" not in chunk["text"] for chunk in chunks)


def test_table_boundary_sentinel_never_leaks_from_content_or_bm25_windows(metadata_db):
    """哨兵随抽取文本落库，但预览接口与 BM25 候选都必须先剥离（issue #287）。

    两张相邻表格之间没有段落，``extract_docx_text`` 会插入 ``_TABLE_BOUNDARY``；
    这段文本原样存进 ``KnowledgeFile.content``。如果读取边界不清理，
    ``get_knowledge_content`` 会把它显示给用户，BM25 也会把
    ``__RAG_TABLE_BOUNDARY__`` 当成一个高文档频率的伪词项并让 NUL 进入候选正文。
    """
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("第一章 表单")
    first = document.add_table(rows=2, cols=2)
    for row_index, cells in enumerate([("姓名", "部门"), ("张三", "研发部")]):
        for column_index, value in enumerate(cells):
            first.rows[row_index].cells[column_index].text = value
    # 相邻第二张表：中间没有段落，只有哨兵能把两张表分开。
    second = document.add_table(rows=2, cols=2)
    for row_index, cells in enumerate([("违规行为", "处罚"), ("迟到", "旷工半天")]):
        for column_index, value in enumerate(cells):
            second.rows[row_index].cells[column_index].text = value

    buffer = BytesIO()
    document.save(buffer)
    extracted = extract_docx_text(buffer.getvalue())
    assert "\x00__RAG_TABLE_BOUNDARY__\x00" in extracted

    db = metadata_db.db
    entry = crud_knowledge_file.create_knowledge_file(
        db,
        knowledge_base_id=5,
        name="表单.docx",
        size=len(extracted),
        content=extracted,
        user_id=7,
    )

    payload = get_knowledge_content(db, entry.id, 7)
    assert "__RAG_TABLE_BOUNDARY__" not in payload["content"]
    assert "\x00" not in payload["content"]

    windows = list(retrieval._bm25_window_texts(extracted))
    assert windows
    assert all("__RAG_TABLE_BOUNDARY__" not in window for window in windows)
    assert all("\x00" not in window for window in windows)

    retrieval.clear_bm25_cache()
    candidates = retrieval.keyword_recall(db, 5, ["姓名", "违规行为"], 10)
    assert candidates
    assert all("__RAG_TABLE_BOUNDARY__" not in chunk["content"] for chunk in candidates)
    assert all("\x00" not in chunk["content"] for chunk in candidates)


def test_token_hard_limit_is_enforced_with_multi_level_headings_and_long_clause():
    """标题前缀 + 正文必须一起算预算；超出硬上限的部分顺延到下一片。"""
    body = "员工应当遵守考勤请假和信息安全要求" * 300
    source = "第一章 总 则\n第一节 适用范围\n第一条\n" + body

    chunks = chunk_text(source, file_id=1)

    assert len(chunks) > 3
    assert all(chunk["metadata"]["token_count"] <= _hard_limit() for chunk in chunks)
    assert all(
        chunk["metadata"]["token_count"] == count_chunk_tokens(chunk["text"])
        for chunk in chunks
    )
    # 顺延不能变成丢字。
    assert chunk_coverage_ratio(source, chunks) >= 0.9


def test_hard_split_never_rewrites_rare_characters_or_emoji():
    """硬切点必须落在 UTF-8 字符边界上：拼接后与原文逐字一致且没有 U+FFFD。"""
    text = "生僻字𠮷𠀀😀🚀🧑‍🚀" * 40

    pieces = _hard_split_by_tokens(text, 7)

    assert len(pieces) > 1
    assert "".join(pieces) == text
    assert "�" not in "".join(pieces)


def test_hard_split_treats_source_replacement_characters_as_normal_text():
    """源文本自带的 U+FFFD 是合法正文，不能当成「切点切坏了」而放弃硬切。

    GBK 文本按 UTF-8 解码（``errors="replace"``）就会得到 U+FFFD。旧实现只要解码结果
    里出现这个字符就判定切点非法，于是整段剩余文本被一次性塞进最后一片，硬上限失效。
    """
    text = ("考勤制度�迟到处理�" * 30)
    max_tokens = 16

    pieces = _hard_split_by_tokens(text, max_tokens)

    assert len(pieces) > 1
    assert "".join(pieces) == text
    # 合法 U+FFFD 不能阻止切点落在字符边界上：每一片都应在硬预算内。
    assert all(count_chunk_tokens(piece) <= max_tokens for piece in pieces)
    assert "".join(pieces).count("�") == text.count("�")


def test_character_path_overlap_still_allows_one_oversized_unit():
    """字符路径保留 develop 的「允许一条超预算单元」重叠语义（issue #287 返工）。

    无标点长句会被硬切成 200 字符的单元；重叠预算是 80，单条 200 字符的超预算。
    token 路径会放弃重叠，字符路径必须仍然带上它，否则这种排版永远没有上下文重叠。
    """
    source = "甲" * 200 + "乙" * 200 + "丙" * 200 + "丁" * 200

    chunks = chunk_text(source, file_id=1, chunk_size=150, chunk_overlap=80)

    assert len(chunks) > 2
    # 第二片以第一片唯一的（超预算）正文单元开头，说明重叠确实发生了。
    assert chunks[1]["text"].startswith("甲" * 200)
    assert "乙" * 200 in chunks[1]["text"]


def test_chunk_token_budget_config_rejects_out_of_range_values(caplog):
    """LIMIT=0 / TARGET=0 / OVERLAP=-5 都回退默认值并留下 warning。"""
    with caplog.at_level("WARNING"):
        limit, target, overlap = normalize_chunk_token_budget(0, 0, -5)

    assert (limit, target, overlap) == (500, 400, 50)
    assert "CHUNK_TOKEN_LIMIT" in caplog.text
    assert "CHUNK_TOKEN_TARGET" in caplog.text
    assert "CHUNK_TOKEN_OVERLAP" in caplog.text
    # 重叠必须严格小于目标，否则每一片都会带上上一片全文。
    assert overlap < target


def test_chunk_token_budget_config_keeps_overlap_strictly_below_target():
    limit, target, overlap = normalize_chunk_token_budget(500, 10, 10)

    assert limit == 500
    assert target == 10
    assert overlap == 9


@pytest.mark.parametrize("value", [None, "", "   \n\t\n  "])
def test_empty_text_produces_no_chunks(value):
    assert chunk_text(value, file_id=1) == []


def test_duplicate_headings_keep_the_body_and_metadata_consistent():
    source = "第一章 总则\n第一章 总则\n本制度自发布之日起施行。\n第一章 总则\n"

    chunks = chunk_text(source, file_id=1)

    joined = "\n".join(chunk["text"] for chunk in chunks)
    assert "本制度自发布之日起施行。" in joined
    assert chunk_coverage_ratio(source, chunks) >= 0.9
    for chunk in chunks:
        assert chunk["metadata"]["token_count"] == count_chunk_tokens(chunk["text"])
        # 重复标题被覆盖时，回收的标题行会先摘出 heading_path 再成片，此时没有路径前缀；
        # 其余切片仍应带着这一章。
        assert chunk["metadata"]["section_path"] in {"", "第一章 总则"}


def test_explicit_chunk_size_keeps_the_character_path_but_reports_real_tokens():
    paragraphs = [
        f"第{i}段内容说明审批流程、职责边界和执行要求，确保上传后的知识库可以稳定检索。"
        for i in range(1, 9)
    ]
    source = "第一章 流程规范\n" + "\n\n".join(paragraphs)

    chunks = chunk_text(source, file_id=1, chunk_size=150, chunk_overlap=80)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["metadata"]["token_count"] == count_chunk_tokens(chunk["text"])
        # 字符路径的切片长度仍按字符预算控制（现有测试依赖），但元数据里的 token 数
        # 必须是真实 token 计数。
        assert chunk["metadata"]["token_count"] > 0


# ---------------------------------------------------------------------------
# knowledge_chunk_metadata 写入 / 删除 / 失败清理（SQLite）
# ---------------------------------------------------------------------------


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


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


def _create_file(db, *, name="考勤制度.txt", user_id=7) -> KnowledgeFile:
    entry = KnowledgeFile(
        knowledge_base_id=5,
        name=name,
        size=1,
        content=INSTITUTION_TEXT,
        user_id=user_id,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def test_metadata_table_declares_unique_file_chunk_and_file_index(metadata_db):
    inspector = inspect(metadata_db.engine)
    unique_sets = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("knowledge_chunk_metadata")
    }
    assert ("file_id", "chunk_id") in unique_sets
    index_columns = {
        tuple(index["column_names"]) for index in inspector.get_indexes("knowledge_chunk_metadata")
    }
    assert ("file_id",) in index_columns


def test_replace_chunk_metadata_is_idempotent_and_maps_every_field(metadata_db):
    db = metadata_db.db
    entry = _create_file(db)
    chunks = chunk_text(INSTITUTION_TEXT, entry.id)

    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, chunks)

    rows = (
        db.query(KnowledgeChunkMetadata)
        .filter_by(file_id=entry.id)
        .order_by(KnowledgeChunkMetadata.id.asc())
        .all()
    )
    assert len(rows) == len(chunks)
    first = rows[0]
    assert first.knowledge_base_id == 5
    assert first.chunk_id == chunks[0]["id"]
    assert first.section_path == chunks[0]["metadata"]["section_path"]
    assert first.article_no == chunks[0]["metadata"]["article_no"]
    assert first.token_count == chunks[0]["metadata"]["token_count"]
    assert first.table_id is None and first.row_no is None
    assert first.parent_id is None and first.parent_text is None
    stored_json = json.loads(first.metadata_json)
    assert stored_json["token_count"] == first.token_count
    assert stored_json["section_path"] == first.section_path

    # 重复写入是「整体替换」：旧行删掉、新行只留这一批，不会累积。
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, chunks[:2])
    remaining = db.query(KnowledgeChunkMetadata).filter_by(file_id=entry.id).all()
    assert len(remaining) == 2
    assert {row.chunk_id for row in remaining} == {chunk["id"] for chunk in chunks[:2]}


def test_delete_chunk_metadata_is_idempotent_for_legacy_files(metadata_db):
    db = metadata_db.db
    entry = _create_file(db)
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, chunk_text(INSTITUTION_TEXT, entry.id))

    crud_knowledge_file.delete_chunk_metadata(db, entry.id)
    assert db.query(KnowledgeChunkMetadata).filter_by(file_id=entry.id).count() == 0
    # 旧文件没有元数据行时同样可调用，不能抛错。
    crud_knowledge_file.delete_chunk_metadata(db, entry.id)
    crud_knowledge_file.delete_chunk_metadata(db, 999999)


def test_orm_cascade_cleans_metadata_when_a_knowledge_base_removes_files(metadata_db):
    """删知识库走的是 ORM ``db.delete(file)``，级联同样不能留下孤儿元数据行。"""
    db = metadata_db.db
    entry = _create_file(db)
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, chunk_text(INSTITUTION_TEXT, entry.id))

    db.delete(entry)
    db.commit()

    assert db.query(KnowledgeChunkMetadata).filter_by(file_id=entry.id).count() == 0


def test_delete_knowledge_file_cleans_chunk_metadata_and_tolerates_legacy_rows(metadata_db):
    db = metadata_db.db
    entry = _create_file(db)
    crud_knowledge_file.replace_chunk_metadata(db, entry.id, 5, chunk_text(INSTITUTION_TEXT, entry.id))

    assert crud_knowledge_file.delete_knowledge_file(db, entry.id, 7) is not None
    assert db.query(KnowledgeFile).filter_by(id=entry.id).count() == 0
    assert db.query(KnowledgeChunkMetadata).filter_by(file_id=entry.id).count() == 0

    # 旧数据没有元数据行：删除照常成功（检索侧同样不依赖这张表）。
    legacy = _create_file(db, name="历史文件.txt")
    assert crud_knowledge_file.delete_knowledge_file(db, legacy.id, 7) is not None
    assert db.query(KnowledgeFile).filter_by(id=legacy.id).count() == 0


# ---------------------------------------------------------------------------
# 上传链路的元数据写入与失败清理
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


def _run_upload(db, monkeypatch, add_chunks):
    monkeypatch.setattr(
        knowledge_service,
        "resolve_knowledge_base",
        lambda _db, _kid, _user_id: SimpleNamespace(id=5),
    )
    monkeypatch.setattr(
        knowledge_service.crud_knowledge_file,
        "extract_file_text",
        lambda _name, _content: INSTITUTION_TEXT,
    )
    monkeypatch.setattr(knowledge_service, "add_chunks", add_chunks)
    monkeypatch.setattr(knowledge_service, "delete_file_chunks", lambda _file_id: None)
    return asyncio.run(
        knowledge_service.upload_knowledge(
            request=_Request(),
            file=_UploadFile("考勤制度.txt", b"content"),
            knowledge_base_id=5,
            user=_User(),
            db=db,
        )
    )


def test_upload_writes_chunk_metadata_and_keeps_the_response_shape(metadata_db, monkeypatch):
    db = metadata_db.db
    indexed = []
    # 本用例钉住 #287 的单层语义；父子切片（#289）默认开启时另见
    # tests/test_parent_child_289.py 的上传断言。
    monkeypatch.setattr(knowledge_service, "PARENT_CHILD_ENABLED", False)
    response = _run_upload(db, monkeypatch, lambda chunks, *_args, **_kwargs: indexed.append(chunks))

    assert set(response) == {"id", "knowledge_base_id", "name", "size", "created_at"}
    assert indexed and indexed[0]
    rows = db.query(KnowledgeChunkMetadata).filter_by(file_id=response["id"]).all()
    assert len(rows) == len(indexed[0])
    assert {row.chunk_id for row in rows} == {chunk["id"] for chunk in indexed[0]}
    assert all(row.token_count > 0 for row in rows)


def test_upload_cleans_file_and_metadata_when_metadata_write_fails(metadata_db, monkeypatch):
    db = metadata_db.db

    def fail_metadata(*_args, **_kwargs):
        raise SQLAlchemyError("metadata write failed")

    monkeypatch.setattr(knowledge_service.crud_knowledge_file, "replace_chunk_metadata", fail_metadata)

    with pytest.raises(HTTPException) as exc_info:
        _run_upload(db, monkeypatch, lambda *_args, **_kwargs: None)

    assert exc_info.value.status_code == 500
    assert "切片元数据写入异常" in exc_info.value.detail
    assert db.query(KnowledgeFile).count() == 0
    assert db.query(KnowledgeChunkMetadata).count() == 0


def test_timeout_cleanup_deletes_chunk_metadata_even_when_vector_delete_fails(monkeypatch):
    """向量没删干净、知识文件行保留等待重建时，切片元数据仍要清掉（issue #287）。"""
    calls = []

    class _CleanupSession:
        def close(self):
            pass

    monkeypatch.setattr(knowledge_service, "SessionLocal", lambda: _CleanupSession())

    def fail_vector_delete(_file_id):
        raise RuntimeError("milvus unavailable")

    monkeypatch.setattr(knowledge_service, "delete_file_chunks", fail_vector_delete)
    monkeypatch.setattr(
        knowledge_service.crud_knowledge_file,
        "delete_chunk_metadata",
        lambda _db, file_id: calls.append(("chunk-metadata", file_id)),
    )
    monkeypatch.setattr(
        knowledge_service.crud_knowledge_file,
        "delete_knowledge_file",
        lambda *_args: calls.append(("knowledge-file",)),
    )

    knowledge_service._delete_file_vectors_and_row(11, 7)

    assert calls == [("chunk-metadata", 11)]


def test_upload_cleans_file_and_metadata_when_vector_write_fails(metadata_db, monkeypatch):
    db = metadata_db.db

    def fail_index(*_args, **_kwargs):
        raise RuntimeError("milvus unavailable")

    with pytest.raises(HTTPException) as exc_info:
        _run_upload(db, monkeypatch, fail_index)

    assert exc_info.value.status_code == 500
    assert db.query(KnowledgeFile).count() == 0
    assert db.query(KnowledgeChunkMetadata).count() == 0
