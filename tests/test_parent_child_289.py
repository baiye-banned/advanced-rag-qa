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
    build_parent_child_chunks,
    chunk_text,
    count_chunk_tokens,
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

# 单个长条款：同一条款被切成多个 base chunk，父级必须把它们聚合回一个 parent。
LONG_ARTICLE_TEXT = "第五条\n" + "".join(
    f"第{index}项规定：员工应当遵守考勤、请假和信息安全要求，由人事部负责解释。"
    for index in range(1, 61)
)


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
        # 前缀与正文一起受 #287 的同一 token 口径约束。
        assert count_chunk_tokens(child["text"]) <= target
        assert child["metadata"]["token_count"] == count_chunk_tokens(child["text"])

    # 病态超长文件名不能让前缀吃掉正文预算。
    long_name = "超长文件名" * 40 + ".txt"
    long_children, _ = build_parent_child_chunks(base_chunks, file_name=long_name)
    assert long_children
    assert max(count_chunk_tokens(child["text"]) for child in long_children) <= _hard_limit()
    assert all(child["text"].strip() for child in long_children)


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
