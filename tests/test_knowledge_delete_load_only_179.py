"""issue #179 回归：知识库删除链路不得把文件正文（LONGTEXT）读进内存。

`list_files_for_knowledge_base` 原先整行实例化 `KnowledgeFile`，把 content 也一起取回，
而两个调用方一个只要 `.id`（`service/knowledge_service.py` 的文件列表，用于按文件清向量），
一个只要能删行（`crud/knowledge_base.py` 的 `delete_knowledge_base_with_files`）。
这与 issue #61 修掉的 `list_knowledge_files` 是同一根因——那次只修了兄弟函数（列表链路），
删除链路留在原地。这里逐条钉住收口：

- 该函数的 SELECT 列清单与 `list_knowledge_files` 对齐（5 个元数据列，无 content），
  按列清单断言而不是按语句条数断言：换写法、加 where 都不该让用例失效；
- 两个调用方的可观察行为不变：`file_ids` 取值不变（删除链路按同一批 id 清向量），
  `db.delete()` 仍按主键发 DELETE，且不为部分加载的实体追加一条「按 id 补读正文」的 SELECT。

注意本文件钉的是「本函数不再读正文」：`db.delete(KnowledgeBase)` 还会触发一条 ORM
级联加载（`KnowledgeBase.files`），它同样整行取 content，但由父实体触发、与本函数无关，
修它要动关系级联语义。该残余在下方用例里被显式钉住，避免把两种读法混为一谈。
"""

from contextlib import contextmanager
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from crud import knowledge_base as crud_knowledge_base
from database import session as db_session
from database.session import Base
from model.models import (
    Conversation,
    KnowledgeBase,
    KnowledgeChunkMetadata,
    KnowledgeFile,
    RevokedToken,
    User,
)
from router import knowledge as knowledge_router
from service import auth_service, knowledge_service


# 与 list_knowledge_files 同一口径的 5 个元数据列（写成字面量而不是从实现里读：
# 实现把列集改宽，用例必须跟着红，而不是跟着一起「通过」）。
EXPECTED_COLUMNS = {
    "knowledge_files_id",
    "knowledge_files_knowledge_base_id",
    "knowledge_files_name",
    "knowledge_files_size",
    "knowledge_files_created_at",
}

# 约 40KB 的正文：整行取文件时，这里就是被搬进内存的那部分。
DOC = "员工考勤管理制度，迟到早退旷工的认定与处罚。" * 2_000

# 本函数自己的查询形状：`WHERE knowledge_files.knowledge_base_id = ?`（被删库的 id 在后）。
BASE_FILTER = "knowledge_files.knowledge_base_id = ?"

# ORM 级联加载的形状：`WHERE ? = knowledge_files.knowledge_base_id`。参数在前是被删除
# 父实体的主键，正是「与文件 id 无关」的判据，与上面的形状恰好互为镜像。
CASCADE_WHERE_RE = re.compile(r"WHERE\s+\?\s*=\s*knowledge_files\.knowledge_base_id\s*$")


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


def _file_statements(statements) -> list[str]:
    """规范化后只留碰 knowledge_files 表的语句（SELECT 与 DELETE 都算）。"""
    return [" ".join(statement.split()) for statement in statements if "FROM knowledge_files" in statement]


def _projected_columns(statement: str) -> list[str]:
    """取 SELECT 的投影列名（`knowledge_files.id AS knowledge_files_id` -> `knowledge_files_id`）。"""
    projection, _, _from = statement.partition(" FROM ")
    assert projection.startswith("SELECT "), statement
    return [item.partition(" AS ")[2] or item for item in projection[len("SELECT "):].split(", ")]


@pytest.fixture()
def api():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        bind=engine,
        tables=[
            User.__table__, RevokedToken.__table__,
            KnowledgeBase.__table__,
            KnowledgeFile.__table__,
            KnowledgeChunkMetadata.__table__,
            Conversation.__table__,
        ],
    )
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()

    alice = User(username="alice", password_hash="x")
    db.add(alice)
    db.commit()

    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record_statement(conn, cursor, statement, parameters, context, executemany):
        # DELETE 也要记：本文件一半的断言落在删除语句的形状上。
        statements.append(statement)

    app = FastAPI()
    app.include_router(knowledge_router.router)
    app.dependency_overrides[db_session.get_db] = lambda: db
    app.dependency_overrides[auth_service.get_current_user] = lambda: alice

    try:
        yield SimpleNamespace(client=TestClient(app), db=db, alice=alice, statements=statements)
    finally:
        db.close()


@contextmanager
def recorded(api):
    """记录一次调用期间引擎实际发出的全部语句。

    先 expunge 清空 identity map：否则已加载的对象会被复用，懒加载与级联都可能被掩盖。
    """
    api.db.expunge_all()
    api.statements.clear()
    yield api.statements


def _add_knowledge_base(api, name, files=0):
    """建库并返回 (库 id, 该库下文件 id 列表)。"""
    base = KnowledgeBase(name=name, user_id=api.alice.id)
    api.db.add(base)
    api.db.flush()
    file_ids = []
    for index in range(files):
        entry = KnowledgeFile(
            knowledge_base_id=base.id,
            user_id=api.alice.id,
            name=f"{name}-{index}.txt",
            size=len(DOC),
            content=DOC,
        )
        api.db.add(entry)
        api.db.flush()
        file_ids.append(entry.id)
    api.db.commit()
    return base.id, file_ids


def test_list_files_for_knowledge_base_projects_metadata_columns_only(api):
    """本函数的 SELECT 只投影 5 个元数据列，content 不在其中。"""
    kid, file_ids = _add_knowledge_base(api, "制度库", files=3)

    with recorded(api) as statements:
        rows = crud_knowledge_base.list_files_for_knowledge_base(api.db, kid)
        measured = _file_statements(statements)

    assert [row.id for row in rows] == file_ids  # 取值不变
    assert len(measured) == 1, measured
    assert set(_projected_columns(measured[0])) == EXPECTED_COLUMNS
    assert "knowledge_files.content" not in measured[0]


def test_delete_knowledge_base_deletes_files_by_primary_key(api, monkeypatch):
    """删除链路：文件行仍按主键逐条 DELETE，且删除前不追加按 id 补读正文的 SELECT。"""
    kid, file_ids = _add_knowledge_base(api, "被删库", files=3)
    _add_knowledge_base(api, "兜底库")  # 至少保留一个知识库

    cleaned: list[int] = []
    monkeypatch.setattr(knowledge_service, "delete_file_chunks", lambda file_id: cleaned.append(file_id))

    with recorded(api) as statements:
        response = api.client.delete(f"/api/knowledge-bases/{kid}")
        measured = _file_statements(statements)

    assert response.status_code == 200
    assert cleaned == file_ids  # 调用方仍然只拿 id

    deletes = [statement for statement in measured if statement.startswith("DELETE")]
    assert deletes and all("WHERE knowledge_files.id = ?" in statement for statement in deletes)

    # 「按 id 补读正文」的形状：SQLAlchemy 若为部分加载的实体回头补查，长的就是这个样子。
    assert not [
        statement
        for statement in measured
        if statement.startswith("SELECT")
        and "knowledge_files.content" in statement
        and "knowledge_files.id = ?" in statement
    ]

    # 行为不变：库与其下文件都真的没了
    assert api.db.query(KnowledgeFile).filter_by(knowledge_base_id=kid).count() == 0
    assert api.db.query(KnowledgeBase).filter_by(id=kid).count() == 0


def test_delete_knowledge_base_file_selects_never_read_content(api, monkeypatch):
    """删除链路上凡本函数发出的文件查询都只取元数据列，不再整行读正文。"""
    kid, _ = _add_knowledge_base(api, "被删库", files=3)
    _add_knowledge_base(api, "兜底库")
    monkeypatch.setattr(knowledge_service, "delete_file_chunks", lambda file_id: None)

    with recorded(api) as statements:
        response = api.client.delete(f"/api/knowledge-bases/{kid}")
        measured = _file_statements(statements)

    assert response.status_code == 200
    selects = [statement for statement in measured if statement.startswith("SELECT")]
    # 删除链路会两次调用本函数（取 file_ids 清向量、取待删行），两条都必须只投影 5 列。
    assert [set(_projected_columns(statement)) for statement in selects if BASE_FILTER in statement] == [
        EXPECTED_COLUMNS,
        EXPECTED_COLUMNS,
    ]

    # 残余读法（不是本函数发出的）：db.delete(entry) 时 ORM 会为 KnowledgeBase.files
    # 补一条级联加载，它仍整行取 content。上面的断言留下的其余 content 读只允许是这一条，
    # 换形状就会红——这样「本函数不再读正文」与「删除链路零正文读取」不会被混为一谈。
    residual = [statement for statement in selects if "knowledge_files.content" in statement]
    assert residual and all(CASCADE_WHERE_RE.search(statement) for statement in residual)
