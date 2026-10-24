"""issue #61 回归：列表接口的 SQL 条数上界与消息历史分页。

修复前三个接口都按行补数据、且只取全量：
- `GET /api/knowledge-bases` 为算 file_count 逐库懒加载 knowledge_files，
  该语句连 LONGTEXT 的 content 列一起读进内存；
- `GET /api/chat/conversations` 逐会话懒加载 knowledge_bases；
- `GET /api/chat/conversations/{cid}` 没有分页，一次返回整段历史。

这里用 SQL 事件钩子锁住「SELECT 条数不随行数增长」「列表链路不读文件正文」，
并覆盖游标翻页的不重复、不遗漏与页大小上限。
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from crud import chat as crud_chat
from database import session as db_session
from database.session import Base
from model.models import (
    Conversation,
    KnowledgeBase,
    KnowledgeChunkMetadata,
    KnowledgeFile,
    Message,
    RevokedToken,
    User,
)
from router import chat as chat_router
from router import knowledge as knowledge_router
from service import auth_service, chat_service


# 接口契约值：默认页大小 50、单页上限 200。这里写字面量而不是从实现里读，
# 否则实现把默认页大小改小/改没，用例会跟着一起「通过」。
PAGE_LIMIT = 50
MAX_LIMIT = 200

# 约 40KB 的正文：列表接口一旦整行取文件，这里就会随文件数把 content 读进内存。
DOC = "员工考勤管理制度，迟到早退旷工的认定与处罚。" * 2_000


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


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
            Message.__table__,
        ],
    )
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()

    alice = User(username="alice", password_hash="x")
    bob = User(username="bob", password_hash="x")
    db.add_all([alice, bob])
    db.commit()

    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record_statement(conn, cursor, statement, parameters, context, executemany):
        if not executemany and statement.strip().upper().startswith("SELECT"):
            statements.append(" ".join(statement.split()))

    app = FastAPI()
    app.include_router(knowledge_router.router)
    app.include_router(chat_router.router)
    app.dependency_overrides[db_session.get_db] = lambda: db
    # 归属校验不是本文件主题，默认固定为 alice，避免登录链路引入额外查询干扰计数；
    # 需要换人的用例用 act_as() 覆盖。
    app.dependency_overrides[auth_service.get_current_user] = lambda: alice

    def act_as(user):
        app.dependency_overrides[auth_service.get_current_user] = lambda: user

    try:
        yield SimpleNamespace(
            client=TestClient(app),
            db=db,
            alice=alice,
            bob=bob,
            act_as=act_as,
            statements=statements,
        )
    finally:
        db.close()


@contextmanager
def recorded_selects(api):
    """记录请求期间引擎实际发出的 SELECT 语句。

    先 expunge 清空 identity map：否则上一次请求已加载的对象会被复用，懒加载被掩盖，
    计数就断不出 N+1。
    """
    api.db.expunge_all()
    api.statements.clear()
    yield api.statements


def _add_knowledge_base(api, name, files=0):
    base = KnowledgeBase(name=name, user_id=api.alice.id)
    api.db.add(base)
    api.db.flush()
    for index in range(files):
        api.db.add(KnowledgeFile(
            knowledge_base_id=base.id,
            user_id=api.alice.id,
            name=f"{name}-{index}.txt",
            size=len(DOC),
            content=DOC,
        ))
    api.db.commit()
    return base


def _add_conversation(api, cid, knowledge_base_id=None):
    conversation = Conversation(id=cid, user_id=api.alice.id, title=f"会话-{cid}", knowledge_base_id=knowledge_base_id)
    api.db.add(conversation)
    api.db.commit()
    return conversation


def _add_messages(api, cid, count):
    for index in range(count):
        api.db.add(Message(
            conversation_id=cid,
            role="assistant" if index % 2 else "user",
            content=f"回答内容-{index}",
            sources="[]",
            attachments="[]",
        ))
    api.db.commit()
    rows = (
        api.db.query(Message)
        .filter(Message.conversation_id == cid)
        .order_by(Message.id.asc())
        .all()
    )
    return [row.id for row in rows]


def _knowledge_base_file_counts(api):
    # 生产里每个请求一个新 Session，这里清掉 identity map 模拟同样的前提：
    # 否则关系属性可能是上一次请求留下的缓存，改动过数据库也看不出来。
    api.db.expunge_all()
    response = api.client.get("/api/knowledge-bases")
    assert response.status_code == 200
    return {item["name"]: item["file_count"] for item in response.json()}


def test_knowledge_base_list_selects_stay_constant_and_never_read_file_content(api):
    for index in range(5):
        _add_knowledge_base(api, f"kb-{index}", files=3)

    with recorded_selects(api) as statements:
        response = api.client.get("/api/knowledge-bases")
        measured = list(statements)

    assert response.status_code == 200
    assert [item["file_count"] for item in response.json()] == [3] * 5
    # 1 条主查询 + 1 条 GROUP BY 计数；这里断言上界，下面再用「加数据不加查询数」锁死 N+1。
    assert len(measured) <= 2
    # 碰 knowledge_files 表的语句只允许是计数聚合：别名写法（FROM knowledge_files AS kf）
    # 也跑不掉，比单纯找 content 字面量更严。
    file_statements = [statement for statement in measured if "FROM knowledge_files" in statement]
    assert file_statements and all("count(" in statement for statement in file_statements)
    assert not any("knowledge_files.content" in statement for statement in measured)

    for index in range(5, 10):
        _add_knowledge_base(api, f"kb-{index}", files=3)

    with recorded_selects(api) as statements:
        api.client.get("/api/knowledge-bases")
        grown = list(statements)

    assert len(grown) == len(measured)


def test_knowledge_base_file_count_tracks_file_add_and_delete(api):
    base = _add_knowledge_base(api, "制度库", files=2)
    assert _knowledge_base_file_counts(api) == {"制度库": 2}

    added = KnowledgeFile(
        knowledge_base_id=base.id,
        user_id=api.alice.id,
        name="新增.txt",
        size=len(DOC),
        content=DOC,
    )
    api.db.add(added)
    api.db.commit()
    assert _knowledge_base_file_counts(api) == {"制度库": 3}

    api.db.delete(added)
    api.db.commit()
    assert _knowledge_base_file_counts(api) == {"制度库": 2}


def test_knowledge_base_create_and_rename_report_file_count(api):
    created = api.client.post("/api/knowledge-bases", json={"name": "新库"})
    assert created.status_code == 200
    assert created.json()["file_count"] == 0

    base = _add_knowledge_base(api, "旧名", files=3)
    renamed = api.client.put(f"/api/knowledge-bases/{base.id}", json={"name": "新名"})
    assert renamed.status_code == 200
    assert renamed.json()["file_count"] == 3


def test_conversation_list_selects_stay_constant(api):
    bases = [_add_knowledge_base(api, f"kb-{index}") for index in range(6)]
    for index, base in enumerate(bases):
        _add_conversation(api, f"conv-{index}", knowledge_base_id=base.id)

    with recorded_selects(api) as statements:
        response = api.client.get("/api/chat/conversations")
        measured = list(statements)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 6
    assert {item["knowledge_base_id"] for item in body} == {base.id for base in bases}
    assert {item["knowledge_base_name"] for item in body} == {base.name for base in bases}
    # 1 条会话查询 + 1 条 selectinload 知识库；懒加载版本是 1 + 会话数。
    assert len(measured) <= 2

    for index in range(6, 12):
        _add_conversation(api, f"conv-{index}", knowledge_base_id=bases[index % len(bases)].id)

    with recorded_selects(api) as statements:
        api.client.get("/api/chat/conversations")
        grown = list(statements)

    assert len(grown) == len(measured)


def test_conversation_list_keeps_unbound_conversation(api):
    _add_conversation(api, "conv-free")

    body = api.client.get("/api/chat/conversations").json()

    assert len(body) == 1
    assert body[0]["id"] == "conv-free"
    assert body[0]["knowledge_base_id"] is None
    assert body[0]["knowledge_base_name"] == ""


def test_message_page_size_contract_matches_implementation():
    assert crud_chat.CHAT_MESSAGE_DEFAULT_LIMIT == PAGE_LIMIT
    assert crud_chat.CHAT_MESSAGE_MAX_LIMIT == MAX_LIMIT


def test_message_history_defaults_to_one_page(api):
    _add_conversation(api, "conv-page")
    message_ids = _add_messages(api, "conv-page", PAGE_LIMIT + 70)

    response = api.client.get("/api/chat/conversations/conv-page")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == PAGE_LIMIT
    # 不传参时返回最新一页，且仍是「旧 -> 新」顺序。
    assert [item["id"] for item in body] == message_ids[-PAGE_LIMIT:]


def test_message_history_cursor_pages_cover_every_message_once(api):
    _add_conversation(api, "conv-cursor")
    message_ids = _add_messages(api, "conv-cursor", 123)

    collected: list[int] = []
    cursor = None
    for _ in range(20):
        params = {"limit": 25}
        if cursor is not None:
            params["before_id"] = cursor
        body = api.client.get("/api/chat/conversations/conv-cursor", params=params).json()
        if not body:
            break
        collected = [item["id"] for item in body] + collected
        cursor = body[0]["id"]
    else:
        pytest.fail("游标翻页没有终止")

    assert collected == message_ids  # 不重复、不遗漏，且整体仍是升序


def test_message_history_page_size_is_bounded(api):
    _add_conversation(api, "conv-limit")
    _add_messages(api, "conv-limit", 3)

    url = "/api/chat/conversations/conv-limit"
    assert api.client.get(url, params={"limit": 0}).status_code == 422
    assert api.client.get(url, params={"limit": MAX_LIMIT + 1}).status_code == 422
    assert api.client.get(url, params={"limit": "abc"}).status_code == 422
    assert api.client.get(url, params={"before_id": 0}).status_code == 422

    # 直接调用 service 的路径（脚本、内部调用）同样拿不到超限页大小。
    assert chat_service.resolve_message_limit(None) == PAGE_LIMIT
    assert chat_service.resolve_message_limit(MAX_LIMIT) == MAX_LIMIT
    with pytest.raises(HTTPException) as excinfo:
        chat_service.resolve_message_limit(MAX_LIMIT + 1)
    assert excinfo.value.status_code == 422


def test_message_history_keeps_field_structure(api):
    _add_conversation(api, "conv-fields")
    _add_messages(api, "conv-fields", 2)

    body = api.client.get("/api/chat/conversations/conv-fields").json()

    assert set(body[0]) == {
        "id",
        "role",
        "content",
        "sources",
        "attachments",
        "ragas_status",
        "ragas_scores",
        "ragas_error",
        "retrieval_trace",
        "feedback",
        "image_analysis_status",
        "image_analysis_error",
        "image_description",
        "created_at",
    }


def test_message_history_unknown_conversation_returns_404(api):
    assert api.client.get("/api/chat/conversations/not-exist").status_code == 404


def test_message_history_is_scoped_to_the_owner(api):
    """归属校验现在随分页查询一起做，必须有用例钉住，避免以后漏掉把整段历史读给他人。"""
    _add_conversation(api, "conv-alice")
    _add_messages(api, "conv-alice", 3)

    api.act_as(api.bob)
    for params in ({}, {"limit": 10}, {"before_id": 999}, {"limit": 10, "before_id": 3}):
        response = api.client.get("/api/chat/conversations/conv-alice", params=params)
        assert response.status_code == 404
        assert "回答内容" not in response.text


def test_knowledge_file_list_selects_metadata_only(api):
    """同一个根因的另一处：文件列表不需要正文，不能把 LONGTEXT 整列读出来。"""
    base = _add_knowledge_base(api, "制度库", files=3)

    with recorded_selects(api) as statements:
        response = api.client.get("/api/knowledge", params={"knowledge_base_id": base.id})
        measured = list(statements)

    assert response.status_code == 200
    assert len(response.json()) == 3
    assert len(measured) == 2  # 归属校验 + 文件列表
    assert not any("knowledge_files.content" in statement for statement in measured)


def test_message_paging_is_clamped_at_the_crud_layer(api):
    """CRUD 是更底层入口：负值 LIMIT 在 SQLite 上等于不设上限，必须在最里层夹住。"""
    _add_conversation(api, "conv-clamp")
    message_ids = _add_messages(api, "conv-clamp", MAX_LIMIT + 10)

    for limit in (-1, 0, MAX_LIMIT * 10):
        rows = crud_chat.list_messages(api.db, "conv-clamp", api.alice.id, limit=limit, before_id=None)
        assert 0 < len(rows) <= MAX_LIMIT
        assert rows[-1].id == message_ids[-1]  # 取的仍是游标之前最新的一页
