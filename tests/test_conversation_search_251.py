"""issue #251：侧栏会话按标题做服务端检索。

覆盖接口契约（`q` 参数、响应仍是数组、空白视同未传）、LIKE 元字符按字面量匹配、
用户隔离、与 (updated_at, id) 键集游标的组合，以及当前测试库的大小写语义。
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import session as db_session
from database.session import Base
from model.models import Conversation, KnowledgeBase, RevokedToken, User
from router import chat as chat_router
from service import auth_service


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
            User.__table__,
            RevokedToken.__table__,
            KnowledgeBase.__table__,
            Conversation.__table__,
        ],
    )
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)()

    alice = User(username="alice", password_hash="x")
    bob = User(username="bob", password_hash="x")
    db.add_all([alice, bob])
    db.commit()

    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[db_session.get_db] = lambda: db
    # 归属校验不是本文件主题：默认固定为 alice，需要换人的用例用 act_as() 覆盖。
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
        )
    finally:
        db.close()


def _add(api, cid, title, *, user=None, updated_at=None):
    conversation = Conversation(
        id=cid,
        user_id=(user or api.alice).id,
        title=title,
    )
    if updated_at is not None:
        conversation.updated_at = updated_at
        conversation.created_at = updated_at
    api.db.add(conversation)
    api.db.commit()
    return conversation


def _search(api, q, **params):
    response = api.client.get("/api/chat/conversations", params={"q": q, **params})
    assert response.status_code == 200
    return response.json()


def _ids(body):
    return [item["id"] for item in body]


# ---------------------------------------------------------------- 基本命中

def test_matches_chinese_title_substring(api):
    _add(api, "conv-kb", "如何配置知识库")
    _add(api, "conv-upload", "文件上传失败怎么办")
    _add(api, "conv-other", "模型参数说明")

    body = _search(api, "知识库")

    assert _ids(body) == ["conv-kb"]
    # 响应结构保持数组，元素字段不变：前端仍靠「整页返回」推断是否还有下一页。
    assert set(body[0]) == {
        "id", "title", "knowledge_base_id", "knowledge_base_name", "created_at", "updated_at",
    }


def test_trims_whitespace_around_query(api):
    _add(api, "conv-hit", "季度总结")
    _add(api, "conv-miss", "周报")

    assert _ids(_search(api, "  季度  ")) == ["conv-hit"]


def test_empty_and_missing_query_keep_the_default_list(api):
    start = datetime(2026, 9, 1, 10, 0, 0)
    for index, title in enumerate(["会话一", "会话二", "会话三"]):
        _add(api, f"conv-{index}", title, updated_at=start + timedelta(seconds=index))

    baseline = _ids(api.client.get("/api/chat/conversations").json())

    for blank in ("", "   ", "\t\n"):
        assert _ids(_search(api, blank)) == baseline


def test_query_longer_than_the_contract_is_rejected(api):
    response = api.client.get("/api/chat/conversations", params={"q": "a" * 201})

    assert response.status_code == 422


# ---------------------------------------------------------------- LIKE 元字符

def test_percent_is_matched_literally_not_as_a_wildcard(api):
    _add(api, "conv-exact", "进度 100% 完成")
    _add(api, "conv-prefix", "进度 1000 完成")
    _add(api, "conv-any", "进度 100 完成")

    assert _ids(_search(api, "100%")) == ["conv-exact"]


def test_underscore_is_matched_literally_not_as_a_wildcard(api):
    _add(api, "conv-underscore", "a_b 配置")
    _add(api, "conv-any-char", "axb 配置")

    assert _ids(_search(api, "a_b")) == ["conv-underscore"]


def test_escape_character_itself_is_matched_literally(api):
    # `!` 是 ESCAPE 字符：若不先把自身翻倍，输入 `!%` 会被解释成「字面百分号」。
    _add(api, "conv-bang", "重要!提示")
    _add(api, "conv-plain", "重要提示")
    _add(api, "conv-empty", "重要")

    assert _ids(_search(api, "!")) == ["conv-bang"]
    assert _ids(_search(api, "!%")) == []


# ---------------------------------------------------------------- 归属隔离

def test_only_returns_the_current_users_conversations(api):
    _add(api, "alice-hit", "共享标题 甲")
    _add(api, "alice-other", "别的内容")
    _add(api, "bob-hit", "共享标题 乙", user=api.bob)

    assert _ids(_search(api, "共享标题")) == ["alice-hit"]

    api.act_as(api.bob)
    assert _ids(_search(api, "共享标题")) == ["bob-hit"]


# ---------------------------------------------------------------- 与游标组合

def test_search_composes_with_keyset_cursor_without_duplicates_or_gaps(api):
    start = datetime(2026, 9, 1, 10, 0, 0)
    # 命中与不命中交替，确保过滤后的游标不会退化成「在整个列表里翻页」。
    expected = []
    for index in range(13):
        cid = f"match-{index:02d}"
        _add(api, cid, f"知识库问题 {index}", updated_at=start + timedelta(seconds=index * 2))
        expected.append(cid)
        _add(
            api,
            f"miss-{index:02d}",
            f"无关会话 {index}",
            updated_at=start + timedelta(seconds=index * 2 + 1),
        )
    expected.reverse()  # 接口按 updated_at 倒序

    collected = []
    cursor = None
    for _ in range(10):
        params = {"limit": 5}
        if cursor is not None:
            params["before_updated_at"], params["before_id"] = cursor
        body = _search(api, "知识库", **params)
        if not body:
            break
        collected.extend(_ids(body))
        cursor = (body[-1]["updated_at"], body[-1]["id"])
    else:
        pytest.fail("游标翻页没有终止")

    assert collected == expected
    assert len(collected) == len(set(collected)) == 13


def test_search_page_exactly_full_ends_with_an_empty_next_page(api):
    """命中数恰好等于页大小时，前端会按「整页 ⇒ 还有」再请求一次，必须拿到空数组。"""
    start = datetime(2026, 9, 1, 10, 0, 0)
    for index in range(5):
        _add(api, f"hit-{index}", f"年度报告 {index}", updated_at=start + timedelta(seconds=index))
    for index in range(7):
        _add(api, f"miss-{index}", f"其他内容 {index}", updated_at=start + timedelta(seconds=20 + index))

    first = _search(api, "年度报告", limit=5)

    assert len(first) == 5
    next_page = _search(
        api,
        "年度报告",
        limit=5,
        before_updated_at=first[-1]["updated_at"],
        before_id=first[-1]["id"],
    )
    assert next_page == []


# ---------------------------------------------------------------- 大小写语义

def test_ascii_case_is_case_insensitive_on_the_test_database(api):
    """显式锁住当前测试库（SQLite）的 LIKE 语义：ASCII 默认不区分大小写。

    SQLite 的 `LIKE` 默认 `case_sensitive_like=OFF`，只对 ASCII 折叠大小写；接口层
    没有额外做 lower()。生产若换成区分大小写的排序规则，这里要跟着改成那条契约。
    """
    _add(api, "conv-alpha", "Alpha 发布计划")
    _add(api, "conv-beta", "beta 发布计划")

    assert _ids(_search(api, "alpha")) == ["conv-alpha"]
    assert _ids(_search(api, "ALPHA")) == ["conv-alpha"]
    assert _ids(_search(api, "aLpHa")) == ["conv-alpha"]
    assert _ids(_search(api, "BETA")) == ["conv-beta"]
