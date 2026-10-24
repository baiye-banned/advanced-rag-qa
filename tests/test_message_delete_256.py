"""删除单条消息（issue #256）：用户消息截断后续，助手消息只删一条。

本文件钉住四层契约：

- **语义**：删用户消息 = 删它和它之后的所有消息（编辑后重发的同款截断）；删助手消息 =
  只删这一条。删完重新 GET 历史不复活，真实行删除而不是前端隐藏。
- **归属**：`message_id + user_id` 一起查；别人的消息与不存在的消息一律 404，且不回显正文。
- **联动状态**：会话 `updated_at` 前进（侧栏排序）；记忆摘要覆盖到被删区间时作废；
  附件键写进 #238 的墓碑状态，**不在请求路径碰 OSS**。
- **事务**：一次提交；提交失败时消息、摘要、附件行全部回滚。
- **保留附件**（issue #253）：`keep_attachments=true` 只保留**目标消息自身**的附件（供
  编辑/重新生成紧接着重发）；同一截断范围里其它消息的附件仍按默认路径登记墓碑。消息删除、
  `updated_at`、记忆摘要与归属语义不变。

夹具骨架取自 `tests/test_message_feedback_250.py`（HTTP 归属）与
`tests/test_conversation_activity_bump_239.py`（内存 SQLite + 时间戳断言）。
"""

import json
import logging
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

from crud import chat as crud_chat
from database import session as db_session
from database.session import Base
from model.models import ChatAttachmentUpload, Conversation, KnowledgeBase, Message, User
from router import chat as chat_router
from service import auth_service, chat_service


# Message.content / retrieval_trace 是 MySQL 方言的 LONGTEXT，SQLite 无法直接建表。
@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


# 播种时间落在过去：删除时写入的是 func.now()，若播种值在未来，断言会假绿。
T0 = datetime(2020, 1, 1)
ATTACHMENT_KEY = "rag-chat/2026/10/message-delete-256.png"
LATER_ATTACHMENT_KEY = "rag-chat/2026/10/message-delete-256-later.png"


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
            KnowledgeBase.__table__,  # list_conversations 会 selectinload，表必须存在
            Conversation.__table__,
            Message.__table__,
            ChatAttachmentUpload.__table__,
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
    app.dependency_overrides[auth_service.get_current_user] = lambda: alice

    def act_as(user):
        app.dependency_overrides[auth_service.get_current_user] = lambda: user

    try:
        yield SimpleNamespace(client=TestClient(app), db=db, alice=alice, bob=bob, act_as=act_as)
    finally:
        db.close()


def _add_conversation(api, cid, *, user=None, updated_at=None):
    conversation = Conversation(
        id=cid,
        user_id=(user or api.alice).id,
        title=f"会话-{cid}",
    )
    if updated_at is not None:
        conversation.updated_at = updated_at
        conversation.created_at = updated_at
    api.db.add(conversation)
    api.db.commit()
    return conversation


def _add_message(api, cid, role, content, *, attachments=None):
    message = Message(
        conversation_id=cid,
        role=role,
        content=content,
        sources="[]",
        attachments=json.dumps(attachments or [], ensure_ascii=False),
    )
    api.db.add(message)
    api.db.commit()
    api.db.refresh(message)
    return message.id


def _delete(api, message_id, **params):
    return api.client.delete(f"/api/chat/messages/{message_id}", params=params)


def _history(api, cid):
    response = api.client.get(f"/api/chat/conversations/{cid}")
    assert response.status_code == 200
    return response.json()


def _reload_conversation(api, cid):
    api.db.expunge_all()
    return api.db.query(Conversation).filter_by(id=cid).one()


def _stored_updated_at(api, cid):
    return _reload_conversation(api, cid).updated_at


# --- 语义：用户消息截断、助手消息单删 ------------------------------------------------


def test_deleting_user_message_truncates_it_and_everything_after(api):
    _add_conversation(api, "conv-truncate")
    first_user = _add_message(api, "conv-truncate", "user", "第一问")
    first_answer = _add_message(api, "conv-truncate", "assistant", "第一答")
    second_user = _add_message(api, "conv-truncate", "user", "第二问")
    second_answer = _add_message(api, "conv-truncate", "assistant", "第二答")

    response = _delete(api, second_user)

    assert response.status_code == 200
    assert response.json() == {
        "message": "ok",
        "deleted_message_ids": [second_user, second_answer],
    }
    # 真的删了行，不是前端藏起来。
    assert [row["id"] for row in _history(api, "conv-truncate")] == [first_user, first_answer]


def test_deleted_messages_do_not_come_back_after_refresh(api):
    _add_conversation(api, "conv-refresh")
    keep = _add_message(api, "conv-refresh", "user", "保留")
    target = _add_message(api, "conv-refresh", "assistant", "删掉")

    assert _delete(api, target).status_code == 200

    # 刷新 = 再发一次 GET；行已经不在库里，不会复活。
    assert [row["id"] for row in _history(api, "conv-refresh")] == [keep]
    api.db.expunge_all()
    assert api.db.query(Message).filter_by(id=target).first() is None


def test_deleting_assistant_message_removes_only_that_row(api):
    _add_conversation(api, "conv-assistant")
    question = _add_message(api, "conv-assistant", "user", "提问")
    answer = _add_message(api, "conv-assistant", "assistant", "回答")
    follow_up = _add_message(api, "conv-assistant", "user", "继续问")

    response = _delete(api, answer)

    assert response.status_code == 200
    assert response.json()["deleted_message_ids"] == [answer]
    assert [row["id"] for row in _history(api, "conv-assistant")] == [question, follow_up]


def test_deleting_the_last_empty_user_message_leaves_an_empty_conversation(api):
    _add_conversation(api, "conv-empty")
    target = _add_message(api, "conv-empty", "user", "")

    assert _delete(api, target).status_code == 200

    assert _history(api, "conv-empty") == []
    # 会话本身还在：删消息不该顺手删掉会话。
    assert _reload_conversation(api, "conv-empty") is not None


def test_deleting_the_same_message_twice_is_404(api):
    _add_conversation(api, "conv-twice")
    target = _add_message(api, "conv-twice", "assistant", "回答")

    assert _delete(api, target).status_code == 200
    repeated = _delete(api, target)

    assert repeated.status_code == 404
    assert _history(api, "conv-twice") == []


# --- 归属与目标校验 -----------------------------------------------------------------


def test_deleting_someone_elses_message_is_404_and_changes_nothing(api):
    _add_conversation(api, "conv-alice")
    target = _add_message(api, "conv-alice", "user", "别人的内容")
    _add_message(api, "conv-alice", "assistant", "别人的回答")

    api.act_as(api.bob)
    response = _delete(api, target)

    assert response.status_code == 404
    assert "别人的内容" not in response.text

    api.act_as(api.alice)
    rows = _history(api, "conv-alice")
    assert [row["id"] for row in rows] == [target, target + 1]


def test_deleting_an_unknown_message_is_404(api):
    assert _delete(api, 999999).status_code == 404


# --- 会话 updated_at -----------------------------------------------------------------


def test_deletion_advances_conversation_updated_at_and_sidebar_order(api):
    _add_conversation(api, "conv-old", updated_at=T0)
    _add_conversation(api, "conv-new", updated_at=T0 + timedelta(seconds=1))
    target = _add_message(api, "conv-old", "user", "提问")
    _add_message(api, "conv-old", "assistant", "回答")

    template = [conversation.id for conversation in crud_chat.list_conversations(api.db, api.alice.id, limit=50)]
    assert template == ["conv-new", "conv-old"], "前置：最近活动优先"

    before = _stored_updated_at(api, "conv-old")
    assert _delete(api, target).status_code == 200
    after = _stored_updated_at(api, "conv-old")

    assert after != before, "删除也是内容变更，updated_at 必须前进"
    assert after > T0 + timedelta(seconds=1), f"没有越过原最新值：{after!r}"
    assert [
        conversation.id for conversation in crud_chat.list_conversations(api.db, api.alice.id, limit=50)
    ] == ["conv-old", "conv-new"]


# --- 记忆摘要 -----------------------------------------------------------------------


def test_memory_summary_is_cleared_when_a_deleted_message_is_covered(api):
    _add_conversation(api, "conv-memory-covered")
    _add_message(api, "conv-memory-covered", "user", "问题")
    target = _add_message(api, "conv-memory-covered", "assistant", "回答")
    conversation = _reload_conversation(api, "conv-memory-covered")
    conversation.memory_summary = "这段摘要里含有已删回答"
    conversation.memory_summary_upto_message_id = target
    conversation.memory_updated_at = T0
    api.db.commit()

    assert _delete(api, target).status_code == 200

    conversation = _reload_conversation(api, "conv-memory-covered")
    # 摘要文本可能残留已删内容，不做句级清洗，直接作废让下一轮重生成。
    assert conversation.memory_summary == ""
    assert conversation.memory_summary_upto_message_id == 0
    assert conversation.memory_updated_at is None


def test_memory_summary_survives_when_pointer_is_before_the_deleted_region(api):
    _add_conversation(api, "conv-memory-kept")
    first_user = _add_message(api, "conv-memory-kept", "user", "第一问")
    first_answer = _add_message(api, "conv-memory-kept", "assistant", "第一答")
    second_user = _add_message(api, "conv-memory-kept", "user", "第二问")
    _add_message(api, "conv-memory-kept", "assistant", "第二答")
    conversation = _reload_conversation(api, "conv-memory-kept")
    conversation.memory_summary = "只覆盖第一轮的摘要"
    conversation.memory_summary_upto_message_id = first_answer
    conversation.memory_updated_at = T0
    api.db.commit()

    assert _delete(api, second_user).status_code == 200

    conversation = _reload_conversation(api, "conv-memory-kept")
    # 指针落在删除区间之前：摘要不包含被删内容，保持原样。
    assert conversation.memory_summary == "只覆盖第一轮的摘要"
    assert conversation.memory_summary_upto_message_id == first_answer
    assert conversation.memory_updated_at == T0
    assert first_user  # 前置消息保留，避免变量被优化掉时误读测试意图


# --- 附件：写墓碑（不碰 OSS） ---------------------------------------------------------


def _register_consumed_upload(api, object_key=ATTACHMENT_KEY, *, user=None):
    """走真实上传链路：登记行 + 发送时消费（consumed_at 非空）。"""
    owner = user or api.alice
    crud_chat.register_attachment_upload(api.db, object_key, owner.id)
    consumed = crud_chat.confirm_attachment_uploads(api.db, [object_key], owner.id)
    api.db.commit()
    assert consumed == 1


def _forbid_oss_path(monkeypatch):
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("删除消息的请求路径不允许回收 OSS 对象")

    monkeypatch.setattr(chat_service, "_delete_oss_object", fail_if_called)
    monkeypatch.setattr(chat_service, "reclaim_orphan_chat_attachments", fail_if_called)
    monkeypatch.setattr(crud_chat, "delete_attachment_uploads", fail_if_called)
    return calls


def test_deleting_an_attachment_message_tombstones_the_consumed_upload_without_touching_oss(api, monkeypatch):
    _add_conversation(api, "conv-attachment")
    target = _add_message(
        api,
        "conv-attachment",
        "assistant",
        "带图回答",
        attachments=[{"object_key": ATTACHMENT_KEY, "name": "photo.png"}],
    )
    # 真实链路：上传即登记、发送即消费。这也正是旧实现漏回收的形状——行一直在。
    _register_consumed_upload(api)
    before = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    assert before.consumed_at is not None
    created_at = before.created_at

    calls = _forbid_oss_path(monkeypatch)
    assert _delete(api, target).status_code == 200
    assert calls == [], "请求路径调用了 OSS 删除"

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 同一行转成墓碑：不删行、不改创建时间、不改归属。
    assert upload.user_id == api.alice.id
    assert upload.created_at == created_at
    assert upload.consumed_at is None
    assert upload.claimed_at is not None
    assert upload.reclaimed_at is not None
    assert upload.claimed_at == upload.reclaimed_at

    # 墓碑不再是 pending 候选（P 已被墓碑排除），但可以被过期租约路径重新领取，
    # 由既有清扫完成真正的 OSS 删除——这里只验证状态机可达，不发网络请求。
    pending = crud_chat.list_pending_attachment_uploads(
        api.db, older_than=datetime.now() + timedelta(seconds=1)
    )
    assert [row.object_key for row in pending] == []

    later = upload.claimed_at + timedelta(seconds=1)
    stale = crud_chat.list_stale_reclaimed_attachment_uploads(
        api.db, now=later, lease_seconds=0
    )
    assert stale == [ATTACHMENT_KEY]
    assert crud_chat.claim_stale_attachment_upload(
        api.db, ATTACHMENT_KEY, now=later, lease_seconds=0
    ) is True

    api.db.expunge_all()
    reclaimed = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 重新领取只推进租约，墓碑本身不动。
    assert reclaimed.reclaimed_at == upload.reclaimed_at
    assert reclaimed.claimed_at == later


def test_pending_upload_row_is_tombstoned_too(api):
    _add_conversation(api, "conv-pending-upload")
    target = _add_message(
        api,
        "conv-pending-upload",
        "assistant",
        "带图回答",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    # 只登记、未消费：pending 行同样要被收敛成墓碑，否则它会继续被当成孤儿候选。
    crud_chat.register_attachment_upload(api.db, ATTACHMENT_KEY, api.alice.id)
    before = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    assert before.consumed_at is None
    created_at = before.created_at

    assert _delete(api, target).status_code == 200

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    assert upload.created_at == created_at
    assert upload.consumed_at is None
    assert upload.claimed_at is not None
    assert upload.reclaimed_at is not None


def test_attachment_owned_by_another_user_is_left_untouched(api):
    _add_conversation(api, "conv-foreign-upload")
    target = _add_message(
        api,
        "conv-foreign-upload",
        "assistant",
        "引用了别人的键",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    consumed_at = T0 + timedelta(days=1)
    api.db.add(ChatAttachmentUpload(
        object_key=ATTACHMENT_KEY,
        user_id=api.bob.id,
        consumed_at=consumed_at,
        created_at=T0,
    ))
    api.db.commit()

    assert _delete(api, target).status_code == 200

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 归属判据（#233）：别人铸的键不归当前用户回收。
    assert upload.user_id == api.bob.id
    assert upload.consumed_at == consumed_at
    assert upload.claimed_at is None
    assert upload.reclaimed_at is None


def test_attachment_without_registration_row_is_skipped_not_fabricated(api, monkeypatch, caplog):
    """没有登记行 = 归属未知：不能替客户端给的键插墓碑。

    `object_key` 来自客户端消息的 attachments（仅形状校验）。若替无行的键插一条墓碑，
    等于把「删除者」写成上传者，任何合法形状的键都能被自己的消息引用后借服务凭据回收
    （#233 返修）。代价是历史老对象可能漏回收——宁可泄漏也不伪造归属。
    """
    _add_conversation(api, "conv-unknown-attachment")
    target = _add_message(
        api,
        "conv-unknown-attachment",
        "assistant",
        "引用了没有登记行的键",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )

    calls = _forbid_oss_path(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="crud.chat"):
        response = _delete(api, target)

    assert response.status_code == 200
    assert response.json()["deleted_message_ids"] == [target]
    assert calls == [], "请求路径调用了 OSS 删除/回收"

    api.db.expunge_all()
    # 不插入任何行：既不伪造归属，也不把它塞进 pending/墓碑队列。
    assert api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).count() == 0
    assert crud_chat.list_pending_attachment_uploads(
        api.db, older_than=datetime.now() + timedelta(seconds=1)
    ) == []

    warnings = [
        record
        for record in caplog.records
        if record.name == "crud.chat"
        and record.levelno == logging.WARNING
        and ATTACHMENT_KEY in record.getMessage()
    ]
    assert warnings, "无登记行的键至少要留下一条 warning"
    rendered = warnings[0].getMessage()
    assert "[scope=delete_message_missing_upload_row]" in rendered
    # 日志只带 scope/object_key，不输出消息正文。
    assert "引用了没有登记行的键" not in rendered


def test_already_reclaimed_attachment_row_is_left_untouched(api):
    _add_conversation(api, "conv-reclaimed-upload")
    target = _add_message(
        api,
        "conv-reclaimed-upload",
        "assistant",
        "带图回答",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    reclaimed_at = T0 + timedelta(hours=1)
    api.db.add(ChatAttachmentUpload(
        object_key=ATTACHMENT_KEY,
        user_id=api.alice.id,
        created_at=T0,
        reclaimed_at=reclaimed_at,
    ))
    api.db.commit()

    assert _delete(api, target).status_code == 200

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 既成事实：墓碑不重写，避免把「已结算」改回「正在删」或缩短既有租约。
    assert upload.consumed_at is None
    assert upload.claimed_at is None
    assert upload.reclaimed_at == reclaimed_at


def test_attachment_still_referenced_by_another_message_is_not_tombstoned(api):
    _add_conversation(api, "conv-shared-attachment")
    target = _add_message(
        api,
        "conv-shared-attachment",
        "assistant",
        "第一次引用",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    _add_message(
        api,
        "conv-shared-attachment",
        "assistant",
        "仍然引用",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    _register_consumed_upload(api)

    assert _delete(api, target).status_code == 200

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 还有存活消息引用这把键：不能判死。
    assert upload.consumed_at is not None
    assert upload.claimed_at is None
    assert upload.reclaimed_at is None


# --- 保留附件（issue #253）：编辑/重新生成后立即重发 ---------------------------------


def test_keep_attachments_preserves_consumed_upload_for_immediate_resend(api, monkeypatch):
    _add_conversation(api, "conv-keep-attachments", updated_at=T0)
    target = _add_message(
        api,
        "conv-keep-attachments",
        "user",
        "带图提问",
        attachments=[{"object_key": ATTACHMENT_KEY, "name": "photo.png"}],
    )
    answer = _add_message(api, "conv-keep-attachments", "assistant", "带图回答")
    _register_consumed_upload(api)

    consumed_before = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    consumed_at = consumed_before.consumed_at
    created_at = consumed_before.created_at
    assert consumed_at is not None

    before_updated_at = _stored_updated_at(api, "conv-keep-attachments")
    assert before_updated_at == T0, "前置：除本次删除外没有别的写操作推进 updated_at"

    calls = _forbid_oss_path(monkeypatch)
    response = _delete(api, target, keep_attachments="true")

    assert response.status_code == 200
    assert response.json()["deleted_message_ids"] == [target, answer]
    assert calls == [], "请求路径调用了 OSS 删除"
    assert _history(api, "conv-keep-attachments") == []
    assert _stored_updated_at(api, "conv-keep-attachments") > before_updated_at

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 行完全不动：仍是 consumed、无租约/墓碑，归属与创建时间也不变。
    assert upload.user_id == api.alice.id
    assert upload.created_at == created_at
    assert upload.consumed_at == consumed_at
    assert upload.claimed_at is None
    assert upload.reclaimed_at is None

    # 发送路径（`_save_user_message`）的判据：再次消费命中 0 行是正常的——行已经是
    # consumed，不再是 pending；关键是墓碑分类为空，因此按「老键」形态照常放行，
    # 同一批 object_key 可以立即随重发消息落库，不会被判死。
    assert crud_chat.confirm_attachment_uploads(api.db, [ATTACHMENT_KEY], api.alice.id) == 0
    assert crud_chat.list_blocked_attachment_uploads(api.db, [ATTACHMENT_KEY]) == []


def test_keep_attachments_tombstones_other_messages_in_the_truncated_tail(api, monkeypatch):
    """保留目标消息的附件，不等于保留同一截断范围里其它消息的附件。

    编辑 A 只重发 A 的 img-A；尾巴里的 B 及其 img-B 永远不会被重发，若也跳过墓碑，
    img-B 会既不在墓碑、也不在 pending、也没有消息引用——永久泄漏（#253 返修）。
    """
    _add_conversation(api, "conv-keep-tail-attachments")
    first_user = _add_message(
        api,
        "conv-keep-tail-attachments",
        "user",
        "第一问（被编辑）",
        attachments=[{"object_key": ATTACHMENT_KEY, "name": "a.png"}],
    )
    first_answer = _add_message(api, "conv-keep-tail-attachments", "assistant", "第一答")
    second_user = _add_message(
        api,
        "conv-keep-tail-attachments",
        "user",
        "第二问（随截断删除）",
        attachments=[{"object_key": LATER_ATTACHMENT_KEY, "name": "b.png"}],
    )
    second_answer = _add_message(api, "conv-keep-tail-attachments", "assistant", "第二答")

    _register_consumed_upload(api, ATTACHMENT_KEY)
    _register_consumed_upload(api, LATER_ATTACHMENT_KEY)
    api.db.expunge_all()
    kept_before = (
        api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    )
    consumed_at = kept_before.consumed_at
    created_at = kept_before.created_at
    assert consumed_at is not None

    calls = _forbid_oss_path(monkeypatch)
    response = _delete(api, first_user, keep_attachments="true")

    assert response.status_code == 200
    assert response.json()["deleted_message_ids"] == [
        first_user,
        first_answer,
        second_user,
        second_answer,
    ]
    assert calls == [], "请求路径调用了 OSS 删除"

    api.db.expunge_all()
    kept = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 目标消息的附件行完全不动，重发不会被拒签。
    assert kept.user_id == api.alice.id
    assert kept.created_at == created_at
    assert kept.consumed_at == consumed_at
    assert kept.claimed_at is None
    assert kept.reclaimed_at is None
    assert crud_chat.confirm_attachment_uploads(api.db, [ATTACHMENT_KEY], api.alice.id) == 0
    assert crud_chat.list_blocked_attachment_uploads(api.db, [ATTACHMENT_KEY]) == []

    later = api.db.query(ChatAttachmentUpload).filter_by(object_key=LATER_ATTACHMENT_KEY).one()
    # 截断尾巴里其它消息的附件：按默认路径转成墓碑，交给既有清扫回收。
    assert later.user_id == api.alice.id
    assert later.consumed_at is None
    assert later.claimed_at is not None
    assert later.reclaimed_at is not None
    assert later.claimed_at == later.reclaimed_at
    assert crud_chat.list_blocked_attachment_uploads(api.db, [LATER_ATTACHMENT_KEY]) == [
        LATER_ATTACHMENT_KEY
    ]


def test_keep_attachments_tombstones_tail_attachments_when_target_has_none(api):
    _add_conversation(api, "conv-keep-no-target-attachments")
    target = _add_message(api, "conv-keep-no-target-attachments", "user", "无附件提问")
    _add_message(api, "conv-keep-no-target-attachments", "assistant", "回答")
    _add_message(
        api,
        "conv-keep-no-target-attachments",
        "user",
        "后续带图提问",
        attachments=[{"object_key": LATER_ATTACHMENT_KEY}],
    )
    _register_consumed_upload(api, LATER_ATTACHMENT_KEY)

    assert _delete(api, target, keep_attachments="true").status_code == 200

    api.db.expunge_all()
    later = api.db.query(ChatAttachmentUpload).filter_by(object_key=LATER_ATTACHMENT_KEY).one()
    # 目标消息没有附件可保留，不代表尾部消息的附件也可以漏出回收视野。
    assert later.consumed_at is None
    assert later.claimed_at is not None
    assert later.reclaimed_at is not None


def test_keep_attachments_preserves_an_assistant_targets_own_attachment(api):
    _add_conversation(api, "conv-keep-assistant")
    _add_message(api, "conv-keep-assistant", "user", "提问")
    target = _add_message(
        api,
        "conv-keep-assistant",
        "assistant",
        "带图回答",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    _register_consumed_upload(api, ATTACHMENT_KEY)
    consumed_at = api.db.query(ChatAttachmentUpload).filter_by(
        object_key=ATTACHMENT_KEY
    ).one().consumed_at

    assert _delete(api, target, keep_attachments="true").status_code == 200

    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    # 目标不是 user 也保留目标自身的附件；单删不会波及任何其它消息。
    assert upload.consumed_at == consumed_at
    assert upload.claimed_at is None
    assert upload.reclaimed_at is None


def test_keep_attachments_applies_existing_memory_summary_rules(api):
    _add_conversation(api, "conv-keep-memory")
    _add_message(api, "conv-keep-memory", "user", "问题")
    target = _add_message(api, "conv-keep-memory", "assistant", "回答")
    conversation = _reload_conversation(api, "conv-keep-memory")
    conversation.memory_summary = "这段摘要里含有已删回答"
    conversation.memory_summary_upto_message_id = target
    conversation.memory_updated_at = T0
    api.db.commit()

    assert _delete(api, target, keep_attachments="true").status_code == 200

    conversation = _reload_conversation(api, "conv-keep-memory")
    # 保留附件不影响记忆规则：覆盖被删区间时照样作废。
    assert conversation.memory_summary == ""
    assert conversation.memory_summary_upto_message_id == 0
    assert conversation.memory_updated_at is None


def test_keep_attachments_still_requires_ownership(api):
    _add_conversation(api, "conv-keep-ownership")
    target = _add_message(
        api,
        "conv-keep-ownership",
        "user",
        "别人的带图提问",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    _register_consumed_upload(api)

    api.act_as(api.bob)
    response = _delete(api, target, keep_attachments="true")

    assert response.status_code == 404
    assert "别人的带图提问" not in response.text

    api.act_as(api.alice)
    assert [row["id"] for row in _history(api, "conv-keep-ownership")] == [target]
    api.db.expunge_all()
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    assert upload.consumed_at is not None
    assert upload.reclaimed_at is None


def test_keep_attachments_only_accepts_boolean(api):
    _add_conversation(api, "conv-keep-invalid")
    target = _add_message(api, "conv-keep-invalid", "assistant", "回答")

    assert _delete(api, target, keep_attachments="not-a-bool").status_code == 422
    # 参数校验失败时消息不能被删。
    assert [row["id"] for row in _history(api, "conv-keep-invalid")] == [target]


# --- 事务失败回滚 -------------------------------------------------------------------


def test_commit_failure_rolls_back_messages_memory_and_attachment_tombstone(api, monkeypatch):
    _add_conversation(api, "conv-rollback")
    target = _add_message(
        api,
        "conv-rollback",
        "user",
        "提问",
        attachments=[{"object_key": ATTACHMENT_KEY}],
    )
    answer = _add_message(api, "conv-rollback", "assistant", "回答")
    _register_consumed_upload(api)
    api.db.expunge_all()
    consumed_at = (
        api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one().consumed_at
    )
    conversation = _reload_conversation(api, "conv-rollback")
    conversation.memory_summary = "删除前的摘要"
    conversation.memory_summary_upto_message_id = answer
    conversation.memory_updated_at = T0
    api.db.commit()
    before_updated_at = _stored_updated_at(api, "conv-rollback")

    def boom():
        raise RuntimeError("commit failed")

    monkeypatch.setattr(api.db, "commit", boom)
    with pytest.raises(RuntimeError):
        crud_chat.delete_message(api.db, target, api.alice.id)

    # 消息、摘要、附件墓碑都必须回到删除前，不能留下半写状态。
    api.db.expunge_all()
    assert [row.id for row in api.db.query(Message).order_by(Message.id.asc()).all()] == [target, answer]
    conversation = api.db.query(Conversation).filter_by(id="conv-rollback").one()
    assert conversation.memory_summary == "删除前的摘要"
    assert conversation.memory_summary_upto_message_id == answer
    assert conversation.memory_updated_at == T0
    assert conversation.updated_at == before_updated_at
    upload = api.db.query(ChatAttachmentUpload).filter_by(object_key=ATTACHMENT_KEY).one()
    assert upload.consumed_at == consumed_at
    assert upload.claimed_at is None
    assert upload.reclaimed_at is None
