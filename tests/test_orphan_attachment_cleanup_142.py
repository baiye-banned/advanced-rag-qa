"""issue #142 回归：上传后从未被发送的聊天附件，必须有回收路径。

修复前 `POST /api/chat/attachments`（`chat_service.upload_chat_attachment`）把对象写进
OSS 就返回键，不落任何库表；唯一的上传出口 `_put_oss_object` 还会带上
`x-oss-object-acl: public-read`。而附件的回收入口只有一条、由消息驱动
（删会话 → `_reclaim_chat_attachments` ← `crud.list_conversation_attachment_keys` ←
`messages.attachments`）：没发送的对象不在任何消息里，结构上触达不到，于是永久留在桶里，
且服务端连一条可以拿来对账的记录都没有。

修复后：上传先登记一条「待确认」行（对象还没写出去就落库，写失败也留痕），发送成功时
这条登记行与消息行在**同一次提交**里被消费（此后由消息驱动那条既有路径负责回收），
超过保留窗口仍未被消费的登记行由清扫任务**实际删除对象**。

断言的都是外部可见结果：

- OSS 面用同一个替身同时顶掉 `httpx.AsyncClient`（上传出口）与 `httpx.Client`（删除出口），
  请求按方法记全，所以「对象真被删了」是请求级证据——替身若把被测函数整个换掉，
  就断不出「请求根本没发出去」；
- 登记、消费、清扫读写的都是库里真实的行，上传走真实接口、发送走真实 `stream_chat`；
- 清扫的时间基准可注入（`now=`），用例因此不依赖真实等待，也不需要打补丁时钟。

有几条是「护栏」用例（窗口内的对象不许删、外来键不许签发 DELETE、单对象失败不拖垮其余
对象）：它们在修复前也「通过」，锁的是新回收路径不得变成新的破坏面。
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from email.utils import formatdate, parsedate_to_datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from crud import chat as crud_chat
from database import session as db_session
from database.session import Base
from model.models import ChatAttachmentUpload, ChatTraceSession, Conversation, Message, RevokedToken, User
from router import chat as chat_router
from schema.schemas import ChatRequest
from service import auth_service, chat_service, oss_service


OSS_CONFIG = {
    "OSS_ACCESS_KEY_ID": "test-id",
    "OSS_ACCESS_KEY_SECRET": "test-secret",
    "OSS_BUCKET": "demo",
    "OSS_ENDPOINT": "https://oss-cn-hangzhou.aliyuncs.com",
}
OSS_HOST = "demo.oss-cn-hangzhou.aliyuncs.com"

# 保留窗口的接口契约值写字面量，不从实现里读：实现把窗口改小/改没时，用例要跟着变红，
# 而不是一起「通过」。
TTL_SECONDS = 24 * 60 * 60

# 清扫链路用的对象键写字面量，形态照抄上传路径实际铸出来的样子
# （rag-chat/<年>/<月>/<日>/<uuid4().hex><扩展名>）。
UPLOAD_A = "rag-chat/2026/09/21/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png"
UPLOAD_B = "rag-chat/2026/09/21/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"
UPLOAD_C = "rag-chat/2026/09/21/cccccccccccccccccccccccccccccccc.webp"

# 本服务从没铸过的键（客户端回带的附件列是自由 JSON，桶里别的东西都可能出现在这个位置）。
FOREIGN_UPLOAD = "finance-archive/2026/q3/payroll.sql"
NEAR_MISS_UPLOAD = "rag-chat/2026/09/21/../../finance-archive/2026/q3/payroll.sql"

PNG_BYTES = b"\x89PNG\r\n\x1a\n-fake-png-payload"


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _RecordingOssClient:
    """对象存储出口的统一替身：上传（async `put`）与回收（sync `delete`）都记在同一个列表里。

    故意不提供别的 HTTP 方法：回收不该走上传路径，上传也不该走删除路径。
    """

    def __init__(self):
        self.requests = []
        self.status_by_key = {}
        self.default_put_status = 200
        self.default_delete_status = 204

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def _status_for(self, url, fallback):
        for key, code in self.status_by_key.items():
            if key in url:
                return code
        return fallback

    async def put(self, url, content=None, headers=None, **_kwargs):
        self.requests.append({"method": "PUT", "url": url, "headers": dict(headers or {})})
        status = self._status_for(url, self.default_put_status)
        return _FakeResponse(status, "InternalError" if status >= 400 else "")

    def delete(self, url, headers=None, **_kwargs):
        self.requests.append({"method": "DELETE", "url": url, "headers": dict(headers or {})})
        status = self._status_for(url, self.default_delete_status)
        return _FakeResponse(status, "AccessDenied" if status >= 400 else "")

    def urls(self, method):
        return [request["url"] for request in self.requests if request["method"] == method]


@pytest.fixture()
def oss_requests(monkeypatch):
    """替换 OSS 的两个 HTTP 出口；同时补齐 OSS 配置。"""
    client = _RecordingOssClient()
    for name, value in OSS_CONFIG.items():
        monkeypatch.setattr(oss_service, name, value)
    monkeypatch.setattr(oss_service.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(oss_service.httpx, "Client", lambda **_kwargs: client)
    return client


@pytest.fixture()
def api(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # 上传链路（写登记行）、发送链路（消费登记行）、删除链路（清学习轨迹）读写的表都要建出来。
    Base.metadata.create_all(
        bind=engine,
        tables=[
            User.__table__, RevokedToken.__table__,
            Conversation.__table__,
            Message.__table__,
            ChatTraceSession.__table__,
            ChatAttachmentUpload.__table__,
        ],
    )
    # 会话配置照抄生产的 database.session.SessionLocal（注意 expire_on_commit 用**默认的 True**）：
    # 这里若图省事写成 expire_on_commit=False，commit 之后读 ORM 行属性就不会触发刷新，
    # 「别的清扫/发送在批中途删掉了这一行」这类缺陷会被这个更宽松的替身悄悄盖掉。
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()

    alice = User(username="alice", password_hash="x")
    bob = User(username="bob", password_hash="x")
    db.add_all([alice, bob])
    db.commit()
    # 主键先取成普通 int：下面所有读 alice/bob 属性的地方，只要可能发生在一次
    # `stream_chat` 之后，就必须拿这个整数值，不能现读 ORM 实例（原因见 alice_id 用处的注释）。
    alice_id, bob_id = alice.id, bob.id

    checkpointer_calls = []
    monkeypatch.setattr(chat_service, "delete_thread_checkpoints", checkpointer_calls.append)

    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[db_session.get_db] = lambda: db
    # 按 id 现查，而不是把 fixture 手里那个实例直接发出去：生产里 `get_current_user` 每个
    # 请求都会自己查一次库，拿到的是**绑定在本次请求会话上**的用户。这里把同一个实例发出去
    # 会在 `stream_chat` 跑完之后变成「已脱管且属性过期」——`stream_chat` 结束时 close() 掉
    # 的正是夹具借给它的这个会话实例（生产语义如此，夹具借出的就是生产语义）。于是后续
    # 请求里读 `user.id` 会抛 DetachedInstanceError，把这条用例的失败归错到实现头上。
    app.dependency_overrides[auth_service.get_current_user] = lambda: db.get(User, alice_id)

    try:
        yield SimpleNamespace(
            client=TestClient(app),
            db=db,
            alice=alice,
            alice_id=alice_id,
            bob=bob,
            bob_id=bob_id,
            checkpointer_calls=checkpointer_calls,
        )
    finally:
        db.close()


def _upload(api, filename="a.png"):
    return api.client.post("/api/chat/attachments", files={"file": (filename, PNG_BYTES, "image/png")})


def _pending_rows(api):
    return crud_chat.list_pending_attachment_uploads(api.db)


def _key_appears_anywhere(api, object_key) -> list[str]:
    """扫描库里所有表的文本列，返回出现这把键的「表.列」清单（对账入口的形态无关判据）。"""
    inspector = inspect(api.db.get_bind())
    hits = []
    for table in inspector.get_table_names():
        for column in (entry["name"] for entry in inspector.get_columns(table)):
            count = api.db.execute(
                text(f'SELECT COUNT(*) FROM "{table}" WHERE CAST("{column}" AS TEXT) LIKE :pattern'),
                {"pattern": f"%{object_key}%"},
            ).scalar()
            if count:
                hits.append(f"{table}.{column}")
    return hits


def _add_pending_row(api, object_key, *, age_seconds=0, user_id=None):
    """直接落一条登记行，用来把「上传很久以前发生」这个时间条件摆出来。"""
    row = ChatAttachmentUpload(
        object_key=object_key,
        user_id=api.alice_id if user_id is None else user_id,
        created_at=datetime.now() - timedelta(seconds=age_seconds),
    )
    api.db.add(row)
    api.db.commit()
    return row


# ---------------------------------------------------------------------------
# (a) 上传即登记：对象还没写出去，登记行就落库了
# ---------------------------------------------------------------------------

def test_an_uploaded_but_unsent_object_leaves_a_reconciliation_record(api, oss_requests):
    """issue #142 的原形：上传成功之后，库里必须留下能按对象对账的痕迹。

    修复前这条接口只把对象写进桶就返回键（`chat_service.py:216-243`），任何一张表里都
    找不到这把键——既没有对账入口，也没有任何一个回收入口能看见它。判据故意写成「全库
    任何一处」，不绑定具体的表名：绑了表名，就变成在考实现对不对得上，而不是考「服务端
    到底记不记得自己铸过这把键」。
    """
    response = _upload(api)

    assert response.status_code == 200
    minted_key = response.json()["object_key"]
    assert _key_appears_anywhere(api, minted_key), (
        "上传接口把对象写进桶之后库里不留任何痕迹：既无法对账，也没有回收路径能看见它"
    )


def test_upload_registers_a_pending_row_for_the_minted_key(api, oss_requests):
    response = _upload(api)

    assert response.status_code == 200
    minted_key = response.json()["object_key"]
    # 登记行用的必须是接口真正铸出来并写进桶的那把键，不是另算的一把。
    assert oss_requests.urls("PUT") == [f"https://{OSS_HOST}/{minted_key}"]

    rows = _pending_rows(api)
    assert [row.object_key for row in rows] == [minted_key]
    assert rows[0].user_id == api.alice.id


def test_failed_object_write_still_leaves_a_reclaimable_pending_row(api, oss_requests):
    """写对象失败时登记行必须还在——这是「先落库、后写对象」的顺序证明。

    登记行若写在 PUT 之后，一次失败的 PUT 就会留下「桶里可能有、库里没有」的对象，
    正是 #142 要堵的那个形状。反过来，PUT 超时的情况下服务端**并不知道**对象是否已经
    落到桶里，所以这条登记行不能顺手删掉，必须留给清扫任务去重试删除。
    """
    oss_requests.default_put_status = 500

    response = _upload(api)

    assert response.status_code == 500
    assert oss_requests.urls("PUT"), "上传路径没有真的尝试写对象"
    rows = _pending_rows(api)
    assert len(rows) == 1, "写对象失败后登记行丢了：对象可能已落桶却没有对账入口"
    # 先把键取成普通字符串再跑清扫。会话是生产口径（expire_on_commit=True），而清扫每处理
    # 一个对象都要提交一次：提交把这些 ORM 行标记为过期，等清扫把该行删掉之后再读
    # .object_key，属性刷新就会撞上「这一行已经不存在了」。用例要读的是自己记下的期望值，
    # 不该依赖那条行在清扫之后还活着。
    pending_key = rows[0].object_key

    report = chat_service.reclaim_orphan_chat_attachments(
        api.db, now=datetime.now() + timedelta(seconds=TTL_SECONDS + 1)
    )
    assert report["reclaimed"] == 1
    # 对象可能压根没写成功，DELETE 会拿到 404——「对象不存在」正是回收的目标状态。
    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{pending_key}"]
    assert _pending_rows(api) == []


# ---------------------------------------------------------------------------
# (b) 清扫任务：实际删除对象（而不是断引用）
# ---------------------------------------------------------------------------

def test_sweep_deletes_the_object_of_an_upload_that_was_never_sent(api, oss_requests):
    """核心用例：上传 → 一直不发送 → 过了保留窗口 → 对象被真的删掉。"""
    key = _upload(api).json()["object_key"]
    assert oss_requests.urls("DELETE") == []

    report = chat_service.reclaim_orphan_chat_attachments(
        api.db, now=datetime.now() + timedelta(seconds=TTL_SECONDS + 1)
    )

    assert report == {"candidates": 1, "reclaimed": 1, "failed": 0, "unclaimable": 0, "skipped": 0}
    # 危害是「公开读的对象永久留在桶里」，所以判据是 DELETE 请求本身，不是「断了引用」。
    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{key}"]
    assert _pending_rows(api) == []


def test_sweep_leaves_objects_inside_the_retention_window_alone(api, oss_requests):
    """护栏：窗口内的登记行不许动。

    上传与发送之间隔着用户打字、挑图、切页面的真实时间，清扫提前动手就会把一条马上
    要发出去的消息引用的对象删掉——对象存储没有回收站，那是不可逆的内容丢失。
    """
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS - 60)
    _add_pending_row(api, UPLOAD_B, age_seconds=TTL_SECONDS + 60)

    report = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert report["reclaimed"] == 1
    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{UPLOAD_B}"]
    assert [row.object_key for row in _pending_rows(api)] == [UPLOAD_A]


def test_sweep_is_idempotent_and_treats_a_missing_object_as_reclaimed(api, oss_requests):
    """重跑安全：登记行删除是幂等的，对象已经不在（404）也算回收成功。

    404 是 httpx 层面的「目标状态已满足」：清扫跑到一半崩掉后重跑、或两个进程同时清扫，
    都不该因为对象已经删掉了而报错并把行留在表里。
    """
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60)
    oss_requests.status_by_key[UPLOAD_A] = 404

    first = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())
    second = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert (first["reclaimed"], first["failed"]) == (1, 0)
    assert second == {"candidates": 0, "reclaimed": 0, "failed": 0, "unclaimable": 0, "skipped": 0}
    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{UPLOAD_A}"]


def test_sweep_batches_the_work_instead_of_walking_the_whole_backlog(api, oss_requests):
    """一次清扫最多处理 batch_limit 个对象，剩下的留给下一次。

    DELETE 是串行外呼，最坏每个要等到连接超时；不设上限时积压一大就再也回不到调用方，
    启动期挂后台线程也会一直占着。
    """
    for index in range(5):
        _add_pending_row(api, f"rag-chat/2026/09/21/{index:032x}.png", age_seconds=TTL_SECONDS + 60)

    first = chat_service.reclaim_orphan_chat_attachments(
        api.db, now=datetime.now(), batch_limit=2
    )
    second = chat_service.reclaim_orphan_chat_attachments(
        api.db, now=datetime.now(), batch_limit=10
    )

    assert first["reclaimed"] == 2
    assert second["reclaimed"] == 3
    assert len(_pending_rows(api)) == 0


def test_sweep_signs_a_real_oss_delete_request(api, monkeypatch):
    """回收请求本身要是 OSS 认得的 DeleteObject，而不是「随便发了个请求」。

    用真实 `httpx.Client` + 模拟传输层：请求对象、URL、头部构造都走真库代码，只换网络出口，
    并独立复算签名（替身自己算的签名不能作为证据）。
    """
    seen = []
    transport = httpx.MockTransport(lambda request: (seen.append(request), httpx.Response(204))[1])
    real_client_cls = httpx.Client

    for name, value in OSS_CONFIG.items():
        monkeypatch.setattr(oss_service, name, value)
    monkeypatch.setattr(oss_service.httpx, "Client", lambda **kwargs: real_client_cls(transport=transport, **kwargs))

    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60)

    assert chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())["reclaimed"] == 1

    request = seen[0]
    assert request.method == "DELETE"
    assert str(request.url) == f"https://{OSS_HOST}/{UPLOAD_A}"
    date = request.headers["Date"]
    expected = base64.b64encode(
        hmac.new(b"test-secret", f"DELETE\n\n\n{date}\n/demo/{UPLOAD_A}".encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")
    assert request.headers["Authorization"] == f"OSS test-id:{expected}"
    assert request.headers["Host"] == OSS_HOST
    # Date 头是「签发那一刻」的墙钟读数。这里再取一次 formatdate 做相等断言，就等于要求
    # 两次互不相关的读数恰好落在同一秒——跨过秒边界必假红。改为校验它是规范的 RFC-2822
    # GMT 时间（回读再格式化要逐字相同），且与当前时刻的偏差落在容忍窗口内。
    requested_at = parsedate_to_datetime(date)
    assert formatdate(requested_at.timestamp(), usegmt=True) == date
    assert abs(requested_at.timestamp() - time.time()) <= 5
    # 回收不该带上传时那条公开读 ACL 头：签名串里没有它，带上就对不上了。
    assert "x-oss-object-acl" not in request.headers
    assert "content-type" not in request.headers


def test_one_object_failing_does_not_stop_the_others_and_keeps_its_row(api, oss_requests, caplog):
    """单个对象删不掉：其余照删，失败的那条登记行留着等下次重扫，且不回显给用户。"""
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60)
    _add_pending_row(api, UPLOAD_B, age_seconds=TTL_SECONDS + 60)
    _add_pending_row(api, UPLOAD_C, age_seconds=TTL_SECONDS + 60)
    oss_requests.status_by_key[UPLOAD_B] = 403

    with caplog.at_level(logging.WARNING, logger="service.chat_service"):
        report = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert (report["reclaimed"], report["failed"]) == (2, 1)
    assert sorted(oss_requests.urls("DELETE")) == sorted(
        [f"https://{OSS_HOST}/{UPLOAD_A}", f"https://{OSS_HOST}/{UPLOAD_B}", f"https://{OSS_HOST}/{UPLOAD_C}"]
    )
    # 删不掉的行必须留着：删行等于把「桶里还有这个对象」这条唯一的线索也丢掉。
    assert [row.object_key for row in _pending_rows(api)] == [UPLOAD_B]
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any(UPLOAD_B in message for message in warnings)


def test_a_failed_delete_puts_the_row_back_with_its_original_timestamp(api, oss_requests):
    """放回队列时按**原时间戳**排队，不是 now()。

    这条语义只在「同一把键反复删失败」（对象被锁、连接抖动）时才看得见：放回时若盖上当前
    时刻，这条行每失败一轮就往后排一轮，最后饿死在新孤儿后面。放回也不该重新排队——这次
    上传并没有被回收，它只是这一轮没删成，仍然按上传时刻算超期。
    """
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 120)
    original = _pending_rows(api)[0].created_at
    oss_requests.status_by_key[UPLOAD_A] = 403

    report = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert report["failed"] == 1
    restored = _pending_rows(api)
    assert [row.object_key for row in restored] == [UPLOAD_A]
    assert restored[0].created_at == original, "放回时重写了时间戳：这条行被排到了队尾"


# ---------------------------------------------------------------------------
# (c) 发送成功 = 转为正式引用，此后归消息驱动的回收路径管
# ---------------------------------------------------------------------------

def test_sending_the_message_consumes_the_pending_row(api, monkeypatch, oss_requests):
    key = _upload(api).json()["object_key"]
    assert [row.object_key for row in _pending_rows(api)] == [key]

    cid = _run_stream_chat(api, monkeypatch, [{"object_key": key, "name": "a.png"}])

    stored = json.loads(api.db.query(Message).filter_by(conversation_id=cid).first().attachments)
    assert [item["object_key"] for item in stored] == [key]
    assert _pending_rows(api) == [], "消息已经落库，登记行却没被消费：清扫会把这条活消息的图删掉"


def test_a_sent_attachment_is_never_deleted_by_the_sweep(api, monkeypatch, oss_requests):
    """护栏：清扫只碰「没有任何消息引用」的对象，不得误删一条已发送消息的图。

    用远在保留窗口之外的 `now` 跑清扫，任何按时间下手的实现都会在这里下手；能不能删只
    取决于它是否已经转为正式引用。
    """
    key = _upload(api).json()["object_key"]
    cid = _run_stream_chat(api, monkeypatch, [{"object_key": key, "name": "a.png"}])
    oss_requests.requests.clear()

    report = chat_service.reclaim_orphan_chat_attachments(
        api.db, now=datetime.now() + timedelta(days=365)
    )

    assert report["reclaimed"] == 0
    assert oss_requests.urls("DELETE") == []
    # 它仍在既有回收路径的管辖内：删会话时才轮到它。
    assert api.client.delete(f"/api/chat/conversations/{cid}").status_code == 200
    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{key}"]


def test_the_sweep_never_deletes_an_object_whose_row_a_send_consumed_mid_sweep(api, monkeypatch, oss_requests):
    """并发臂（A/B 交错闭合）：清扫取完候选之后、动手之前，用户把这条附件发送成功。

    这是「按一份先前的快照动手」那版留下的残余风险：清扫若照着候选名单直接删，就会删掉一条
    已经落库消息引用的对象——用户看得见的图，扫一次就没了，而对象存储没有回收站。

    现在的顺序把判定收窄成「**每个对象动手之前先赢下那一行**」（条件删除 + 提交）：赢不下来
    就说明这条键已经成了正式引用（或已被另一条清扫领走），对象必须留着。

    交错点是确定性的，不靠线程调度碰运气：替身包住「领行」这一步，在真正的领行发生之前，先用
    发送侧那次真实的消费把登记行删掉——正是并发时会发生的事。断言的是外部可见结果：一条
    DELETE 都不能发出去，而那条消息仍然引用着这把键。
    """
    key = _upload(api).json()["object_key"]
    _add_conversation(api, "c-interleave")
    # 把登记行做旧，确保它一定是本次清扫的候选——否则交错臂的前提就不成立。
    pending = _pending_rows(api)[0]
    pending.created_at = datetime.now() - timedelta(seconds=TTL_SECONDS + 60)
    api.db.commit()
    oss_requests.requests.clear()

    real_claim = crud_chat.claim_attachment_upload

    def claim_after_the_send_lands(db, object_key, older_than):
        # 交错：消息行与登记行消费在**同一次提交**里落库（与 stream_chat 一致），
        # 且发生在清扫真正领走这一行之前。
        db.add(Message(
            conversation_id="c-interleave",
            role="user",
            content="这张图是什么",
            attachments=json.dumps([{"object_key": object_key, "name": "a.png"}], ensure_ascii=False),
        ))
        assert crud_chat.confirm_attachment_uploads(db, [object_key], api.alice_id) == 1, (
            "交错臂没摆成：这次「发送」并没有真的消费掉登记行"
        )
        db.commit()
        return real_claim(db, object_key, older_than)

    monkeypatch.setattr(crud_chat, "claim_attachment_upload", claim_after_the_send_lands)

    report = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert report["candidates"] == 1, "交错臂前提不成立：清扫没把这条登记行当成候选"
    assert report["skipped"] == 1
    assert report["reclaimed"] == 0
    assert oss_requests.urls("DELETE") == [], "清扫删掉了一条已落库消息引用的对象"
    stored = json.loads(api.db.query(Message).filter_by(conversation_id="c-interleave").first().attachments)
    assert [item["object_key"] for item in stored] == [key]


def test_a_zero_retention_window_does_not_sweep_an_in_flight_upload(api, oss_requests):
    """护栏：保留窗口被压到 0 时，正在上传的那条登记行不许被扫掉。

    窗口是配置项，TTL=0（或负数）会把「超期」退化成「比此刻更早的都算」——包括上传请求自己
    刚登记、对象还在写的那一行。删掉它，留下的正是一个「桶里可能有、库里没有」的对象，也就是
    这次修复自己要消灭的形态。实现里那条不可配置的下限（一分钟，远大于一次上传的耗时）就是
    为这个入口设的。
    """
    key = _upload(api).json()["object_key"]
    oss_requests.requests.clear()

    report = chat_service.reclaim_orphan_chat_attachments(api.db, ttl_seconds=0)

    assert report["reclaimed"] == 0
    assert oss_requests.urls("DELETE") == []
    assert [row.object_key for row in _pending_rows(api)] == [key]


def test_a_failed_send_leaves_the_upload_for_the_sweep(api, monkeypatch, oss_requests):
    """那一轮发送失败：图片识别失败时 `stream_chat` 在落用户消息之前就返回了。

    这条路径下没有消息引用这个对象，登记行必须原样留着，由清扫任务兜底。
    """
    key = _upload(api).json()["object_key"]
    _stub_stream_chat_dependencies(api, monkeypatch)

    async def failing_build_effective_question(question, attachments):
        return question, {"status": "failed", "error": "图片内容识别失败"}

    monkeypatch.setattr(chat_service, "_build_effective_question", failing_build_effective_question)

    response = asyncio.run(
        chat_service.stream_chat(
            ChatRequest(question="", attachments=[{"object_key": key, "name": "a.png"}]),
            authorization="Bearer token",
        )
    )
    asyncio.run(_collect_stream(response.body_iterator))

    assert api.db.query(Message).count() == 0
    assert [row.object_key for row in _pending_rows(api)] == [key]

    assert chat_service.reclaim_orphan_chat_attachments(
        api.db, now=datetime.now() + timedelta(seconds=TTL_SECONDS + 1)
    )["reclaimed"] == 1


def test_a_rolled_back_message_keeps_its_attachment_reclaimable(api, monkeypatch, oss_requests):
    """顺序护栏：登记行的消费与消息行必须在同一次提交里。

    提交失败时两条都不能留下半边——消息没落库而登记行被消费掉，对象就脱离了所有回收
    路径（清扫看不见它、消息驱动也看不见它）。
    """
    key = _upload(api).json()["object_key"]
    _add_conversation(api, "c-rollback")
    _stub_stream_chat_dependencies(api, monkeypatch)

    with monkeypatch.context() as patch:
        patch.setattr(chat_service, "SessionLocal", lambda: _FlushThenFailSession(api.db))
        with pytest.raises(RuntimeError):
            asyncio.run(
                chat_service.stream_chat(
                    ChatRequest(conversation_id="c-rollback", question="看看这张图",
                                attachments=[{"object_key": key, "name": "a.png"}]),
                    authorization="Bearer token",
                )
            )

    assert [row.object_key for row in _pending_rows(api)] == [key], "提交失败却消费了登记行"


def test_consuming_the_upload_never_commits_on_its_own(api, oss_requests):
    """契约：消费登记行必须留在调用方那次提交里，`confirm` 自己不许 commit。

    「消息行 + 消费」分开提交会冒出一个中间态：登记行已经删掉、消息行还没落库。发送正好在这条
    缝里失败（进程被杀、连接断），对象就既不在消息里、也不在登记表里——又回到这次修复要消灭的
    孤儿形状。上面那条回滚用例盖不住这一臂：它的失败替身对**每一次** commit 都抛，于是「消费
    先提交、消息后提交」里第一个提交就炸了，状态与不分开提交时不可区分。这里改从提交次数直接
    钉住契约——`confirm` 内部只要自己提交一次，计数就不再是 0。
    """
    key = _upload(api).json()["object_key"]

    session = _CountingSession(api.db)
    consumed = crud_chat.confirm_attachment_uploads(session, [key], api.alice_id)

    assert consumed == 1, "夹具没摆成：这条登记行本来就没被消费掉"
    assert session.commits == 0, "消费动作自己提交了：它就不再与消息行同一次提交"


# ---------------------------------------------------------------------------
# (d) 入口：启动期清扫 + 对账
# ---------------------------------------------------------------------------

def test_lifespan_schedules_the_orphan_sweep(monkeypatch):
    """启动期必须挂上清扫任务（issue #142 点名的缺口：lifespan 里没有任何清扫）。"""
    calls = []
    # 两条启动门禁都置空：本用例测的是清扫是否挂上，不测配置门禁，而它跑在测试环境里
    # （未配 MYSQL_PASSWORD），否则会死在门禁上而不是断言上。
    monkeypatch.setattr(main, "ensure_secret_key_configured", lambda: None)
    monkeypatch.setattr(main, "ensure_mysql_password_configured", lambda: None)
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "seed_default_users", lambda: None)
    monkeypatch.setattr(main, "schedule_orphan_attachment_sweep", lambda: calls.append("sweep"))

    async def _startup():
        async with main.lifespan(FastAPI()):
            pass

    asyncio.run(_startup())

    assert calls == ["sweep"]


def test_scheduled_sweep_runs_on_a_background_thread(api, monkeypatch, oss_requests):
    """清扫挂在后台守护线程上、不阻塞启动，而且**不止跑一轮**。

    两件事都要证：

    - 后台：删除是串行外呼，最坏每个对象要等到连接超时；放进启动路径会把就绪时间拖成分钟级。
    - 常驻：只扫启动那一轮是不够的——一个跑几个月不重启的进程永远等不到第二轮，用户上传后
      不发送的对象会一直攒在桶里，正是 issue #142 要消灭的形态。

    第二件事用 `batch_limit=1` 把它摆成可判定的：一轮只处理一个对象，于是第二把键必须由
    循环的**下一轮**回收。写成「跑一次就结束」的实现这里只会看到一条 DELETE。
    stop_event 在断言前先置位并 join，跑完不留活线程。
    """
    monkeypatch.setattr(chat_service, "SessionLocal", lambda: api.db)
    monkeypatch.setattr(chat_service, "CHAT_ATTACHMENT_SWEEP_BATCH_LIMIT", 1)
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60)
    _add_pending_row(api, UPLOAD_B, age_seconds=TTL_SECONDS + 60)

    stop = threading.Event()
    # interval 传 1 秒（实现另有 1 秒下限），两轮一共约 1 秒，远在下面的等待预算之内。
    thread = chat_service.schedule_orphan_attachment_sweep(interval_seconds=1.0, stop_event=stop)
    try:
        assert isinstance(thread, threading.Thread)
        assert thread.daemon, "非守护线程会在进程退出时把关闭流程挂住"
        deadline = time.monotonic() + 15
        while len(oss_requests.urls("DELETE")) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        # stop 置位后循环必须自己退出：join 回来还在跑就是关停语义没接上。
        stop.set()
        thread.join(timeout=15)

    assert not thread.is_alive(), "stop 置位后清扫循环仍不退出"
    # 线程里跑的必须是真清扫（不是「起了个线程什么都没做」），且两把键都删掉了。
    assert sorted(oss_requests.urls("DELETE")) == sorted(
        [f"https://{OSS_HOST}/{UPLOAD_A}", f"https://{OSS_HOST}/{UPLOAD_B}"]
    )
    assert _pending_rows(api) == []


def test_the_sweep_entry_point_never_raises_into_the_caller(api, monkeypatch, caplog):
    """清扫入口自己吞异常：它跑在启动路径/后台线程上，抛出去只会换成一个没人接的栈。"""
    def _boom(db, **_kwargs):
        raise RuntimeError("oss unreachable")

    monkeypatch.setattr(chat_service, "SessionLocal", lambda: api.db)
    monkeypatch.setattr(chat_service, "reclaim_orphan_chat_attachments", _boom)

    with caplog.at_level(logging.WARNING, logger="service.chat_service"):
        assert chat_service.run_orphan_attachment_sweep() is None

    assert any("oss unreachable" in record.getMessage() for record in caplog.records)


def test_pending_rows_are_the_reconciliation_entry_point(api, oss_requests):
    """「桶里有、库里没有」必须能列出来——这正是修复前完全缺失的那块。

    修复后上传即登记，于是这张表就是那个对账入口：行在表里 = 服务端铸过这把键、还没有任何
    消息引用它。
    """
    first = _upload(api).json()["object_key"]
    second = _upload(api).json()["object_key"]

    rows = _pending_rows(api)

    assert sorted(row.object_key for row in rows) == sorted([first, second])
    assert all(row.user_id == api.alice.id for row in rows)
    assert all(isinstance(row.created_at, datetime) for row in rows)


def test_reconciliation_can_be_scoped_to_one_user_and_one_window(api, oss_requests):
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60, user_id=api.alice.id)
    _add_pending_row(api, UPLOAD_B, age_seconds=0, user_id=api.bob.id)

    stale = crud_chat.list_pending_attachment_uploads(api.db, older_than=datetime.now() - timedelta(seconds=TTL_SECONDS))
    bobs = crud_chat.list_pending_attachment_uploads(api.db, user_id=api.bob.id)

    assert [row.object_key for row in stale] == [UPLOAD_A]
    assert [row.object_key for row in bobs] == [UPLOAD_B]


# ---------------------------------------------------------------------------
# (e) 破坏面护栏：新回收路径不许碰不属于本服务、或仍活着的对象
# ---------------------------------------------------------------------------

def test_sweep_never_signs_a_delete_for_a_key_this_service_never_minted(api, oss_requests, caplog):
    """登记的键仍然要过铸造形态护栏。

    登记行是服务端自己写的，理论上只可能是本服务铸的键；但回收动作是全仓唯一一处用服务端
    凭据签 DELETE 的地方，护栏必须在下手的那一刻再判一次——穿越键
    （`rag-chat/.../../../finance-archive/x`）会被 httpx 规范化成桶里另一个对象的 URL，
    只查前缀的实现在这里就会删错对象。
    """
    _add_pending_row(api, FOREIGN_UPLOAD, age_seconds=TTL_SECONDS + 60)
    _add_pending_row(api, NEAR_MISS_UPLOAD, age_seconds=TTL_SECONDS + 60)
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60)

    with caplog.at_level(logging.WARNING, logger="service.chat_service"):
        report = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{UPLOAD_A}"]
    assert report["reclaimed"] == 1
    # 留着是为了让该键的发送被拒（墓碑）；重复处理由 `reclaimed_at` 过滤挡住。
    assert report["unclaimable"] == 2
    assert _pending_rows(api) == []
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    for key in (FOREIGN_UPLOAD, NEAR_MISS_UPLOAD):
        assert any(key in message for message in warnings), f"没有为 {key!r} 留下可按对象对账的告警"


def test_sweep_does_not_touch_another_users_pending_upload(api, monkeypatch, oss_requests):
    """清扫是全量的、不按用户过滤，但删除的对象集合只由「已超期且没被消费」决定。

    bob 的上传还没超期时,alice 的超期对象被回收，bob 的不许动。
    """
    _add_pending_row(api, UPLOAD_A, age_seconds=TTL_SECONDS + 60, user_id=api.alice.id)
    _add_pending_row(api, UPLOAD_B, age_seconds=60, user_id=api.bob.id)

    report = chat_service.reclaim_orphan_chat_attachments(api.db, now=datetime.now())

    assert report["reclaimed"] == 1
    assert oss_requests.urls("DELETE") == [f"https://{OSS_HOST}/{UPLOAD_A}"]
    assert [row.object_key for row in _pending_rows(api)] == [UPLOAD_B]


def test_an_attachment_key_minted_by_someone_else_is_not_confirmed_by_my_send(api, monkeypatch, oss_requests):
    """发送只消费自己的登记行。

    附件列是客户端回带的自由 JSON，消息可以引用一把别人铸的键。若发送方也能把它消费掉，
    那把键就会一直被这条消息钉住，原主「上传了没发送」的对象再也回不到清扫任务手里。
    """
    _add_pending_row(api, UPLOAD_A, age_seconds=0, user_id=api.bob.id)

    cid = _run_stream_chat(api, monkeypatch, [{"object_key": UPLOAD_A, "name": "a.png"}])

    stored = json.loads(api.db.query(Message).filter_by(conversation_id=cid).first().attachments)
    assert [item["object_key"] for item in stored] == [UPLOAD_A]
    assert [row.object_key for row in _pending_rows(api)] == [UPLOAD_A], "别人的待确认行被这条消息消费了"


def test_no_http_route_can_delete_an_arbitrary_object_key(api):
    """回收面不在 HTTP 上：本轮没有新增任何「按 object_key 删除」的对外接口。

    清扫只认「库里登记过、且超过保留窗口」的键，是服务端单方面发起的动作；多一个接受
    客户端指定键的删除接口，就是给任何登录用户开一个借服务端 AK 删桶对象的入口。

    issue #256 的删消息接口同样不碰回收面：它以 message_id 为参数，object_key 由服务端
    从消息附件里推导、只登记待回收行，客户端仍然无法指定要删哪个对象。
    """
    chat_routes = [route for route in chat_router.router.routes]
    delete_routes = [
        route for route in chat_routes
        if "DELETE" in getattr(route, "methods", set())
    ]
    assert [route.path for route in delete_routes] == [
        "/api/chat/messages/{message_id}",
        "/api/chat/conversations/{cid}",
    ]


# ---------------------------------------------------------------------------
# 真实 stream_chat 的最小夹具（模型、检索、轨迹短路）
# ---------------------------------------------------------------------------

class _FakeChatTrace:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.trace_id = "trace-test"

    async def add(self, *_args, **_kwargs):
        pass

    async def attach(self, **_kwargs):
        pass

    async def finish(self, *_args, **_kwargs):
        pass

    def snapshot(self):
        return {"trace_id": self.trace_id, "events": []}


class _FlushThenFailSession:
    """把真实会话包一层：commit 先把待写语句 flush 进事务、再失败。

    先 flush 是关键：只让 commit 抛异常的话，登记行的 DELETE 根本没下发到数据库，
    用例会「因为什么都没发生」而通过，锁不住原子性。
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def commit(self):
        self._db.flush()
        raise RuntimeError("commit failed")


class _CountingSession:
    """包一层真实会话，只数 commit 次数，其余原样转发。

    用来钉「谁负责提交」这类契约：只看外部结果分不出「一次提交」与「两次提交」（后者在两次
    之间失败时才会露馅），而提交次数是直接可辨的。
    """

    def __init__(self, db):
        self._db = db
        self.commits = 0

    def __getattr__(self, name):
        return getattr(self._db, name)

    def commit(self):
        self.commits += 1
        return self._db.commit()


async def _collect_stream(iterator):
    chunks = []
    async for chunk in iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def _add_conversation(api, cid):
    api.db.add(Conversation(id=cid, user_id=api.alice_id, title=f"会话-{cid}"))
    api.db.commit()


def _stub_stream_chat_dependencies(api, monkeypatch):
    """把模型、检索、轨迹三块短路掉，只留落库与附件消费这段真实逻辑。"""
    # 用户名同样先取成普通字符串：这个 lambda 由被测代码在流开头调用，那时会话还开着、
    # 现读也读得到；但把「夹具持有的 ORM 实例」闭包进去，就等于让替身的可用性取决于
    # 会话还开不开着，没必要。
    alice_username = api.alice.username

    async def fake_build_effective_question(question, attachments):
        return question, {"status": "skipped"}

    async def fake_recent_memory_text(*_args, **_kwargs):
        return ""

    async def fake_decide_need_rag(*_args, **_kwargs):
        return {"need_rag": False, "route": "direct", "confidence": 1.0, "reason": "test"}

    async def fake_stream_rag_answer(*_args, **_kwargs):
        yield "回答"

    monkeypatch.setattr(
        chat_service,
        "resolve_knowledge_base",
        lambda db, knowledge_base_id, user_id: SimpleNamespace(id=1, name="kb", user_id=user_id),
    )
    monkeypatch.setattr(chat_service, "SessionLocal", lambda: api.db)
    monkeypatch.setattr(
        chat_service,
        "authenticate",
        lambda db, authorization: db.query(User).filter_by(username=alice_username).first(),
    )
    monkeypatch.setattr(chat_service, "TraceRecorder", _FakeChatTrace)
    monkeypatch.setattr(chat_service, "_build_effective_question", fake_build_effective_question)
    monkeypatch.setattr(chat_service, "_build_recent_memory_text", fake_recent_memory_text)
    monkeypatch.setattr(chat_service, "_build_memory_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        chat_service, "_build_memory_aware_retrieval_question", lambda question, memory_context: question
    )
    monkeypatch.setattr(chat_service, "decide_need_rag", fake_decide_need_rag)
    monkeypatch.setattr(chat_service, "stream_rag_answer", fake_stream_rag_answer)
    monkeypatch.setattr(chat_service, "_trace_sse_payloads", lambda trace: [])
    monkeypatch.setattr(chat_service, "_build_sources", lambda chunks: [])
    async def _noop_trace(*args, **kwargs):
        return None

    monkeypatch.setattr(chat_service, "_attach_grounding_trace", _noop_trace)
    monkeypatch.setattr(chat_service, "_safe_trace_attach", _noop_trace)
    monkeypatch.setattr(chat_service, "_safe_trace_finish", _noop_trace)
    monkeypatch.setattr(chat_service, "_schedule_memory_summary_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_service, "schedule_ragas_evaluation", lambda *args, **kwargs: None)


def _run_stream_chat(api, monkeypatch, attachments) -> str:
    """跑一次真实聊天流（模型、检索、轨迹短路），返回新建会话的 id。"""
    _stub_stream_chat_dependencies(api, monkeypatch)

    # 用取好的整数主键而不是 api.alice.id：`stream_chat` 在流结束时会 close() 掉夹具借给它的
    # 那个会话实例，之后 api.alice 已经脱管；而流里的 commit 又把它标记成过期，于是这里再读
    # .id 会触发一次没有会话可用的属性刷新，抛 DetachedInstanceError——失败会落在「建会话」
    # 这段夹具代码上，看起来像实现坏了。整数主键不随会话生命周期变化。
    alice_id = api.alice_id

    def _conversation_ids():
        return {cid for (cid,) in api.db.query(Conversation.id).filter_by(user_id=alice_id).all()}

    before = _conversation_ids()
    response = asyncio.run(
        chat_service.stream_chat(
            ChatRequest(question="这张图是什么", attachments=attachments),
            authorization="Bearer token",
        )
    )
    asyncio.run(_collect_stream(response.body_iterator))

    created = _conversation_ids() - before
    assert len(created) == 1, f"这次聊天没有新建出唯一一个会话：{created}"
    return created.pop()
