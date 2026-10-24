
import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Annotated, NamedTuple
from uuid import uuid4

from fastapi import Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database.checkpointer import delete_thread_checkpoints
from config import (
    CHAT_ATTACHMENT_LEASE_SECONDS,
    CHAT_ATTACHMENT_PENDING_TTL_SECONDS,
    CHAT_ATTACHMENT_SWEEP_BATCH_LIMIT,
    CHAT_ATTACHMENT_SWEEP_INTERVAL_SECONDS,
    CHAT_ATTACHMENT_TOMBSTONE_TTL_SECONDS,
    MEMORY_WINDOW_TURNS,
)
from crud import chat as crud_chat
from crud.pagination import LIST_DEFAULT_LIMIT, LIST_MAX_LIMIT
from database.session import SessionLocal, get_db
from rag.learning_trace import TraceRecorder, compact_trace_reference
from model.models import Conversation, Message, User, _new_id
from rag.ragas_eval import schedule_ragas_evaluation
from schema.schemas import ChatRequest, MessageFeedbackRequest, RenameRequest
from service import rate_limit
from service.auth_service import authenticate, get_current_user
from service.oss_service import (
    ForeignObjectKeyError,
    _delete_oss_object,
    _public_oss_url,
    _put_oss_object,
    is_service_minted_key,
)
from service.pagination_service import resolve_conversation_cursor, resolve_list_limit
from service.trace_service import _safe_trace_add, _safe_trace_attach, _safe_trace_finish, _trace_sse_payloads
from service.utils_service import (
    CHAT_ATTACHMENT_MAX_BYTES,
    _build_sources,
    _check_answer_grounding,
    _internal_error_detail,
    resolve_image_upload_type,
)
from service.knowledge_service import resolve_knowledge_base
from rag.memory_service import (
    _build_memory_aware_retrieval_question,
    _build_memory_context,
    _build_recent_memory_text,
    _schedule_memory_summary_update,
)
from rag.vision_service import _build_effective_question
from rag.milvus_client import embedding_backend_status, embedding_trace_status
from rag.chains import stream_rag_answer
from rag.llm import ANSWER_GENERATION_FAILED_MESSAGE
from rag.retrieval import decide_need_rag, retrieve_knowledge


logger = logging.getLogger(__name__)

# 失败分支回给用户的固定文案：异常原文（驱动报错、路径、上游地址）只进 logger，不进 SSE 帧
# 或 HTTP detail。学习轨迹事件同时带上 trace_id，用户报错时可直接与日志里的 trace 对上。
ASSISTANT_SAVE_FAILED_MESSAGE = "保存回答失败"
EMPTY_MODEL_RESPONSE_MESSAGE = "本次未生成回答，请重新发送"
RAGAS_SCHEDULE_FAILED_MESSAGE = "RAGAS 评估调度失败"
MEMORY_SUMMARY_SCHEDULE_FAILED_MESSAGE = "长期记忆压缩调度失败"
OSS_UPLOAD_FAILED_MESSAGE = "图片上传失败，请稍后重试"
# 这条键已被判为待回收（墓碑）时回给用户的文案（issue #238）。与其它失败分支一样，异常原文
# （键、时间戳、外呼错误）只进日志，不进 SSE 帧。
ATTACHMENT_RECLAIMED_MESSAGE = "附件已失效，请重新上传后再发送"

# 保留窗口的下限（不可配置）：一次上传从登记到写对象只需要秒级，把窗口压到它之下就会让
# 清扫去删一条正在写入的对象。见 reclaim_orphan_chat_attachments 的说明。
CHAT_ATTACHMENT_MIN_PENDING_TTL_SECONDS = 60


class AttachmentReclaimedError(Exception):
    """这次发送引用了一把**已经被清扫判为待回收**的键，消息不得落库（issue #238）。

    它不是参数错误、也不是「上传失败」，而是一次必须被用户看见的拒绝：清扫已经先一步把
    这把键判死（删除意图已提交），对象要么已经删掉、要么正在删。此时把消息落库，留下的
    就是一条引用不存在对象的活消息——「用户看到自己发出去了、点开图片是 404」，而且这条
    消息此后会被删除路径当成归属凭据，让一把已删除的键继续有资格签发 DELETE。

    刻意做成**独立异常**而不是把 `_save_user_message` 的返回值改成布尔：调用方（`stream_chat`）
    必须显式处理这条路径，而不能靠「忘了看返回值」蒙混过去——拒签在这里是**非静默**的。

    构造参数只带 object_key：`str(exc)` 会进日志，键本身不含敏感信息，但不带上 user_id
    与时间戳——它可能被上游当成 detail 回显，没有必要。
    """

    def __init__(self, object_key: str):
        self.object_key = object_key
        super().__init__(f"attachment object_key has been reclaimed: {object_key}")


def _serialize_conversation(conv: Conversation, user_id: int) -> dict:
    # 历史遗留的跨用户绑定不对外暴露，避免泄露他人知识库
    knowledge_base = (
        conv.knowledge_base
        if conv.knowledge_base and conv.knowledge_base.user_id == user_id
        else None
    )
    return {
        "id": conv.id,
        "title": conv.title,
        "knowledge_base_id": knowledge_base.id if knowledge_base else None,
        "knowledge_base_name": knowledge_base.name if knowledge_base else "",
        "created_at": conv.created_at.isoformat() if conv.created_at else "",
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
    }


def list_conversations(
    limit: Annotated[int, Query(ge=1, le=LIST_MAX_LIMIT)] = LIST_DEFAULT_LIMIT,
    before_updated_at: Annotated[datetime | None, Query()] = None,
    before_id: Annotated[str | None, Query()] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """会话列表，按「最近活动」倒序分页返回。

    游标是 (before_updated_at, before_id) 复合键，两个参数要么都不给（最新一页）、
    要么都给；只给一个会被 resolve_conversation_cursor 判 422。取值直接回带上一次
    响应里最后一条的 updated_at 与 id，语义与消息接口的 before_id 同族（键集游标，
    不是 offset）。不传时返回最新一页，页大小默认 LIST_DEFAULT_LIMIT。
    """
    cursor = resolve_conversation_cursor(before_updated_at, before_id)
    rows = crud_chat.list_conversations(
        db,
        user.id,
        limit=resolve_list_limit(limit),
        before=cursor,
    )
    return [_serialize_conversation(c, user.id) for c in rows]


def resolve_message_limit(limit: int | None) -> int:
    """页大小兜底校验。

    HTTP 请求已由 Query(ge/le) 拦截，这里是为了让直接调用 service 的路径（脚本、内部调用）
    也拿不到超过上限的页大小，避免绕过接口层把整段历史一次读出。
    """
    if limit is None:
        return crud_chat.CHAT_MESSAGE_DEFAULT_LIMIT
    if limit < 1 or limit > crud_chat.CHAT_MESSAGE_MAX_LIMIT:
        raise HTTPException(422, f"limit 必须在 1 到 {crud_chat.CHAT_MESSAGE_MAX_LIMIT} 之间")
    return limit


def get_messages(cid: str,
                 limit: Annotated[int, Query(ge=1, le=crud_chat.CHAT_MESSAGE_MAX_LIMIT)] = crud_chat.CHAT_MESSAGE_DEFAULT_LIMIT,
                 before_id: Annotated[int | None, Query(ge=1)] = None,
                 user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """会话消息历史，按页返回。

    before_id 是游标：只取 id 小于它的消息（更早的一页）。不传时返回最新一页，
    页大小默认 CHAT_MESSAGE_DEFAULT_LIMIT，保证单次响应体有上限。
    """
    # 归属校验在 list_messages 里随分页查询一起做，这里不再单独查一次会话，避免重复查询。
    rows = crud_chat.list_messages(
        db,
        cid,
        user.id,
        limit=resolve_message_limit(limit),
        before_id=before_id,
    )
    if rows is None:
        raise HTTPException(404, "对话不存在")
    return [crud_chat.serialize_message(m) for m in rows]


def _service_minted_attachments(attachments: list[dict] | None, conversation_id: str) -> list[dict]:
    """只留下带「本服务铸造的键」的附件条目，其余整条丢弃并记 warning。

    附件列是客户端回带的：`/api/chat/stream` 的 body 里写什么就落什么。这个键后来会成为
    删除会话时服务端凭据签发 DeleteObject 的目标，所以在**写入点**也按铸造形态过滤一次，
    库里就不再有本服务没写过的键——删除侧另有一道同样的护栏，两道各自独立，都不可省。

    丢弃而不是回 422：整条请求失败会让「键不合规」的用户连消息都发不出去，而这类条目对
    应用没有意义（前端只回带上传接口的返回值，正常条目必带本服务的键）。被丢弃的键只进
    服务端日志，便于按会话对账：合法请求这条日志恒不出现。
    """
    accepted: list[dict] = []
    rejected: list[object] = []
    for item in attachments or []:
        if isinstance(item, dict) and is_service_minted_key(item.get("object_key")):
            accepted.append(item)
        else:
            rejected.append(item.get("object_key") if isinstance(item, dict) else item)
    if rejected:
        logger.warning(
            "chat attachments dropped: %d of %d entries carried no service-minted object_key "
            "(conversation_id=%s rejected_keys=%s)",
            len(rejected),
            len(attachments or []),
            conversation_id,
            [str(key)[:120] for key in rejected[:5]],
        )
    return accepted


def _reclaim_chat_attachments(db: Session, attachment_owners: list[tuple[str, int | None]],
                              requester_id: int) -> None:
    """尽力回收会话里的聊天附件对象，任何一个删不掉都不影响会话删除的结果。

    顺序是「先落库、后回收」：数据库是权威，最坏情况是 OSS 上多留一个孤儿对象（占空间、
    可重跑、可对账）；反过来先删对象再删行，一旦删行失败，用户会看到一个仍然存在的会话里
    图片全部失效——对象存储没有回收站，那是不可逆的内容丢失，比泄漏一个对象严重得多。

    归属判据（issue #233）在这里，而且是**先于形态护栏**的第一道闸：一把键只有
    `chat_attachment_uploads` 里的属主与请求方是同一个人，才轮到 `_delete_oss_object`
    去签发 DELETE。附件列由客户端回带（`schema/schemas.py` 的 `attachments: list[dict]`
    不校验），「这把键出现在我的会话消息里」不构成归属凭据——拿它当凭据，任何登录用户都
    能在自己的会话里写上别人的键，借服务的凭据删掉别人的对象。

    没有登记行（属主为 None，例如历史数据或客户端编的键）与属主是别人，都按「拒签」处理：
    只留一条带 object_key 的 warning，**不回显给用户**——会话已经删掉了，把「有一把键没删」
    写进响应体只会泄露别人对象的存在，且用户对此无能为力。

    删除失败（含被形态护栏拒签的 ForeignObjectKeyError）同样只记 warning 不上抛：会话此
    时已经删掉了，再把回收失败变成 5xx 只会让用户以为没删成功而去重试。日志带上 object_key，
    便于按对象对账后重跑（删除本身是幂等的）。

    回收成功的键在这里、也只在**删除确实没抛异常**之后释放登记行：以「删成功」为准逐键
    释放，失败（含 404 之外的状态码）的行留着，它就是这条泄漏唯一的对账线索。释放交给
    crud 时仍然是逐键的，理由见 crud_chat.delete_attachment_uploads。
    """
    reclaimed_keys: list[str] = []
    for object_key, owner_id in attachment_owners:
        if owner_id != requester_id:
            logger.warning(
                "chat attachment reclaim skipped, not the owner: object_key=%s owner_id=%s requester_id=%s",
                object_key,
                owner_id,
                requester_id,
            )
            continue
        try:
            _delete_oss_object(object_key)
        except Exception as exc:
            logger.warning(
                "chat attachment reclaim failed: object_key=%s error=%s",
                object_key,
                exc,
                exc_info=exc,
            )
            continue
        reclaimed_keys.append(object_key)
    if reclaimed_keys:
        try:
            crud_chat.delete_attachment_uploads(db, reclaimed_keys, requester_id)
        except Exception as exc:
            # 会话已删、对象已删，这里唯一失败的是账本落账。不让它把 200 变成 5xx：
            # 行留下即对账凭据（crud/chat.py:599-601），与 :223-225 的既定方针一致。
            logger.warning("chat attachment reclaim: releasing registration rows failed: "
                           "keys=%d error=%s", len(reclaimed_keys), exc, exc_info=exc)


def reclaim_orphan_chat_attachments(db: Session, *, now: datetime | None = None,
                                    ttl_seconds: int | None = None,
                                    batch_limit: int | None = None) -> dict:
    """回收「上传了但从未被发送」的附件对象，返回这次的处置统计。

    这是 issue #142 补的那条路径。删会话驱动的 `_reclaim_chat_attachments` 只看得见消息里
    引用到的键，未发送的对象不在任何消息里，必须由**登记表**驱动：上传时先落一条「待确认」
    行，发送成功时被消费掉，超过保留窗口还没被消费的才轮到删对象。

    四条顺序/边界，每条都对应一种不可逆的损失：

    ① **逐行「先领行、再删对象」**，而不是按一份先前的快照直接动手。发送侧的消费与清扫侧的
       领取是同一行的条件 UPDATE，数据库保证只有一个能赢；领不到就跳过，不去碰对象——否则
       「清扫取到候选之后、动手之前用户正好发送成功」会删掉一条已落库消息引用的对象。
       领行把这条判定从「整批一份快照」收窄到「每个对象各判一次」，窗口从分钟级降到一次
       外呼的时间。
    ② **领行写下的是墓碑（`claimed_at` 与 `reclaimed_at` 同一次写入、且当场提交），删对象
       失败时把它双清回 pending**（`clear_attachment_claim`）。@238 之前是「领行 = 条件删行 +
       失败时按原时间戳重插」，那让「清扫赢」与「对象已删」之间没有任何可判读的中间态：
       行没了，发送侧无从知道这把键已经被判死，一条并发到达的消息会照常落库并引用一个正被
       删除的对象。现在判死先于外呼落库，失败再把墓碑撤回（对象确实还在桶里，放行才是对的），
       方向仍是「宁可多留一轮，不可删错」。
       与 ① 一起，删对象**只在拿到墓碑之后**发生；墓碑留在行上直到删除成功
       （`mark_attachment_reclaimed` 只清租约、保留墓碑）——**留痕是刻意的**：它就是发送侧
       拒签这把键的唯一依据。
    ③ **删除前仍然过一遍铸造形态护栏**（`_delete_oss_object` 自己会拒）。登记行是服务端
       写的、理论上只可能是本服务铸的键，但这里是全仓唯一一处用服务端凭据签 DELETE 的
       地方，护栏必须在下手那一刻再判一次：`rag-chat/.../../../finance-archive/x` 这种键会
       被 httpx 规范化成桶里另一个对象的 URL。判不出来的行（`ForeignObjectKeyError`）永远
       签不出 DELETE，**同样墓碑化**（不再丢弃）：这键这次签不出，不代表它没有被别的路径
       引用，把行删掉会让一把本该被拒的键悄悄回到「无行 ⇒ 合法」分支；墓碑同样由 GC 按保留期
       收走，并留一条可按对象对账的 warning。
    ④ **失败只记 warning、不上抛**。单对象删不掉（403/网络）时其余对象照删，失败的那行
       双清回 pending 留给下一轮。`_ensure_oss_config` 抛的 HTTPException 也走这条：一次没配
       OSS 的进程不该把登记行清空。
    ⑤ **候选有两路**：pending（`created_at` 超期且 `reclaimed_at IS NULL`，即 ①-④ 处理的那批）
       与**租约超时的墓碑**（`reclaimed_at IS NOT NULL AND claimed_at < now - 租约`）。后者是
       ①「先领行」的收尾：上一轮领走一行、外呼还没回来进程就没了（崩溃/被杀），这行会永远
       停在「正在删」而没有任何路径再碰它——对象既不在桶里被重试删、也不在库里被结算，成了
       「桶里有、库里没有」的隐形孤儿。R 路把它重领一次（只刷新租约、墓碑不动）并重试外呼。
    ⑥ **末尾按保留期清理已落定的墓碑**（`gc_reclaimed_attachment_uploads`，`claimed_at IS NULL`
       的才算落定）。墓碑的作用有时效：过了保留期那条迟到消息早就不可能来了，行留着只是占表。
       `CHAT_ATTACHMENT_TOMBSTONE_TTL_SECONDS <= 0` 时整段跳过（关闭开关）——代价是这张表
       只增不减，换来「删除记录永不丢失」。
    """
    ttl = CHAT_ATTACHMENT_PENDING_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    # 保留窗口的下限：TTL 配成 0（或负数）时窗口会退化成「比此刻更早的都算超期」，那会把
    # 一条**上传请求自己刚登记、对象还在写**的行也扫进来，删完留下一个桶里有、库里没有的
    # 对象——正是 #142 的形态，而且是这次修复自己造出来的。一分钟远大于一次上传的耗时。
    ttl = max(CHAT_ATTACHMENT_MIN_PENDING_TTL_SECONDS, ttl)
    limit = CHAT_ATTACHMENT_SWEEP_BATCH_LIMIT if batch_limit is None else batch_limit
    moment = now or datetime.now()
    cutoff = moment - timedelta(seconds=ttl)

    # 先把候选键取成普通字符串列表，之后每处理一行都要提交：会话是 expire_on_commit=True 的，
    # 留着 ORM 行对象会在下一次读属性时触发刷新，而并发的另一条清扫可能已经把那一行改掉了，
    # 刷新会抛 ObjectDeletedError 把整批打断——后面的候选一个都处理不到。
    # 两路候选按定义互斥（P 要求 reclaimed_at 为空，R 要求非空），不会重复处理同一行。
    pending_keys = [
        row.object_key
        for row in crud_chat.list_pending_attachment_uploads(db, older_than=cutoff, limit=limit)
    ]
    stale_keys = crud_chat.list_stale_reclaimed_attachment_uploads(
        db, now=moment, lease_seconds=CHAT_ATTACHMENT_LEASE_SECONDS, limit=limit
    )
    stale_key_set = set(stale_keys)
    candidates = pending_keys + stale_keys

    reclaimed = 0
    failed = 0
    unclaimable = 0
    skipped = 0
    for object_key in candidates:
        if object_key in stale_key_set:
            # R 路：只刷新租约，墓碑不动——外呼成功之后这一行仍是墓碑（留着让发送被拒）。
            if not crud_chat.claim_stale_attachment_upload(
                db, object_key, lease_seconds=CHAT_ATTACHMENT_LEASE_SECONDS
            ):
                # 已经被另一条清扫重领、或这一行在两次读之间被结算了：不碰对象。
                skipped += 1
                continue
        else:
            # 返回值只是给日志/计数用的上下文，这一轮不需要，只关心「领到没有」。
            if crud_chat.claim_attachment_upload(db, object_key, cutoff) is None:
                # 已经被发送消费（或另一条清扫领走）：这个对象已经归别人管，绝不能删。
                skipped += 1
                continue
        try:
            _delete_oss_object(object_key)
        except ForeignObjectKeyError as exc:
            logger.warning(
                "chat attachment orphan sweep: refusing to delete an object key this service "
                "never minted, keeping the tombstone so this key stays refused: "
                "object_key=%s error=%s",
                object_key,
                exc,
            )
            crud_chat.mark_attachment_reclaimed(db, object_key)
            unclaimable += 1
            continue
        except Exception as exc:
            logger.warning(
                "chat attachment orphan reclaim failed, clearing the tombstone back to pending "
                "for the next sweep: object_key=%s error=%s",
                object_key,
                exc,
                exc_info=exc,
            )
            crud_chat.clear_attachment_claim(db, object_key)
            failed += 1
            continue
        # 删成功：行留下当墓碑（这一把键的发送从此被拒），只把租约清掉——这一次领取已经用完了。
        crud_chat.mark_attachment_reclaimed(db, object_key)
        reclaimed += 1

    # 墓碑 GC（⑥）。开关置 0 表示关闭：一行都不删。用与本轮同一个时刻算保留窗口——
    # 一次清扫只认一个时钟，跟候选的 cutoff 同源，测试与排查时才能整体把时间拨到某一刻。
    tombstone_ttl = CHAT_ATTACHMENT_TOMBSTONE_TTL_SECONDS
    if tombstone_ttl > 0:
        crud_chat.gc_reclaimed_attachment_uploads(
            db,
            older_than=moment - timedelta(seconds=tombstone_ttl),
            limit=limit,
        )

    return {"candidates": len(candidates), "reclaimed": reclaimed, "failed": failed,
            "unclaimable": unclaimable, "skipped": skipped}


def run_orphan_attachment_sweep() -> dict | None:
    """后台/启动期入口：自己开会话、自己吞异常，绝不把失败带回调用方。

    跑在启动路径或守护线程上，抛出去只会变成一个没人接的栈：清扫是尽力而为的维护动作，
    漏扫一轮的代价是对象多留一会儿，而启动失败是服务起不来。
    """
    db = SessionLocal()
    try:
        report = reclaim_orphan_chat_attachments(db)
        if report["candidates"]:
            logger.info(
                "chat attachment orphan sweep: candidates=%d reclaimed=%d failed=%d "
                "unclaimable=%d skipped=%d",
                report["candidates"], report["reclaimed"], report["failed"],
                report["unclaimable"], report["skipped"],
            )
        return report
    except Exception as exc:
        logger.warning("chat attachment orphan sweep failed: %s", exc, exc_info=exc)
        return None
    finally:
        db.close()


def sweep_orphan_attachments_forever(stop: threading.Event, interval_seconds: float) -> None:
    """先扫一轮，之后每 interval_seconds 再扫一轮，直到 stop 被置位。

    **首轮跑在循环里**（而不是只在启动时扫一次）是必须的：只扫启动那一轮时，一个跑几个月
    不重启的进程永远等不到下一轮，用户上传后不发送的对象会一直攒在桶里——那正是 issue #142
    要消灭的形态，不能因为「服务一直没重启」就复活。
    """
    while True:
        run_orphan_attachment_sweep()
        if stop.wait(interval_seconds):
            return


def schedule_orphan_attachment_sweep(interval_seconds: float | None = None,
                                     stop_event: threading.Event | None = None) -> threading.Thread:
    """把清扫循环挂到后台守护线程上，返回该线程（不阻塞启动）。

    不进启动路径同步跑：DELETE 是串行外呼、单个最坏要等到连接超时，积压一批就能把就绪
    时间拖成分钟级。守护线程是为了不在进程退出时被它挂住——清扫随时可以中断，重跑幂等。
    多个 worker 各自起一条也无妨：领行是条件删除、对象删除幂等，两条清扫不会互相删错。

    stop_event 只给用例用（跑完把线程停干净）；生产路径不传，循环随守护线程一起结束。
    """
    interval = CHAT_ATTACHMENT_SWEEP_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    thread = threading.Thread(
        target=sweep_orphan_attachments_forever,
        args=(stop_event or threading.Event(), max(1.0, interval)),
        name="chat-attachment-sweep",
        daemon=True,
    )
    thread.start()
    return thread


def delete_conversation(cid: str, user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    # 对象键必须在删行之前取：messages 随会话级联删除，删完就再也读不到引用了哪些对象。
    # 连属主一起取：回收判据是「谁铸的键」而不是「谁的消息里出现过它」，见
    # _reclaim_chat_attachments 与 crud_chat.list_conversation_attachment_owners。
    attachment_owners = crud_chat.list_conversation_attachment_owners(db, cid, user.id)
    if attachment_owners is None:
        # 存在性/归属闸（issue #260）：与 crud_chat.delete_conversation 用的是同一个
        # get_conversation 判据，所以 404 的结果集合不变，只是提前到任何写之前返回。
        # 闸门同时保证下面 attachment_owners 非 None——「去掉 or []」与它同进同退。
        raise HTTPException(404, "对话不存在")
    # 清理前置（issue #260）：checkpointer 的清理必须落在「点无可退」之前。
    # crud_chat.delete_conversation 里那一次 commit 之后，会话与消息已成事实；此时若
    # 清理抛错，请求 5xx 但状态已变——登记行停在 consumed、对象永不回收（#260）。
    # 放在 commit 之前，清理失败 = 这次删除整体没发生，用户重试是一次完整的重试。
    delete_thread_checkpoints(cid)
    conv = crud_chat.delete_conversation(db, cid, user.id)
    if not conv:
        # 并发删除兜底：另一方已经删掉了。上面的清理是幂等的（DELETE ... WHERE thread_id = ?），
        # 这里按 404 收尾，对外语义与「会话不存在」一致。
        raise HTTPException(404, "对话不存在")
    _reclaim_chat_attachments(db, attachment_owners, user.id)   # 闸门已保证非 None，不再需要 `or []`
    return {"message": "ok"}


def delete_message(message_id: int, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """删除单条消息（issue #256）；截断/单条语义与联动清理都在 CRUD 的同一次事务里完成。"""
    deleted_ids = crud_chat.delete_message(db, message_id, user.id)
    if deleted_ids is None:
        # 查无此消息与「消息属于别人」共用同一分支：归属校验随查询完成，不区分两者。
        raise HTTPException(404, "消息不存在")
    return {"message": "ok", "deleted_message_ids": deleted_ids}


def rename_conversation(cid: str, body: RenameRequest, user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "会话标题不能为空")
    conv = crud_chat.rename_conversation(db, cid, user.id, title)
    if not conv:
        raise HTTPException(404, "对话不存在")
    return {"message": "ok"}


def submit_message_feedback(message_id: int, body: MessageFeedbackRequest,
                            user: User = Depends(get_current_user),
                            db: Session = Depends(get_db)):
    # 归属校验随查询一起做：查无此消息、或消息不属于当前用户，一律 404（不回显正文）。
    message = crud_chat.get_message_by_user(db, message_id, user.id)
    if not message:
        raise HTTPException(404, "消息不存在")
    # 反馈只针对「回答」：对提问点赞没有意义，前端不给入口，服务端兜底拦下。
    if message.role != "assistant":
        raise HTTPException(422, "只能对助手消息反馈")
    crud_chat.set_message_feedback(db, message, body.feedback)
    return {"message_id": message.id, "feedback": message.feedback}


async def _attach_grounding_trace(
    retrieval_trace: dict,
    trace: TraceRecorder,
    *,
    answer: str,
    retrieved_contexts: list[str],
    need_rag: bool,
) -> None:
    if not need_rag:
        retrieval_trace["grounding"] = {"status": "skipped", "reason": "direct mode"}
        return

    grounding = _check_answer_grounding(answer, retrieved_contexts)
    retrieval_trace["grounding"] = grounding
    await _safe_trace_add(
        trace,
        "grounding_checked",
        "_check_answer_grounding",
        uses={
            "answer_chars": len(answer),
            "retrieved_contexts_count": len(retrieved_contexts),
        },
        creates={"grounding": grounding},
        result={
            "status": grounding.get("status", ""),
            "unsupported_count": grounding.get("unsupported_count", 0),
        },
        note="系统对最终回答做轻量证据校验，并把结果写入 retrieval_trace.grounding。",
    )


async def upload_chat_attachment(file: UploadFile = File(...),
                                 user: User = Depends(get_current_user),
                                 db: Session = Depends(get_db)):
    image_type = resolve_image_upload_type(
        file.content_type,
        file.filename,
        allow_filename_fallback=True,
    )
    if not image_type:
        raise HTTPException(400, "仅支持 png、jpg、jpeg、webp 图片")
    content_type, ext = image_type

    content = await file.read()
    if len(content) > CHAT_ATTACHMENT_MAX_BYTES:
        raise HTTPException(400, "图片不能超过 5MB")

    object_key = f"rag-chat/{datetime.now().strftime('%Y/%m/%d')}/{uuid4().hex}{ext}"
    # 先落库、后写对象（issue #142）：顺序反过来时，一次「对象已经进桶、进程随即退出」就
    # 留下一个库里没有任何痕迹的对象，既无法对账也没有任何回收入口能看见它——而它的 ACL
    # 是 public-read。先登记之后，写失败最坏只是一条指向不存在对象的行，清扫任务顺手抹掉。
    #
    # 写失败时**不删**这条登记行：PUT 超时的那个分支里，服务端并不知道对象到底有没有落桶，
    # 留着才能让清扫任务去重试删除；顺手删行等于把「桶里可能有这个对象」的唯一线索丢掉。
    crud_chat.register_attachment_upload(db, object_key, user.id)
    try:
        await _put_oss_object(object_key, content, content_type)
    except Exception as exc:
        logger.warning(
            "chat attachment object write failed, pending row kept for the sweep: object_key=%s",
            object_key,
            exc_info=exc,
        )
        raise HTTPException(500, _internal_error_detail(OSS_UPLOAD_FAILED_MESSAGE, "oss_upload", exc))

    return {
        "name": file.filename or f"image{ext}",
        "size": len(content),
        "content_type": content_type,
        "object_key": object_key,
        "url": _public_oss_url(object_key),
    }


class _RequestIdentity(NamedTuple):
    """鉴权结果的标量副本：ORM 行连同它的 Session 都不离开工作线程。"""

    user_id: int
    username: str


class _KnowledgeBaseRef(NamedTuple):
    """知识库的 (id, name) 标量副本。"""

    id: int
    name: str


class _ConversationState(NamedTuple):
    """`stream_chat` 后续步骤真正会用到的会话字段。"""

    id: str
    title: str
    knowledge_base: _KnowledgeBaseRef
    memory_summary: str
    memory_summary_upto_message_id: int
    created: bool


def _authenticate_request(authorization: str) -> _RequestIdentity:
    """整条鉴权链（验签 → 回查用户 → 校验世代 → 查吊销登记）在一个工作线程里跑完。

    `stream_chat` 早先自己开一个请求级 Session 并用到底，于是这些同步 DB 调用全落在事件
    循环线程上（issue #187）。改成每一段同步 DB 工作各自「开 session → 用 → 关」整段留在
    同一个工作线程里：Session 不跨线程传递，也不跨 await 长期持有。

    鉴权失败时 HTTPException 照常上抛，`finally` 保证连接先还回池——这正是原先那个
    `except: db.close(); raise` 要保证的事。
    """
    db = SessionLocal()
    try:
        user = authenticate(db, authorization)
        return _RequestIdentity(user.id, user.username)
    finally:
        db.close()


def _resolve_request_knowledge_base(knowledge_base_id: int | None, user_id: int) -> _KnowledgeBaseRef:
    """解析本次请求使用的知识库；解析不出来时抛 HTTPException(404)，语义与原先一致。"""
    db = SessionLocal()
    try:
        knowledge_base = resolve_knowledge_base(db, knowledge_base_id, user_id)
        return _KnowledgeBaseRef(knowledge_base.id, knowledge_base.name)
    finally:
        db.close()


def _load_or_create_conversation(
    cid: str | None,
    user_id: int,
    requested_knowledge_base: _KnowledgeBaseRef,
    title: str,
) -> _ConversationState:
    """读会话（没有就建一条），并把后续步骤要用的字段一次复制成标量。

    「只复用当前用户自己的知识库绑定，遗留的跨用户绑定回退到本次解析结果」这条判定也放在
    这里：它要读 `conversation.knowledge_base`，而那是**懒加载**关系。放在协程里读会变成
    事件循环线程上的一次隐式 SELECT，放到线程外读又会撞 DetachedInstanceError——只有在
    会话还活着的这个工作线程里读，才是既有语义又不额外欠一次查询。
    """
    db = SessionLocal()
    try:
        conversation = db.query(Conversation).filter_by(id=cid, user_id=user_id).first() if cid else None
        created = conversation is None
        if created:
            conversation = Conversation(
                id=_new_id(),
                user_id=user_id,
                knowledge_base_id=requested_knowledge_base.id,
                title=title,
            )
            db.add(conversation)
            db.commit()

        knowledge_base = requested_knowledge_base
        if not created:
            bound = conversation.knowledge_base
            if bound and bound.user_id == user_id:
                knowledge_base = _KnowledgeBaseRef(bound.id, bound.name)

        return _ConversationState(
            id=conversation.id,
            title=conversation.title,
            knowledge_base=knowledge_base,
            memory_summary=conversation.memory_summary or "",
            memory_summary_upto_message_id=conversation.memory_summary_upto_message_id or 0,
            created=created,
        )
    finally:
        db.close()


def _save_user_message(cid: str, user_id: int, display_question: str,
                       accepted_attachments: list[dict]) -> int:
    """写入用户消息并返回它的 id；引用了已被清扫判死的键时抛 `AttachmentReclaimedError`。

    拒签的判据是**这次消费的 rowcount**（issue #238，M5）：`confirm_attachment_uploads` 命中
    的行数少于本次提交的**去重键数**，就说明至少有一把键没能被消费掉。这是唯一从「写入是否
    真的发生」读出来的信号——先查一遍再写（读-再-写）会在两次快照之间漏掉变化，而这里要
    挡的恰是「清扫刚刚赢下这一行」这种瞬时窗口。

    少消费了并不直接等于「被拒」：早于 #142 的键本来就没有登记行，消费不到是正常的。
    因此只在这时补一次**同事务**的分类——`list_blocked_attachment_uploads` 挑出其中带墓碑的
    那些（键域，跨用户同样成立）。有墓碑 ⇒ 这条键的对象正在（或已经）被删，消息必须不落库；
    没有 ⇒ 老键形态，照常发送。

    分类必须与消费在同一次提交之前：`confirm` 不提交，读到墓碑后抛异常，`finally` 里的
    `close()` 把这次事务整个丢弃，消费与消息行一起回滚——落不下「消息已落库、对象被删」
    的形态。返回的键只用于日志，不随 SSE 帧发给客户端。
    """
    db = SessionLocal()
    try:
        user_message = Message(
            conversation_id=cid,
            role="user",
            content=display_question,
            attachments=json.dumps(accepted_attachments, ensure_ascii=False),
        )
        db.add(user_message)
        # 附件从「待确认」转为「正式引用」：这次消费必须与消息行同一次提交。两者分开时，
        # 「消息已落库、登记行还在」的中间态会让清扫任务把这条活消息引用的对象当成孤儿
        # 删掉（对象存储没有回收站），反过来则是消息没落库却把对象永久钉在登记表里。
        # 早于本次修复就存在的键没有登记行，消费不到是正常的——它们由消息驱动那条既有的
        # 回收路径负责，与这里无关。
        requested_keys = [
            key
            for key in (item.get("object_key") for item in accepted_attachments)
            if isinstance(key, str) and key
        ]
        confirmed = crud_chat.confirm_attachment_uploads(db, requested_keys, user_id)
        if confirmed < len(set(requested_keys)):
            # 同事务、提交前的分类：把「这把键已经判死」从「这颗键本来就没有登记行」里摘出来。
            blocked = crud_chat.list_blocked_attachment_uploads(db, requested_keys)
            if blocked:
                raise AttachmentReclaimedError(blocked[0])
        # 新消息即「最近活动」：把会话行一起 touch，侧栏才会把它顶到最前（issue #239）。
        # 与上面两处写入共用同一个 db、同一次 commit——分开提交会出现单边状态。
        crud_chat.touch_conversation(db, cid, user_id)
        db.commit()
        db.refresh(user_message)
        return user_message.id
    finally:
        db.close()


def _save_assistant_message(cid: str, answer: str, sources: list[dict], ragas_status: str,
                            retrieval_trace: dict) -> int:
    """写入 assistant 消息并返回它的 id；失败时由 `finally` 里的 close 回滚这次事务。"""
    db = SessionLocal()
    try:
        assistant_message = Message(
            conversation_id=cid,
            role="assistant",
            content=answer,
            sources=json.dumps(sources, ensure_ascii=False),
            ragas_status=ragas_status,
            retrieval_trace=json.dumps(retrieval_trace, ensure_ascii=False),
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        return assistant_message.id
    finally:
        db.close()


def _update_assistant_message_trace(message_id: int, retrieval_trace_json: str) -> None:
    """回答落库后补一次 retrieval_trace（学习轨迹引用）；消息已不在时静默跳过。"""
    db = SessionLocal()
    try:
        message = db.query(Message).filter_by(id=message_id).first()
        if message is None:
            return
        message.retrieval_trace = retrieval_trace_json
        db.commit()
    finally:
        db.close()


async def stream_chat(body: ChatRequest, authorization: str = Header("")):
    # 与 get_current_user 共用同一条鉴权链（验签 → 回查 → 世代 → 吊销登记）：这个入口
    # 不走 FastAPI 依赖注入、自己开会话，早先直接调用 decode_token，于是任何挂在依赖上的
    # 吊销判定都到不了这里（issue #184）。用户名改从鉴权后的那一行上取，不再单独解一次。
    #
    # 鉴权（连同它自己的 Session）整段交给工作线程，本函数不再持有请求级 Session：
    # 早先那句 `db = SessionLocal()` 之后，`query`/`add`/`commit`/`close` 全部落在事件循环
    # 线程上，一次网络 DB 往返期间同进程所有并发请求的 token 流都被冻住（issue #187）。
    # 现在每一段同步 DB 工作都是一个「开 session → 用 → 关」整段在工作线程里完成的步骤，
    # Session 对象既不跨线程传递，也不跨 await 长期持有。
    principal = await asyncio.to_thread(_authenticate_request, authorization)
    username = principal.username
    # 认过身份再占并发槽：一次聊天流会一直占着服务端资源到上游结束，没上限时少量连接就能
    # 打满（issue #183）。满员时立刻拒绝，不排队——排队会把请求拖到上游超时才释放。
    slot = rate_limit.chat_stream_slots.try_acquire()
    if slot is None:
        raise HTTPException(
            429,
            "当前并发聊天请求已达上限，请稍后重试。",
            headers={"Retry-After": "1"},
        )
    # 并发槽的归属分两段：外层持有到「把流交出去」为止，此后由生成器自持（各自 finally 归还）。
    # handed_off 就是这条交接线——交付前任何退出（普通异常、取消等 BaseException 族）都由外层
    # 兜底归还；交付后外层绝不 release，避免与生成器 finally 重复释放、把容量算大。
    handed_off = False
    try:
        trace = TraceRecorder(user_id=principal.user_id)
        await trace.add(
            "request_received",
            "stream_chat",
            creates={"trace_id": trace.trace_id},
            params={
                "conversation_id": body.conversation_id,
                "knowledge_base_id": body.knowledge_base_id,
                "question": body.question,
                "attachments_count": len(body.attachments or []),
            },
            result={"username": username},
            note="后端收到一次聊天请求，先建立 trace_id，后续所有步骤都会挂到这次请求下面。",
        )
        cid = body.conversation_id
        knowledge_base = await asyncio.to_thread(
            _resolve_request_knowledge_base, body.knowledge_base_id, principal.user_id
        )
        raw_question = (body.question or "").strip()
        display_question = raw_question or ("请分析这张图片" if body.attachments else "")
        if not display_question:
            await trace.add(
                "request_rejected",
                "stream_chat",
                uses={"raw_question": raw_question, "attachments_count": len(body.attachments or [])},
                result={"error": "问题不能为空"},
                note="没有文字问题，也没有图片附件，无法继续进入 RAG 流程。",
            )
            await trace.finish("failed")
            raise HTTPException(400, "问题不能为空")
        await trace.add(
            "input_normalized",
            "stream_chat",
            creates={"raw_question": raw_question, "display_question": display_question},
            result={"knowledge_base_id": knowledge_base.id, "knowledge_base_name": knowledge_base.name},
            note="系统整理用户输入，并确定本次请求要使用哪个知识库。",
        )

        effective_question, image_analysis = await _build_effective_question(
            raw_question,
            body.attachments,
        )
        await trace.add(
            "effective_question_built",
            "_build_effective_question",
            params={"raw_question": raw_question, "attachments_count": len(body.attachments or [])},
            creates={
                "effective_question": effective_question,
                "image_analysis_status": image_analysis.get("status", ""),
                "image_description": image_analysis.get("description", ""),
            },
            result={"image_analysis_error": image_analysis.get("error", "")},
            note="如果有图片，系统会先把图片转成文字描述，再与用户问题合并为真正用于检索和生成的问题。",
        )
        if body.attachments and image_analysis.get("status") == "failed" and not raw_question:
            await trace.add(
                "image_failed_directly",
                "_build_effective_question",
                uses={"attachments_count": len(body.attachments or [])},
                result={"error": image_analysis.get("error", "")},
                note="用户只发了图片但图片识别失败，因此不会进入知识库检索和模型回答。",
            )
            await trace.finish("failed")

            async def failure_stream():
                # 这条早退流同样占着一个并发槽，收尾必须归还（正常走完/被断开都走 finally）。
                try:
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                    analysis_data = json.dumps(
                        {
                            "type": "image_analysis",
                            "analysis": image_analysis,
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {analysis_data}\n\n"
                    error_data = json.dumps(
                        {
                            "type": "error",
                            "message": image_analysis.get("error") or "图片内容识别失败，请检查清晰度后重新上传。",
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {error_data}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    slot.release()

            response = StreamingResponse(failure_stream(), media_type="text/event-stream")
            handed_off = True
            return response
        title_source = raw_question or effective_question or display_question
        title = title_source[:30] + ("..." if len(title_source) > 30 else "")
        conversation = await asyncio.to_thread(
            _load_or_create_conversation,
            cid,
            principal.user_id,
            knowledge_base,
            title,
        )
        cid = conversation.id
        knowledge_base = conversation.knowledge_base
        if conversation.created:
            await trace.add(
                "conversation_created",
                "stream_chat",
                creates={"conversation_id": cid, "title": title},
                result={"knowledge_base_id": knowledge_base.id},
                note="这是新对话，系统创建 conversation，并把它绑定到当前知识库。",
            )
        else:
            await trace.add(
                "conversation_loaded",
                "stream_chat",
                uses={"conversation_id": cid},
                result={"knowledge_base_id": knowledge_base.id, "title": conversation.title},
                note="这是已有对话，系统复用它原本绑定的知识库，避免会话中途串库。",
            )
        await trace.attach(conversation_id=cid)

        # save user message
        accepted_attachments = _service_minted_attachments(body.attachments, cid)
        try:
            user_message_id = await asyncio.to_thread(
                _save_user_message,
                cid,
                principal.user_id,
                display_question,
                accepted_attachments,
            )
        except AttachmentReclaimedError as exc:
            # 这条消息引用的键已经被清扫任务判为待回收（issue #238）：对象正被删或已删，
            # 消息绝不能落库。返回一条 SSE 错误帧而不是抛 HTTPException——这是流式端点，
            # 前端按帧类型分支，与上面「图片识别失败」那条早退流保持同一种收尾形状。
            # 异常原文（object_key、时间戳）只进日志与 trace，不进给用户的文案。
            logger.warning(
                "chat attachment reclaimed, refusing to save the message: %s", exc
            )
            await trace.add(
                "attachment_reclaimed",
                "_save_user_message",
                uses={"conversation_id": cid, "attachments_count": len(accepted_attachments)},
                result={"error": ATTACHMENT_RECLAIMED_MESSAGE},
                note="附件在发送前已被清扫任务判为待回收，消息被拒，不会写入数据库。",
            )
            await trace.finish("failed")

            async def reclaimed_stream():
                # 与其它早退流一样占着并发槽，收尾必须归还（正常走完/被断开都走 finally）。
                try:
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                    error_data = json.dumps(
                        {"type": "error", "message": ATTACHMENT_RECLAIMED_MESSAGE},
                        ensure_ascii=False,
                    )
                    yield f"data: {error_data}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    slot.release()

            response = StreamingResponse(reclaimed_stream(), media_type="text/event-stream")
            handed_off = True
            return response
        await trace.add(
            "user_message_saved",
            "Message",
            creates={"user_message_id": user_message_id},
            params={"content": display_question, "attachments_count": len(body.attachments or [])},
            note="用户消息先写入数据库，后面的滑动窗口会排除这条当前消息，避免重复塞进 prompt。",
        )

        recent_text = await _build_recent_memory_text(
            cid,
            current_message_id=user_message_id,
            # 上界取自上面那次会话快照：紧接着的用户消息写入不会推进它，取值与原先读同一个
            # 会话对象等价，但读的位置从事件循环线程挪到了工作线程（issue #187）。
            summary_upto=conversation.memory_summary_upto_message_id,
            trace_id=trace.trace_id,
        )
        memory_context = _build_memory_context(
            conversation,
            recent_text=recent_text,
        )
        retrieval_question = _build_memory_aware_retrieval_question(effective_question, memory_context)
        await trace.add(
            "memory_built",
            "_build_memory_context",
            uses={
                "conversation_id": cid,
                "memory_summary": conversation.memory_summary,
                "summary_upto_message_id": conversation.memory_summary_upto_message_id,
            },
            creates={
                "recent_text": recent_text,
                "memory_context": memory_context,
                "retrieval_question": retrieval_question,
            },
            result={
                "memory_used": bool(memory_context),
                "used_for_retrieval": retrieval_question != effective_question,
                "window_turns": MEMORY_WINDOW_TURNS,
            },
            note="系统构造短期滑动窗口和长期摘要记忆，并生成真正用于 RAG 检索的问题。",
        )

        rag_gate = await decide_need_rag(
            effective_question,
            memory_context,
            knowledge_base.name,
            body.attachments,
        )
        need_rag = bool(rag_gate.get("need_rag", True))
        generation_mode = "rag" if need_rag else "direct"
        await trace.add(
            "rag_gate_decided",
            "decide_need_rag",
            uses={
                "effective_question": effective_question,
                "memory_context": memory_context,
                "knowledge_base_name": knowledge_base.name,
                "attachments_count": len(body.attachments or []),
            },
            creates={"rag_gate": rag_gate},
            result={
                "need_rag": need_rag,
                "route": rag_gate.get("route", generation_mode),
                "confidence": rag_gate.get("confidence", 0),
                "source": rag_gate.get("source", ""),
                "reason": rag_gate.get("reason", ""),
            },
            note="模型先判断这轮问题是否需要进入知识库检索。若判定为直答，则跳过 RAG，只用问题和会话记忆生成回答。",
        )

        context = ""
        sources = []
        retrieved_contexts = []
        knowledge_chunks = []
        retrieval_trace = {
            # 同一份状态会随 retrieval_trace 落库并回查给用户：last_error 原文（上游地址 +
            # 原始异常）只留服务端，轨迹里给固定文案。
            "embedding": embedding_trace_status(embedding_backend_status()),
            "query_plan": {},
            "routes": [],
            "rrf": [],
            "rerank": {"status": "skipped", "items": []},
            "memory": {
                "used": bool(memory_context),
                "used_for_retrieval": False,
                "window_turns": MEMORY_WINDOW_TURNS,
                "summary_available": bool(conversation.memory_summary),
                "summary_upto_message_id": conversation.memory_summary_upto_message_id,
            },
            "rag_gate": rag_gate,
            "mode": generation_mode,
            "effective_question": effective_question,
            "retrieval_question": retrieval_question,
        }
        if body.attachments:
            retrieval_trace["image_analysis_status"] = image_analysis.get("status", "")
            retrieval_trace["image_analysis_error"] = image_analysis.get("error", "")
            retrieval_trace["image_description"] = image_analysis.get("description", "")

        if need_rag:
            knowledge_chunks, retrieval_trace = await retrieve_knowledge(
                retrieval_question,
                knowledge_base_id=knowledge_base.id,
                # 关键词召回自带会话（issue #187）：stream_chat 不再持有请求级 Session，
                # 由检索侧那一步自己「开 session → 用 → 关」，整段落在同一个工作线程里，
                # 避免把 Session 对象递到别的线程上。
                db=None,
                trace_recorder=trace,
            )
            retrieval_trace = retrieval_trace or {}
            retrieval_trace["memory"] = {
                "used": bool(memory_context),
                "used_for_retrieval": retrieval_question != effective_question,
                "window_turns": MEMORY_WINDOW_TURNS,
                "summary_available": bool(conversation.memory_summary),
                "summary_upto_message_id": conversation.memory_summary_upto_message_id,
            }
            retrieval_trace["rag_gate"] = rag_gate
            retrieval_trace["mode"] = generation_mode
            retrieval_trace["effective_question"] = effective_question
            retrieval_trace["retrieval_question"] = retrieval_question
            if body.attachments:
                retrieval_trace["image_analysis_status"] = image_analysis.get("status", "")
                retrieval_trace["image_analysis_error"] = image_analysis.get("error", "")
                retrieval_trace["image_description"] = image_analysis.get("description", "")
            await trace.add(
                "retrieval_completed",
                "retrieve_knowledge",
                params={"question": retrieval_question, "knowledge_base_id": knowledge_base.id},
                creates={
                    "query_plan": retrieval_trace.get("query_plan", {}),
                    "routes": retrieval_trace.get("routes", []),
                    "rrf": retrieval_trace.get("rrf", []),
                    "rerank": retrieval_trace.get("rerank", {}),
                },
                result={"final_chunks_count": len(knowledge_chunks)},
                note="Advanced RAG retrieval completed with query planning, multi-route recall, RRF fusion, and rerank.",
            )
            if knowledge_chunks:
                context = "\n\n".join(
                    f"[来源: {c['file_name']}]\n{c['content']}"
                    for c in knowledge_chunks
                )
            sources = _build_sources(knowledge_chunks)
            retrieved_contexts = [c.get("content", "") for c in knowledge_chunks if c.get("content")]
            await trace.add(
                "context_built",
                "_build_sources",
                creates={"context": context, "sources": sources},
                result={"sources_count": len(sources), "retrieved_contexts_count": len(retrieved_contexts)},
                note="系统把最终选中的 chunk 拼成给大模型看的知识库上下文，并生成前端可展开的参考资料。",
            )
        else:
            retrieval_trace["skip_reason"] = rag_gate.get("reason", "")
            await trace.add(
                "retrieval_skipped",
                "decide_need_rag",
                uses={
                    "effective_question": effective_question,
                    "memory_context": memory_context,
                },
                result={
                    "need_rag": False,
                    "route": "direct",
                    "confidence": rag_gate.get("confidence", 0),
                    "reason": rag_gate.get("reason", ""),
                },
                note="本轮判定为直答，不进入知识库检索，也不启动 RAGAS。回答将只结合问题与会话记忆。",
            )

        async def event_stream():
            try:
                full = ""
                failed = False
                first_chunk_seen = False
                conversation_data = json.dumps({
                    "type": "conversation",
                    "conversation": {
                        "id": cid,
                        "title": conversation.title,
                        "knowledge_base_id": knowledge_base.id,
                        "knowledge_base_name": knowledge_base.name,
                    },
                }, ensure_ascii=False)
                if body.attachments:
                    analysis_data = json.dumps(
                        {
                            "type": "image_analysis",
                            "analysis": image_analysis,
                        },
                        ensure_ascii=False,
                    )
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                    yield f"data: {analysis_data}\n\n"
                for payload in _trace_sse_payloads(trace):
                    yield payload
                yield f"data: {conversation_data}\n\n"
                if need_rag and sources:
                    data = json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                await trace.add(
                    "generation_started",
                    "stream_rag_answer",
                    params={
                        "question": effective_question,
                        "memory_context": memory_context,
                        "context": context,
                        "mode": generation_mode,
                    },
                    note=(
                        "开始调用文本模型。"
                        if need_rag
                        else "开始调用文本模型。当前问题被判定为直答，不进入知识库检索，只结合问题与会话记忆回答。"
                    ),
                )
                for payload in _trace_sse_payloads(trace):
                    yield payload
                async for event in stream_rag_answer(
                    effective_question,
                    context,
                    memory_context,
                    trace,
                    use_rag=need_rag,
                ):
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                    if isinstance(event, dict):
                        if event.get("type") == "error":
                            failed = True
                            await trace.add(
                                "generation_failed",
                                "_stream_deepseek_response",
                                result={"message": event.get("message") or event.get("content") or ANSWER_GENERATION_FAILED_MESSAGE},
                                note="模型生成阶段失败，系统会返回错误事件，并且不会保存失败 assistant 消息。",
                            )
                            for payload in _trace_sse_payloads(trace):
                                yield payload
                            data = json.dumps(
                                {
                                    "type": "error",
                                    "message": event.get("message") or event.get("content") or ANSWER_GENERATION_FAILED_MESSAGE,
                                },
                                ensure_ascii=False,
                            )
                            yield f"data: {data}\n\n"
                            break
                        if event.get("type") == "reset":
                            # 后备模型从头重新生成整段回答：先作废已下发的增量，
                            # 落库文本也从零重新累积，绝不与重置前的内容拼接。
                            full = ""
                            first_chunk_seen = False
                            await trace.add(
                                "stream_reset",
                                "_stream_openai_chat_chunks",
                                params={"reason": event.get("reason") or ""},
                                note="已下发 reset 事件作废此前流式内容，本轮回答改为只保留重置后重新生成的部分。",
                            )
                            for payload in _trace_sse_payloads(trace):
                                yield payload
                            data = json.dumps(
                                {
                                    "type": "reset",
                                    "reason": event.get("reason") or "",
                                    "message": event.get("message") or "",
                                },
                                ensure_ascii=False,
                            )
                            yield f"data: {data}\n\n"
                            continue
                        chunk = event.get("content", "")
                    else:
                        chunk = event
                    data = json.dumps({"content": chunk}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    if not failed:
                        if chunk and not first_chunk_seen:
                            first_chunk_seen = True
                            await trace.add(
                                "first_content_chunk",
                                "_stream_openai_chat_chunks",
                                result={"chunk": chunk},
                                note="大模型开始返回第一段流式内容，前端会逐步拼接为正在生成的回答。",
                            )
                            for payload in _trace_sse_payloads(trace):
                                yield payload
                        full += chunk

                # save assistant message
                if full and not failed:
                    await _safe_trace_add(
                        trace,
                        "assistant_ready_to_save",
                        "Message",
                        uses={
                            "full_answer": full,
                            "sources_count": len(sources),
                            "need_rag": need_rag,
                        },
                        note=(
                            "模型完整回答成功，系统准备保存 assistant 消息，并启动摘要判断。"
                            if not need_rag
                            else "模型完整回答成功，系统准备保存 assistant 消息，并启动 RAGAS 和摘要判断。"
                        ),
                    )
                    await _attach_grounding_trace(
                        retrieval_trace,
                        trace,
                        answer=full,
                        retrieved_contexts=retrieved_contexts,
                        need_rag=need_rag,
                    )
                    retrieval_trace["learning_trace"] = compact_trace_reference(trace.snapshot())
                    assistant_message_id = None
                    try:
                        # 保存 assistant 消息是另一段同步 DB 工作，同样整段落在工作线程里；
                        # 它自己的会话由 `_save_assistant_message` 收尾（失败即回滚）。
                        assistant_message_id = await asyncio.to_thread(
                            _save_assistant_message,
                            cid,
                            full,
                            sources,
                            "pending" if need_rag else "",
                            retrieval_trace,
                        )
                    except Exception as exc:
                        logger.warning("Assistant message save failed after stream finished [trace_id=%s]: %s",
                                       trace.trace_id, exc, exc_info=True)
                        await _safe_trace_add(
                            trace,
                            "assistant_save_failed",
                            "Message",
                            result={"error": ASSISTANT_SAVE_FAILED_MESSAGE, "trace_id": trace.trace_id},
                            note="模型回答已经生成完毕，但保存 assistant 消息失败。系统仍会结束流，避免前端误报 network error。",
                        )

                    if assistant_message_id is not None:
                        await _safe_trace_attach(trace, conversation_id=cid, message_id=assistant_message_id)
                        await _safe_trace_add(
                            trace,
                            "assistant_message_saved",
                            "Message",
                            creates={"assistant_message_id": assistant_message_id},
                            result={"ragas_status": "pending" if need_rag else ""},
                            note="assistant 消息保存成功，历史会话刷新后仍可从这条消息打开流程。",
                        )
                        if need_rag:
                            try:
                                schedule_ragas_evaluation(
                                    assistant_message_id,
                                    effective_question,
                                    full,
                                    retrieved_contexts,
                                    trace.trace_id,
                                )
                                await _safe_trace_add(
                                    trace,
                                    "ragas_scheduled",
                                    "schedule_ragas_evaluation",
                                    params={
                                        "message_id": assistant_message_id,
                                        "question": effective_question,
                                        "answer_chars": len(full),
                                        "contexts_count": len(retrieved_contexts),
                                    },
                                    note="RAGAS 在 assistant 保存后异步启动，不阻塞用户看到答案。",
                                )
                            except Exception as exc:
                                logger.warning("RAGAS schedule failed after stream finished [trace_id=%s]: %s",
                                               trace.trace_id, exc, exc_info=True)
                                await _safe_trace_add(
                                    trace,
                                    "ragas_schedule_failed",
                                    "schedule_ragas_evaluation",
                                    result={"error": RAGAS_SCHEDULE_FAILED_MESSAGE, "trace_id": trace.trace_id},
                                    note="RAGAS 调度失败，但不影响主回答完成。",
                                )
                        else:
                            await _safe_trace_add(
                                trace,
                                "ragas_skipped",
                                "schedule_ragas_evaluation",
                                result={"need_rag": False, "reason": rag_gate.get("reason", "")},
                                note="当前问题被路由为直答，因此不启动 RAGAS。",
                            )
                        try:
                            _schedule_memory_summary_update(cid, trace.trace_id)
                            await _safe_trace_add(
                                trace,
                                "memory_summary_update_scheduled",
                                "_schedule_memory_summary_update",
                                params={"conversation_id": cid},
                                note="系统异步检查长期记忆是否超过上限，若超过则进行二次摘要。",
                            )
                        except Exception as exc:
                            logger.warning("Memory summary schedule failed after stream finished [trace_id=%s]: %s",
                                           trace.trace_id, exc, exc_info=True)
                            await _safe_trace_add(
                                trace,
                                "memory_summary_update_schedule_failed",
                                "_schedule_memory_summary_update",
                                result={"error": MEMORY_SUMMARY_SCHEDULE_FAILED_MESSAGE, "trace_id": trace.trace_id},
                                note="长期记忆压缩调度失败，但不影响主回答完成。",
                            )
                        try:
                            retrieval_trace["learning_trace"] = compact_trace_reference(trace.snapshot())
                            await asyncio.to_thread(
                                _update_assistant_message_trace,
                                assistant_message_id,
                                json.dumps(retrieval_trace, ensure_ascii=False),
                            )
                        except Exception as exc:
                            logger.warning("Assistant trace reference update failed: %s", exc, exc_info=True)
                    await _safe_trace_finish(
                        trace,
                        "done" if assistant_message_id is not None else "partial",
                        conversation_id=cid,
                        message_id=assistant_message_id,
                    )
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                elif failed:
                    await _safe_trace_add(
                        trace,
                        "assistant_not_saved",
                        "Message",
                        result={"saved": False},
                        note="回答生成失败，遵循项目规则：不把失败内容保存为正式 assistant 消息。",
                    )
                    await _safe_trace_finish(trace, "failed", conversation_id=cid)
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                else:
                    # 这里只处理生成循环正常结束后的零正文，不覆盖前置早退或取消。
                    # reset 作废的旧正文不能算成功；也不把错误提示存为正式回答。
                    await _safe_trace_add(
                        trace,
                        "generation_failed",
                        "stream_rag_answer",
                        result={"reason": "empty_response", "message": EMPTY_MODEL_RESPONSE_MESSAGE},
                        note="模型生成正常结束但未返回正文，本轮按失败处理。",
                    )
                    await _safe_trace_add(
                        trace,
                        "assistant_not_saved",
                        "Message",
                        result={"saved": False},
                        note="模型未返回正文，不保存空的 assistant 消息。",
                    )
                    # 客户端收到 error 后可能断开，先持久化终态再发送错误帧。
                    await _safe_trace_finish(trace, "failed", conversation_id=cid)
                    for payload in _trace_sse_payloads(trace):
                        yield payload
                    data = json.dumps(
                        {"type": "error", "message": EMPTY_MODEL_RESPONSE_MESSAGE},
                        ensure_ascii=False,
                    )
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # 客户端断开时，Starlette 的 StreamingResponse 会取消正在跑流的任务，把
                # CancelledError 抛进挂起的 yield（其它 ASGI 实现也可能用 aclose() → GeneratorExit）；
                # CancelledError 与 GeneratorExit 都继承自 BaseException，外层 `except Exception`
                # 兜不住，原先写在函数体末尾的收尾逻辑不会执行。只有放进 finally 才能保证
                # 断连路径同样把 trace 落到终态。finally 中不得再 yield。
                #
                # 这里不再需要补关会话（issue #187）：每一次写库的 Session 都在它自己的工作线程里
                # 「开 → 用 → 关」，断开路径上不存在「借出去还没还」的连接，
                # #58 那条「断开必须归还连接池」的保证因此从「靠这段 finally」变成结构性成立。
                try:
                    # 正常路径已把 status 写成 done/partial/failed，这里只兜断开等异常收尾。
                    # 这一次写回是协程（issue #201），写库整段在工作线程里完成：断连时即使
                    # 调用方在这个 `await` 上被打断，写入也已经交出去了、会在工作线程里跑完
                    # 并关掉自己的会话——终态不会因为取消而丢，它不再是「借出去要还的连接」。
                    if getattr(trace, "status", None) == "running":
                        await _safe_trace_finish(trace, "failed", conversation_id=cid)
                except Exception as exc:
                    logger.warning("Learning trace teardown failed: %s", exc, exc_info=True)
                finally:
                    # 并发槽跟数据库会话一样，必须在 finally 里归还：正常跑完、流里抛异常、
                    # 客户端断开（CancelledError/GeneratorExit）三条路径都经过这里。漏掉任何
                    # 一条，一次断连就会永久占着一个槽。
                    slot.release()

        response = StreamingResponse(event_stream(), media_type="text/event-stream")
        handed_off = True
        return response
    finally:
        # 流还没交出去就退出（占槽后、StreamingResponse 之前的失败或被取消）：生成器的
        # finally 够不到这段窗口，外层必须把槽还回去。取消/GeneratorExit 属于 BaseException，
        # 早先那条 `except Exception` 接不住，一次取消就永久占着一个槽（issue #258）。
        if not handed_off:
            slot.release()
