from datetime import datetime, timedelta

from sqlalchemy import and_, func, literal, or_
from sqlalchemy.orm import Session, selectinload

from config import CHAT_ATTACHMENT_LEASE_SECONDS
from crud import trace as crud_trace
from crud.pagination import LIST_DEFAULT_LIMIT, clamp_limit, datetime_cursor_value, seconds_text
from model.models import ChatAttachmentUpload, ChatTraceSession, Conversation, Message
from service.json_utils import load_json_value


# 消息历史默认页大小与单页上限：不传分页参数时也必须有上限，否则会话越长单次响应越大。
CHAT_MESSAGE_DEFAULT_LIMIT = 50
CHAT_MESSAGE_MAX_LIMIT = 200

# 标题搜索的 LIKE 转义符。**显式写 ESCAPE**：MySQL 默认把反斜杠当转义符，SQLite 默认没有
# 转义符，不写这一句同一个模式在两边含义不同。选 `!` 而不是反斜杠，还能让反斜杠本身
# 保持字面含义（用户搜索 `a\b` 时不必再处理一层转义）。
LIKE_ESCAPE = "!"


def _escape_like(value: str) -> str:
    """把用户输入转成可放进 LIKE 模式的字面量子串（不含两侧的 `%`）。

    只转义 LIKE 的元字符：`!`（转义符自身）、`%`（任意串）、`_`（任意单字符）。
    顺序必须先把 `!` 翻倍，再给通配符加前缀，否则刚插入的转义符会被再转义一次。
    """
    return (
        value.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
        .replace("%", LIKE_ESCAPE + "%")
        .replace("_", LIKE_ESCAPE + "_")
    )


# 登记行状态的三个判据（issue #238）。**每一条只在这里定义一次**，清扫、领取、消费、
# 拒签四条链路共用同一段 SQL：判据一旦在多处各写一份，就会各自演化出微妙的差异，
# 而这次修复的全部要点就是「发送侧与清扫侧看到的是同一个事实」。
def _pending_upload_clause():
    """P：尚未结算的行——既没被消息消费，也没被判为待回收。

    清扫候选、领取（claim）、确认消费（confirm）三处都用它。它同时是「消费」与「回收」
    两条路径互斥的落点：一条行一旦被任一方赢下，就不再是另一方的候选。
    """
    return and_(
        ChatAttachmentUpload.consumed_at.is_(None),
        ChatAttachmentUpload.reclaimed_at.is_(None),
    )


def _reclaimed_upload_clause():
    """B：墓碑——删除意图已提交，行还留着。

    发送侧的拒签与清扫的收尾/回退都以它为准，且**只看它**：不加租约条件，也不看
    claimed_at。租约过期只说明「上一次删除外呼没做完」，不说明「这个对象还在」——
    把租约过期当成放行条，正是这次修复要消灭的误判。

    判据落在键域上（只按 object_key 认行），因此跨用户也成立：一把别人铸、正在被回收的
    键，我照样发不出去——对象存储没有回收站，跨用户的差别不构成放行的理由。
    """
    return ChatAttachmentUpload.reclaimed_at.is_not(None)


def _stale_reclaimed_upload_clause(now: datetime, lease_seconds: int | None = None):
    """R：租约已过期的墓碑——上一轮清扫死在半路上留下的行，可以重新领取并重试删除。

    只有 claimed_at 非空且停留在租约之外的行才符合：claimed_at 为空的是已经结算完的墓碑
    （删除成功过，没什么可重试的），租约之内的是此刻可能正在外呼的行（重领会造成两次
    并发删除同一对象）。用它重新领取时**只刷新 claimed_at，不动 reclaimed_at**——
    墓碑一旦写下就不回退，否则发送侧会看到拒签凭空消失。
    """
    lease = CHAT_ATTACHMENT_LEASE_SECONDS if lease_seconds is None else lease_seconds
    return and_(
        _reclaimed_upload_clause(),
        ChatAttachmentUpload.claimed_at.is_not(None),
        ChatAttachmentUpload.claimed_at < now - timedelta(seconds=lease),
    )


def serialize_message(message: Message) -> dict:
    retrieval_trace = load_json_value(message.retrieval_trace, {})
    if not isinstance(retrieval_trace, dict):
        retrieval_trace = {}

    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "sources": load_json_value(message.sources, []),
        "attachments": load_json_value(message.attachments, []),
        "ragas_status": message.ragas_status or "",
        "ragas_scores": load_json_value(message.ragas_scores, {}),
        "ragas_error": message.ragas_error or "",
        # 未反馈统一回 0：存量行与新建行同款语义（issue #250）。
        "feedback": message.feedback or 0,
        "retrieval_trace": retrieval_trace,
        "image_analysis_status": retrieval_trace.get("image_analysis_status", ""),
        "image_analysis_error": retrieval_trace.get("image_analysis_error", ""),
        "image_description": retrieval_trace.get("image_description", ""),
        "created_at": message.created_at.isoformat() if message.created_at else "",
    }


def list_conversations(
    db: Session,
    user_id: int,
    *,
    limit: int = LIST_DEFAULT_LIMIT,
    before: tuple[datetime, str] | None = None,
    q: str | None = None,
) -> list[Conversation]:
    """按 user_id 取一页会话，按「最近活动」倒序，与旧接口的数组顺序一致。

    翻页语义（issue #191）：**键集游标**，不用 offset——offset 在翻页期间有新会话插入时
    会整体位移，同一行会被翻到两次、另有一行永远翻不到。游标落在排序键本身上，
    取「排在这条记录之后」的那一页，插入多少新行都不影响已经翻过的区间。

    会话的排序键是 `(updated_at, 自增主键)` 的**复合键**，两个理由：
    1) 会话主键是 uuid（`model/models.py` 的 `_new_id`），不是自增整数，按它排序等于乱序，
       既不能当序键也不能当游标——这是会话列表与知识库/文件列表（都是自增整数主键）
       在游标形态上不一样的原因；
    2) `updated_at` 是秒级 DATETIME（`func.now()` 在 MySQL 上只到秒），同一秒内改动的多个
       会话按它排序不稳定，翻页会重复或漏行。补一个唯一且不重复的主键做末位键，
       `(updated_at, id)` 就是全序，游标比较才有确定答案。

    已知边界：`updated_at` 是可变的——五类写入方都会改写它：新消息（`_save_user_message` 的
    `touch_conversation`）、删除消息（`delete_message` 的 `touch_conversation`，issue #256）、
    重命名（`rename_conversation`）、记忆摘要写入（`rag/memory_service.py` 的
    `_write_memory_summary_text`）、删除知识库时的会话改绑（`crud/knowledge_base.py`）；
    启动期那条裸 SQL 回填（`database/session.py`）绕过 ORM、**不会**改写它（有意为之）。
    翻页途中被改写的那个会话会跳到游标之前（用户已经翻过的区间），它要么已经在上一次响应里、
    要么要等侧栏下一次刷新才出现——侧栏每轮问答结束后都会重取最新一页并重置游标
    （`src/stores/chat.js` 的 `fetchConversations`），收敛得掉。新消息 bump 之后这个边界被触发
    的频率比修之前高（每问一次就顶一次），但形状没变：不重复、不静默丢行。反过来把序键换成
    不可变的 `created_at` 能彻底消掉它，但侧栏会从「最近活动优先」变成「创建时间优先」，
    属于本单之外的可见行为变更，不做（见 issue #239 的修复计划）。

    `before` 是 (updated_at, id) 复合游标，两者必须同时给出（service 层负责拦半个游标）。

    `q`（issue #251）按标题做子串匹配，和游标过滤是 AND 关系：翻页始终在同一个搜索结果集内
    进行，不会因为翻到第二页就漏出未命中的会话。匹配用 LIKE、不是全文索引，`%`/`_`/`!`
    由 `_escape_like` 转成字面量；大小写行为取决于数据库的 LIKE 语义（SQLite 对 ASCII
    不区分大小写，MySQL 默认排序规则通常也不区分），接口不额外做大小写归一。
    """
    limit = clamp_limit(limit)
    # 排序与游标过滤共用同一个秒级表达式：两处必须同一口径，否则游标落不到排序位置上
    # （列里存的文本形状不唯一，见 seconds_text）。排序用整列时，带小数的行会排在
    # 秒级行之间，而游标只按秒级比较，两者对不齐。
    timestamp_key = seconds_text(Conversation.updated_at)
    query = (
        db.query(Conversation)
        # 预加载知识库：序列化时每个会话都要读 conv.knowledge_base，逐行懒加载会让查询数随会话数增长。
        .options(selectinload(Conversation.knowledge_base))
        .filter_by(user_id=user_id)
    )
    if q:
        # 先转义再套两侧通配符：用户输入原样拼进去会把 `%`/`_` 当成通配符（搜 `100%` 会命中
        # 所有以 `100` 开头的标题）。ESCAPE 与 _escape_like 必须用同一个字符。
        query = query.filter(
            Conversation.title.like(f"%{_escape_like(q)}%", escape=LIKE_ESCAPE)
        )
    if before is not None:
        before_updated_at, before_id = before
        # 等价于 (updated_at, id) < (before_updated_at, before_id) 的行值比较，写成展开式是为了
        # SQLite 与 MySQL 共用同一段 SQL（两者都支持行值比较，但展开式在两边都不依赖方言支持）。
        # 两档都要归一后再比：时间戳截到秒级文本（理由见 seconds_text），主键本来就是文本。
        cursor_value = literal(datetime_cursor_value(before_updated_at))
        query = query.filter(or_(
            timestamp_key < cursor_value,
            and_(timestamp_key == cursor_value, Conversation.id < before_id),
        ))
    return (
        query.order_by(timestamp_key.desc(), Conversation.id.desc())
        .limit(limit)
        .all()
    )


def get_conversation(db: Session, cid: str, user_id: int) -> Conversation | None:
    return db.query(Conversation).filter_by(id=cid, user_id=user_id).first()


def touch_conversation(db: Session, cid: str, user_id: int) -> bool:
    """把会话的 updated_at 推到现在，让它回到侧栏「最近活动」的最前面。

    **故意不提交**（同 `confirm_attachment_uploads`）：调用方必须把这次 touch 与消息行、
    附件登记行的写入放进同一次 commit，否则会出现「消息已落库但侧栏没动」或
    「侧栏动了但消息没落库」的单边状态。

    按 (id, user_id) 定位，与 `get_conversation` 同一形状：用户域限定不是可选的——
    cid 来自请求体，不限定就等于「拿别人的 cid 顶别人的会话」。

    用 `func.now()` 而不是 `datetime.now()`：前者落到列里是秒级文本（与列默认值同形状，
    `seconds_text` 的前提），后者会引入带微秒的第三种文本形状。

    读改写而不是 `query(...).update({...})`：后者在本仓库的 `FakeDb` 替身上不存在
    （`tests/conftest.py` 只实现了 `filter_by`/`first`），会让所有把 `stream_chat` 跑在
    FakeDb 上的用例集体报错；这条窄路径在替身上退化成静默 no-op，是既有用例零改写的正路。
    """
    conversation = get_conversation(db, cid, user_id)
    if conversation is None:
        return False
    conversation.updated_at = func.now()
    return True


def list_messages(
    db: Session,
    cid: str,
    user_id: int,
    *,
    limit: int = CHAT_MESSAGE_DEFAULT_LIMIT,
    before_id: int | None = None,
) -> list[Message] | None:
    """按会话取一页消息，返回「旧 -> 新」顺序，与旧接口的数组顺序一致。

    游标用自增主键而不是 created_at：created_at 是秒级 DATETIME，同一秒内的消息按它排序
    不稳定，翻页会重复或漏行。先倒序取下 limit 条再反转，取到的就是 cursor 之前最新的那页。

    页大小在这里兜底夹取（而不是只依赖服务层的 422 校验）：CRUD 是更底层的入口，脚本或内部
    调用可能绕开接口层，负值在 SQLite 上等价于「不设上限」，会把整段历史一次读出来。
    """
    conversation = get_conversation(db, cid, user_id)
    if not conversation:
        return None
    limit = CHAT_MESSAGE_DEFAULT_LIMIT if limit is None else max(1, min(limit, CHAT_MESSAGE_MAX_LIMIT))
    query = db.query(Message).filter(Message.conversation_id == cid)
    if before_id is not None:
        query = query.filter(Message.id < before_id)
    rows = query.order_by(Message.id.desc()).limit(limit).all()
    rows.reverse()
    return rows


def get_message_export_upper_id(db: Session, cid: str) -> int | None:
    """捕获本次导出的消息 ID 上界；调用方必须先校验会话归属。"""
    return db.query(func.max(Message.id)).filter(Message.conversation_id == cid).scalar()


def list_message_export_batch(
    db: Session,
    cid: str,
    *,
    upper_id: int,
    after_id: int | None = None,
) -> list:
    """直接查库导出，按主键升序读一批；不复用倒序的聊天历史分页。

    只读取 Markdown 需要的字段，避免把附件凭据、检索轨迹和评测数据带进导出。
    ID 上界固定消息范围，不构成原子快照：读取期间的修改/删除仍可能被观察到。
    """
    query = db.query(
        Message.id, Message.role, Message.created_at, Message.content, Message.sources
    ).filter(Message.conversation_id == cid, Message.id <= upper_id)
    if after_id is not None:
        query = query.filter(Message.id > after_id)
    return query.order_by(Message.id.asc()).limit(CHAT_MESSAGE_MAX_LIMIT).all()


def list_conversation_attachment_owners(
    db: Session, cid: str, user_id: int
) -> list[tuple[str, int | None]] | None:
    """会话内所有消息引用到的 OSS 对象键，连同**属主**一起，去重后按「旧 -> 新」返回。

    必须在会话行被删除之前调用：`messages` 随会话级联删除，附件列里的对象键会一起消失，
    删完之后就再也说不清这个会话扔下过哪些对象。只看 attachments 一列，不把正文
    （LONGTEXT）读进内存；也不走 list_messages——那个有分页上限，翻不到的消息会漏收。

    属主取自 `chat_attachment_uploads` 的 `user_id`（issue #233）：那把键是谁铸的，就只有
    谁能回收它。键在附件列里出现**不构成归属凭据**——附件列由客户端回带，任何登录用户都能
    在自己的消息里写上别人的键，拿它当凭据等于让任何人借自己的会话删别人的对象。
    登记表里查无此键（历史数据、或客户端编的键）时属主是 None，调用方据此拒签。

    一次查询取回整批属主（而不是逐键查）：这是删会话路径上的热查询，键多时逐键会变成
    N+1。会话不存在或不属于该用户时返回 None，与 list_messages 的约定一致。
    """
    if not get_conversation(db, cid, user_id):
        return None

    rows = (
        db.query(Message.attachments)
        .filter(Message.conversation_id == cid)
        .order_by(Message.id.asc())
        .all()
    )
    keys: list[str] = []
    seen: set[str] = set()
    for (raw_attachments,) in rows:
        for item in load_json_value(raw_attachments, []):
            # 历史脏数据与上传中断留下的记录都可能不是 dict、或缺 object_key。
            if not isinstance(item, dict):
                continue
            object_key = item.get("object_key")
            if not isinstance(object_key, str) or not object_key or object_key in seen:
                continue
            seen.add(object_key)
            keys.append(object_key)
    if not keys:
        return []

    owners = dict(
        db.query(ChatAttachmentUpload.object_key, ChatAttachmentUpload.user_id)
        .filter(ChatAttachmentUpload.object_key.in_(keys))
        .all()
    )
    return [(key, owners.get(key)) for key in keys]


def register_attachment_upload(db: Session, object_key: str, user_id: int) -> ChatAttachmentUpload:
    """登记一次上传铸出的对象键，并立刻提交。

    提交必须发生在**写对象之前**（调用方负责顺序）：反过来时，一次「对象已经进桶、进程
    随即退出」就留下一个库里没有任何痕迹的对象——正是 issue #142 的形状。先落库之后，
    写对象失败最坏只是一条指向不存在对象的登记行，清扫任务顺手抹掉即可。

    对象键是主键，重复登记会撞唯一约束；上传接口每次铸的都是新的 uuid4().hex，正常路径
    不会重复。
    """
    upload = ChatAttachmentUpload(object_key=object_key, user_id=user_id)
    db.add(upload)
    db.commit()
    return upload


def confirm_attachment_uploads(db: Session, object_keys: list[str], user_id: int) -> int:
    """消费已经转为正式引用的登记行，返回消费掉的条数。**故意不提交**。

    调用方必须把它和消息行放进同一次 commit：分开提交时，「消息已落库、登记行还在」的
    中间态会让清扫任务把一条活消息引用的对象当成孤儿删掉；反过来则是消息没落库却把登记行
    消费掉，对象从此脱离所有回收路径。

    只消费 user_id 自己的行：附件列由客户端回带，一条消息可以引用一把别人铸的键（形态校验
    只认键的形状，不认归属）。若发送方也能替别人消费，那把键就被这条消息永久钉住，
    原主「上传了没发送」的对象再也回不到清扫任务手里。

    消费是**盖 `consumed_at` 而不是删行**（issue #233）：行是归属的唯一凭据，删掉它，删会话
    时就只剩「这个键出现在我的会话里」这种伪判据可用。行留下的副作用只是「已消费的行不再
    是清扫候选」——由判据 P（`consumed_at IS NULL AND reclaimed_at IS NULL`，见
    `_pending_upload_clause`）承担（与原先物理删除时的可见行为一致）。

    过滤里还要 P（issue #238）：一把已经被判为待回收的键（墓碑）不能被这条消息「复活」。
    少了 P，清扫已经提交删除意图、随后消息才落库，这条消息就会把一个正在（或已经被）删掉
    的对象记成正式引用，而对象存储没有回收站——这正是墓碑要挡住的那一臂。加上 P 之后，
    这条 UPDATE 命中 0 行，调用方据此知道「这把键已经不属于你了」，走拒签分支。
    """
    keys = [key for key in dict.fromkeys(object_keys or []) if isinstance(key, str) and key]
    if not keys:
        return 0
    return (
        db.query(ChatAttachmentUpload)
        .filter(
            ChatAttachmentUpload.user_id == user_id,
            ChatAttachmentUpload.object_key.in_(keys),
            _pending_upload_clause(),
        )
        .update(
            {ChatAttachmentUpload.consumed_at: datetime.now()},
            synchronize_session=False,
        )
    )


def list_pending_attachment_uploads(
    db: Session,
    *,
    older_than=None,
    user_id: int | None = None,
    limit: int | None = None,
) -> list[ChatAttachmentUpload]:
    """「桶里有、库里没有」的对账入口：本服务铸过、还没有被任何消息消费的对象键。

    这是 #142 要补的那块——上传即登记之后，行还在这张表里且未被消费就等于「服务端写过一个
    对象，但没有任何消息引用它」。清扫任务用 older_than 取超期的那批，排查/导出可以不加
    时间条件看全量，也可以按 user_id 收窄到单个账号。

    `consumed_at IS NULL` 这道过滤是 #233 引入的：消费从「删行」改成「盖时间戳」之后，
    已消费的行会留在表里，不加这道过滤就会把活消息引用的对象当成孤儿候选——比不修还糟
    （对象存储没有回收站）。这也是老代码与新代码不能同时在线的原因，见
    database/session.py 的补列分支与 scripts/backfill_attachment_owner.py 的回滚说明。

    `reclaimed_at IS NULL` 是 #238 加的，与它同属判据 P：已经判过死刑的行（墓碑）不再是
    候选——删除意图已提交，该做的是把它删完/结算，而不是当成一条新的孤儿再排一轮。
    候选里**不需要**再带 `claimed_at IS NULL`：claimed_at 只由领取写下，而领取与判据 P
    的写入同生共死（I12），P 已经把它排除干净了；真有一条租约内的行漏进来，领取那步
    赢不下来，最坏只是多算一个 skipped，不会碰对象。

    按 created_at 升序（同刻按 object_key 兜底定序）：一批超过 limit 时，先被回收的
    是积压更久的那批，且顺序稳定可复现。
    """
    query = db.query(ChatAttachmentUpload).filter(_pending_upload_clause())
    if older_than is not None:
        query = query.filter(ChatAttachmentUpload.created_at < older_than)
    if user_id is not None:
        query = query.filter(ChatAttachmentUpload.user_id == user_id)
    query = query.order_by(
        ChatAttachmentUpload.created_at.asc(),
        ChatAttachmentUpload.object_key.asc(),
    )
    if limit is not None:
        query = query.limit(max(1, int(limit)))
    return query.all()


def list_stale_reclaimed_attachment_uploads(
    db: Session, *, now, lease_seconds: int | None = None, limit: int | None = None
) -> list[str]:
    """租约已过期的墓碑键，返回键本身（issue #238，清扫候选的第二路）。

    与 `list_pending_attachment_uploads` 那张「谁还没被处理过」的名单不同，这条取的是
    「**上一轮已经开工、但没做完**」的行（判据 R）：清扫领走一行、墓碑已提交，进程在外呼
    返回之前就没了，这一行会永远停在「正在删」——待回收名单（P）不要它，因为墓碑非空；
    也没有别的路径回头碰它。不回头的代价是那个对象**永远躲在两个视角之外**：库里说它
    「已判死」，桶里却好端端躺着——正是 #142 那个形状，而且是这次修复自己造出来的。

    只取（不领）：领取是下一句条件 UPDATE 的事，中间可能被别人抢走，那时跳过即可。
    按 claimed_at 升序（同刻按 object_key 兜底）：停得越久的越先重试。

    判据落在键域上（R 不含 user_id），因此跨用户的行同样会被重试删除——对象存储没有
    回收站，谁铸的键不影响「它该被删」这个结论。
    """
    query = db.query(ChatAttachmentUpload.object_key).filter(
        _stale_reclaimed_upload_clause(now, lease_seconds)
    )
    query = query.order_by(
        ChatAttachmentUpload.claimed_at.asc(),
        ChatAttachmentUpload.object_key.asc(),
    )
    if limit is not None:
        query = query.limit(max(1, int(limit)))
    return [object_key for (object_key,) in query.all()]


def claim_attachment_upload(db: Session, object_key: str, older_than, *, now=None) -> dict | None:
    """原子地把一条「已超期且尚未结算」的登记行**判为待回收**；判不下来返回 None。

    判的动作是**条件 UPDATE + 提交**：`WHERE object_key = ? AND created_at < ? AND P`，
    `SET claimed_at = now, reclaimed_at = now`。这里刻意**不再删行**（#238 之前是条件
    DELETE）：删行是一个不可回退的既成事实，一旦提交，行没了、归属凭据没了、发送侧也无从
    知道「这个对象已经被判死」；而写两个时间戳是可判读、可结算、可回退的。

    关键在于**提交发生在碰对象之前**：墓碑一旦落库，这把键的发送就再也发不出去
    （`_reclaimed_upload_clause` 会命中它），而对象此刻还在桶里。于是顺序只剩两种结果——
    清扫先提交，则对象必被删、消息必不落库；发送先提交（消费赢了 P），则这条 UPDATE
    rowcount=0，清扫不碰对象。两者都不会留下「删掉的对象被一条活消息引用」的形态。

    并发的发送与另一条清扫竞争的是同一行，数据库的行级原子性保证只有一个能拿到
    rowcount=1；**判据只取 rowcount**，不取先前读到的任何快照（M5）。领不到的一方据此
    知道「这个对象已经归别人管了」，于是不去碰对象本身。

    返回的行数据只是给调用方做日志/计数用的上下文，不再承担「失败时放回队列」的职责
    ——那件事由 `clear_attachment_claim` 就地把同一行的两个时间戳清回去完成，不需要重建行，
    因此 created_at 保持原值不变（重建会让这条行排到队尾，且会与既有行撞主键）。
    """
    moment = datetime.now() if now is None else now
    claimed_rows = (
        db.query(ChatAttachmentUpload)
        .filter(
            ChatAttachmentUpload.object_key == object_key,
            ChatAttachmentUpload.created_at < older_than,
            _pending_upload_clause(),
        )
        .update(
            {
                ChatAttachmentUpload.claimed_at: moment,
                ChatAttachmentUpload.reclaimed_at: moment,
            },
            synchronize_session=False,
        )
    )
    db.commit()
    if not claimed_rows:
        return None
    # 领取成功后重新读一次，只是为了把上下文交给调用方：刚提交，读到的就是刚写下的值，且
    # user_id/created_at 未被本次写入改动。
    row = db.query(ChatAttachmentUpload).filter_by(object_key=object_key).first()
    if row is None:
        # 防御性的、实际不可达：这一行刚被写成「claimed_at 非空」的墓碑，而唯一会删行的 GC
        # 要求 claimed_at IS NULL，两次读之间没有哪条路径能把它清掉。留着只是让「领取成功」
        # 这件事永远有一个实打实的行作为前提，读到这行时别当成一条真实的竞态。
        return None
    return {"object_key": row.object_key, "user_id": row.user_id, "created_at": row.created_at}


def claim_stale_attachment_upload(
    db: Session, object_key: str, *, now=None, lease_seconds: int | None = None
) -> bool:
    """把一条**租约已过期**的墓碑重新领取，返回是否领到（issue #238）。

    这是给「上一轮清扫领了这一行、外呼还没回来进程就没了」准备的重试入口。它只推进
    claimed_at，**绝不改动 reclaimed_at**：墓碑是既成事实，回退它等于让发送侧凭空放行。
    判据同样只看 rowcount：行在租约之内（可能正有另一次外呼在飞）或已经结算过，都领不到。
    """
    moment = datetime.now() if now is None else now
    claimed_rows = (
        db.query(ChatAttachmentUpload)
        .filter(
            ChatAttachmentUpload.object_key == object_key,
            _stale_reclaimed_upload_clause(moment, lease_seconds),
        )
        .update({ChatAttachmentUpload.claimed_at: moment}, synchronize_session=False)
    )
    db.commit()
    return bool(claimed_rows)


def mark_attachment_reclaimed(db: Session, object_key: str) -> bool:
    """把一条墓碑标记为**已结算**：清掉租约，行留下（issue #238）。

    行必须留下。它就是「这个对象已经删掉了」在库里的唯一痕迹，发送侧靠它把迟到的那条消息
    拒掉。清掉的是 claimed_at——删除已经做完，没人还占着这一行；reclaimed_at 原样保留，
    直到墓碑过保留期由 GC 清走。

    只在墓碑上生效（B），并附 `consumed_at IS NULL`：后者是**防御性的、不可达的**——
    一条行要么被消费、要么被判为待回收，两者由同一条 P 互斥（见
    `_pending_upload_clause`），被消费过的行不可能同时带着墓碑。写上它只是让这条 UPDATE
    在任何情况下都不会去动一条归属已经结算过的行；读到这行时别把它当成一条真实约束。
    """
    settled = (
        db.query(ChatAttachmentUpload)
        .filter(
            ChatAttachmentUpload.object_key == object_key,
            _reclaimed_upload_clause(),
            ChatAttachmentUpload.consumed_at.is_(None),
        )
        .update({ChatAttachmentUpload.claimed_at: None}, synchronize_session=False)
    )
    db.commit()
    return bool(settled)


def clear_attachment_claim(db: Session, object_key: str) -> bool:
    """把一条墓碑**双清**回 pending：删除外呼失败，这一行回到待回收队列（issue #238）。

    清的是两个时间戳：claimed_at（没人再占着这一行）与 reclaimed_at（墓碑撤回，发送侧
    重新放行——对象确实还在桶里，放行才是对的）。**created_at 不动**：这条行还没被回收，
    它仍按上传时刻排队，下一轮清扫会立刻再试它，不会因为失败过一次就被排到所有新孤儿后面。
    这也正是「不再执行 `restore_attachment_upload` 的删行+重插」的原因——重插会换掉
    created_at，把一条积压已久的孤儿洗成新行（见该函数的移除说明）。

    同样附上防御性的 `consumed_at IS NULL`（不可达，理由见 `mark_attachment_reclaimed`），
    以及 `claimed_at IS NOT NULL`：只回退**当前确实被领走的**墓碑，不去动一条已经结算的行。
    """
    cleared = (
        db.query(ChatAttachmentUpload)
        .filter(
            ChatAttachmentUpload.object_key == object_key,
            _reclaimed_upload_clause(),
            ChatAttachmentUpload.claimed_at.is_not(None),
            ChatAttachmentUpload.consumed_at.is_(None),
        )
        .update(
            {
                ChatAttachmentUpload.claimed_at: None,
                ChatAttachmentUpload.reclaimed_at: None,
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return bool(cleared)


def list_blocked_attachment_uploads(db: Session, object_keys: list[str]) -> list[str]:
    """从一批键里挑出**已经是墓碑**的那些，返回键本身（issue #238）。只读，不提交。

    调用方是发送路径：`confirm_attachment_uploads` 命中 0 行之后，用它把「这把键被清扫
    判死了」从「这颗键本来就没有登记行」里区分出来——前者要拒签，后者是 #142 之前就存在的
    合法形态（老键没有登记行），照旧发送。这个分类**必须与消费在同一次事务、同一次提交前
    完成**：分成两次读，中间落进来一条墓碑，拒签就漏了。

    它只是**事后分类器**，不是判据来源：本次发送能不能放行，唯一依据是
    `confirm_attachment_uploads` 的 rowcount（M5）。先分类再更新就成了「读-再-写」，
    两次快照之间的变化会让分类结论过期。

    判据落在**键域**上：只问「这把键的登记行是不是墓碑」，不问它属于谁。附件列由客户端
    回带，一条消息可以引用一把别人铸的键，而对象删了就是删了——跨用户的差别不构成放行的
    理由（M4/I7）。
    """
    keys = [key for key in dict.fromkeys(object_keys or []) if isinstance(key, str) and key]
    if not keys:
        return []
    rows = (
        db.query(ChatAttachmentUpload.object_key)
        .filter(
            ChatAttachmentUpload.object_key.in_(keys),
            _reclaimed_upload_clause(),
        )
        .all()
    )
    return [object_key for (object_key,) in rows]


def gc_reclaimed_attachment_uploads(
    db: Session, *, older_than, limit: int | None = None
) -> int:
    """清理超过保留期的**已结算**墓碑，返回清掉的条数并提交（issue #238）。

    为什么要有 GC：墓碑是「这个键被删了」的凭据，但它的用处有时效——过了保留期，那条
    迟到的消息早就不可能来了，行留着只是占表。清掉之后这把键退回「表里查无此行」的语义，
    发送不再被拒（老键形态）。

    `claimed_at IS NULL` 是**判定条件、不是防御**（A1）：墓碑有两种形态——已结算
    （claimed_at 为空）与正被处理（claimed_at 非空）。只有前者能清。正在处理的行看上去
    也可能「很老」：重试路径每轮只刷新 claimed_at、不动 reclaimed_at，一条反复重试失败
    的行可以让 reclaimed_at 停留在很久以前（默认租约 300 秒对 7 天保留期，最多可重试约
    2000 次）。少了这个条件，GC 会把一条**此刻正在被外呼删除**的行连根清掉——行没了，
    发送侧不再拒签，那条迟到的消息就会落库并引用一个刚被删掉的对象，正是墓碑要挡的那一臂
    （M3 的静默终点状态从墓碑窗口之外被重新打开）。
    """
    keys = [
        object_key
        for (object_key,) in db.query(ChatAttachmentUpload.object_key)
        .filter(
            _reclaimed_upload_clause(),
            ChatAttachmentUpload.reclaimed_at < older_than,
            ChatAttachmentUpload.claimed_at.is_(None),
        )
        .order_by(ChatAttachmentUpload.reclaimed_at.asc(), ChatAttachmentUpload.object_key.asc())
        .limit(max(1, int(limit)) if limit is not None else 1000000)
        .all()
    ]
    if not keys:
        return 0
    # 同一条 WHERE 再来一遍：上面那次读只是挑批次，删的时候判据必须重新成立一次
    # （两次之间行可能被别的进程改过），返回的 rowcount 才是真正删掉的条数。
    deleted = (
        db.query(ChatAttachmentUpload)
        .filter(
            ChatAttachmentUpload.object_key.in_(keys),
            _reclaimed_upload_clause(),
            ChatAttachmentUpload.reclaimed_at < older_than,
            ChatAttachmentUpload.claimed_at.is_(None),
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return int(deleted)


def delete_attachment_uploads(db: Session, object_keys: list[str], requester_id: int) -> int:
    """释放（物理删掉）已经回收完成的登记行，返回释放的条数，并提交。

    这是纯粹的增长控制（issue #233）：消费改成盖 `consumed_at` 之后，行不再随发送消失，
    删会话时得有人把它们收走，否则这张表只增不减。调用方是「对象确实删掉了」的那批键
    ——行留着已经没有用处（对象不在了，归属也无从主张），留着只是垃圾。

    只释放 requester_id 自己铸的行（`user_id = ?`）：别人的行是别人的账，哪怕刚刚顺手
    删掉了它的对象，也没有资格替别人销账。这在正常路径上不会命中——回收判据本身就是
    「属主 == 请求方」。

    这里**有意不加 `consumed_at IS NOT NULL` 过滤**：本函数不做状态判定，只认调用方递进来的
    「刚刚 OSS 删除成功」的键——那批键在正常语义下必然已消费，再补一道状态过滤只是把同一件
    事重判一次；而万一有未消费的行落进来（例如同键被另一条消息引用过），不加过滤反而仍能
    把它释放，正是想要的结果。读到这行时**别顺手补上这道过滤**：那会把「按结果释放」变成
    「按状态释放」，凭空多出一个无声的失败出口（该释放的行留在表里，且不报错）。

    **逐键独立，绝不整批 `IN`**：调用方传进来的是「删成功的键」，一次整批删除会把
    「哪把键释放了、哪把没有」压成一个数字。一旦某把键其实没删成功，它的行会跟着整批
    消失，从此既不在清扫候选里（已被消费），也不在回收路径上，泄漏没有任何补偿入口。
    逐键调用换来的是失败可归因，代价可以接受：只在删会话时走一次，键的数量是人的输入规模。
    """
    released = 0
    for object_key in dict.fromkeys(object_keys or []):
        if not isinstance(object_key, str) or not object_key:
            continue
        released += (
            db.query(ChatAttachmentUpload)
            .filter(
                ChatAttachmentUpload.object_key == object_key,
                ChatAttachmentUpload.user_id == requester_id,
            )
            .delete(synchronize_session=False)
        )
    db.commit()
    return released


def delete_conversation(db: Session, cid: str, user_id: int) -> Conversation | None:
    conversation = get_conversation(db, cid, user_id)
    if not conversation:
        return None
    # 学习轨迹按 conversation_id 关联，列上没有外键约束（model/models.py），数据库不会级联，
    # 必须显式清理。与会话删除共用同一次提交：分两次提交时中间失败就会留下
    # 「会话已删、轨迹仍能读」的残留（issue #127）。不按 LEARNING_TRACE_ENABLED 分流——
    # 关掉开关只是不再写新轨迹，已经写下的行仍然要随会话删除。
    crud_trace.delete_trace_sessions_for_conversation(db, cid)
    db.delete(conversation)
    # ↓ 点无可退（issue #260）：这一行之后，会话/消息/轨迹已成事实。调用方必须把所有
    # 不在这条事务里的前置清理（checkpointer 等外部状态）放在这里之前；本行之后的动作
    # 只允许 best-effort，不得再把成功删除变成 5xx。
    db.commit()
    return conversation


def rename_conversation(db: Session, cid: str, user_id: int, title: str) -> Conversation | None:
    conversation = get_conversation(db, cid, user_id)
    if not conversation:
        return None
    conversation.title = title
    db.commit()
    db.refresh(conversation)
    return conversation


def get_message_by_user(db: Session, message_id: int, user_id: int) -> Message | None:
    return (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Message.id == message_id, Conversation.user_id == user_id)
        .first()
    )


def _attachment_keys(raw_attachments) -> list[str]:
    """从一列 JSON 附件里取出对象键；脏数据项直接跳过（与 list_conversation_attachment_owners 同款）。"""
    keys: list[str] = []
    for item in load_json_value(raw_attachments, []):
        if not isinstance(item, dict):
            continue
        object_key = item.get("object_key")
        if isinstance(object_key, str) and object_key:
            keys.append(object_key)
    return keys


def delete_message(db: Session, message_id: int, user_id: int,
                   keep_attachments: bool = False) -> list[int] | None:
    """删除一条消息，返回实际删掉的消息 id（按 id 升序）；查无此消息或不属于该用户返回 None。

    语义（issue #256）：
    - 用户消息 = **截断**：连同同一会话里 id 不小于它的后续消息一起删。这与「编辑后重发」
      是同一件事，只是这里不重新写入；
    - 助手消息 = 只删这一条，它前面的提问与后面的回答都保留。

    归属沿用 `get_message_by_user`：消息与 conversations 做 join 再按 user_id 过滤，
    别人的消息在查询里就不存在，因此对调用方只呈现 404，不会回显正文。

    一次提交里完成四件事：删消息、推进会话 `updated_at`、必要时作废记忆摘要、给不再被
    任何消息引用的附件键写下墓碑。任何一步失败都整体回滚——不存在「消息删了但侧栏没动」
    或「摘要清了但消息还在」的中间态。

    附件**不在这里删对象**（不调用 OSS / `_reclaim_chat_attachments` / `delete_attachment_uploads`）：
    请求路径只把键收敛到 #238 的墓碑状态（`claimed_at` + `reclaimed_at` 同时写下、清掉
    `consumed_at`），真正的 OSS DELETE 由既有的 `reclaim_orphan_chat_attachments` 在租约
    到期后领取执行；失败时 `clear_attachment_claim` 会把行放回队列。不能只给「还没有行」
    的键插 pending 行——真实链路上传时一定已登记、发送时盖了 `consumed_at`，那样写会让
    所有实际用过的附件都漏出回收视野（对象桶是 public-read，泄漏即长期可读）。

    `keep_attachments=True`（issue #253）：编辑/重新生成会在删除后**立即**用同一批
    `object_key` 重发，此时若照常登记墓碑，发送路径会把这些键判死而丢掉附件。该分支因此
    跳过附件墓碑这一步，完全不改 `ChatAttachmentUpload` 行（保持 consumed 状态与归属），
    消息删除、`updated_at` 推进、记忆摘要作废等其余语义与默认路径完全一致。调用方承诺
    「删完马上重发」；若重发失败，这些行会停在 consumed 且不再被任何消息引用——属已知
    泄漏（对象仍在桶里，但清扫不可见），不在本函数内补偿，由 PR 后续计划跟踪。默认
    `False` 时行为与 #311 完全一致。
    """
    message = get_message_by_user(db, message_id, user_id)
    if message is None:
        return None
    conversation = get_conversation(db, message.conversation_id, user_id)
    if conversation is None:
        # 防御性分支：上面的 join 已经保证会话属于该用户。保留它让归属判据在两条查询里
        # 都成立，读到这行时别当成一条真实的竞态。
        return None

    if message.role == "user":
        # 截断：同会话、id >= 目标消息的全部行（包括目标本身）。
        delete_condition = and_(
            Message.conversation_id == message.conversation_id,
            Message.id >= message.id,
        )
    else:
        # 助手消息只删这一条；其它角色（历史脏数据）同样按单条处理。
        delete_condition = Message.id == message.id

    rows = (
        db.query(Message.id, Message.attachments)
        .filter(delete_condition)
        .order_by(Message.id.asc())
        .all()
    )
    deleted_ids = [row_id for row_id, _raw in rows]

    # 附件键必须在删行之前收集；删完就再也读不到这些消息引用过什么。保留附件的删除
    # 不做这一步：调用方马上重发同一批键，任何墓碑登记都会让这次重发被拒签。
    orphan_keys: list[str] = []
    if not keep_attachments:
        candidate_keys = {
            object_key
            for _row_id, raw_attachments in rows
            for object_key in _attachment_keys(raw_attachments)
        }
        if candidate_keys:
            surviving_keys: set[str] = set()
            surviving_rows = (
                db.query(Message.attachments)
                .filter(Message.id.notin_(deleted_ids))
                .all()
            )
            for (raw_attachments,) in surviving_rows:
                surviving_keys.update(_attachment_keys(raw_attachments))
            orphan_keys = sorted(candidate_keys - surviving_keys)

    try:
        db.query(Message).filter(delete_condition).delete(synchronize_session=False)

        # 删除也是内容变更：侧栏按 updated_at 排序，不推进它，删完的会话仍沉在旧位置。
        # `onupdate` 只在会话行真的被 UPDATE 时触发，所以必须显式 touch（同 #239）。
        touch_conversation(db, conversation.id, user_id)

        # 记忆摘要若覆盖了任一被删消息，摘要文本里就可能残留已删内容。不做句级清洗
        # （摘要是模型生成的连续文本，按行/句裁剪只会得到一段断裂的中间态），直接作废、
        # 让下一轮重新生成；指针之前被删时摘要仍然有效，不碰。
        pointer = conversation.memory_summary_upto_message_id or 0
        if any(deleted_id <= pointer for deleted_id in deleted_ids):
            conversation.memory_summary = ""
            conversation.memory_summary_upto_message_id = 0
            conversation.memory_updated_at = None

        if orphan_keys:
            # 附件回收复用 #238 的墓碑/租约状态机，不另造一套 pending 语义（见 docstring）。
            # 同一批键共用一个时间戳：墓碑与租约起点同时写下，且整批可读作同一时刻的意图。
            now = datetime.now()
            existing_rows = (
                db.query(ChatAttachmentUpload)
                .filter(ChatAttachmentUpload.object_key.in_(orphan_keys))
                .all()
            )
            registered_keys: set[str] = set()
            for row in existing_rows:
                registered_keys.add(row.object_key)
                # 归属判据（#233）：别人铸的键不归当前用户回收，保持原状。
                if row.user_id != user_id:
                    continue
                # 已经是墓碑（可能正被清扫领取、也可能已结算）：不重写，避免缩短/抹掉
                # 既有租约状态，也避免把「删除已成」改回「正在删」。
                if row.reclaimed_at is not None:
                    continue
                # pending 或 consumed 都收敛为墓碑：清掉消费标记（使其不再被当成活跃引用），
                # 写下 claimed_at 让清扫在租约到期后重新领取并真正删除对象。
                # 行必须保留——它是归属凭据；created_at 也不动——它仍按上传时刻排队。
                row.consumed_at = None
                row.claimed_at = now
                row.reclaimed_at = now
            # 历史老键（本服务登记过但库里没有行）：直接插入一条完整墓碑。
            db.add_all([
                ChatAttachmentUpload(
                    object_key=object_key,
                    user_id=user_id,
                    created_at=now,
                    consumed_at=None,
                    claimed_at=now,
                    reclaimed_at=now,
                )
                for object_key in orphan_keys
                if object_key not in registered_keys
            ])

        db.commit()
    except Exception:
        db.rollback()
        raise
    return deleted_ids


def get_trace_session_by_message(db: Session, message_id: int, user_id: int) -> ChatTraceSession | None:
    return db.query(ChatTraceSession).filter_by(message_id=message_id, user_id=user_id).first()


def set_message_feedback(db: Session, message: Message, feedback: int) -> Message:
    # 幂等：同值重复提交不产生新行、不改 created_at，只覆盖 feedback 列。
    message.feedback = feedback
    db.commit()
    db.refresh(message)
    return message
