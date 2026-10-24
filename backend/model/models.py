import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, SmallInteger, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database.session import Base


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    avatar = Column(String(500), default="")
    # 令牌世代（issue #184）：改密时 +1，让改密前签发的全部 token 立即失效。
    # 存量库由 database/session.py 的补列迁移补上（NOT NULL DEFAULT 0），因此老用户的
    # 当前世代同样是 0，不需要额外回填脚本。
    token_version = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime, server_default=func.now())

    conversations = relationship("Conversation", back_populates="user")


class RevokedToken(Base):
    """已登出的 token 登记行，按 jti 一条（issue #184）。

    与 `users.token_version` 分工不同，两者缺一不可：token_version 是「整个用户」的世代，
    改密时递增，把该用户名下所有已签发 token 一次作废；这张表是「单枚 token」的吊销面，
    登出时只登记当前这一枚，其他会话的 token 不受影响。若登出也走 token_version，用户
    在手机上退出登录会把桌面端的会话一起踢掉。

    jti 是 token 自带的随机 id（`uuid4().hex`，32 字符），token 自身带 exp（默认最长 24h）；
    过期行不会再被任何请求命中，本仓暂不引入清理任务，将来需要回收时按 token 的 exp 删即
    可（这里不存 exp，是因为在没有回收方之前它只是一列没人读的数据）。
    """

    __tablename__ = "revoked_tokens"

    jti = Column(String(36), primary_key=True)
    created_at = Column(DateTime, server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_knowledge_bases_user_name"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 归属用户；历史数据为 NULL（不可见），由 scripts/backfill_knowledge_owner.py 回填。
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    conversations = relationship("Conversation", back_populates="knowledge_base")
    files = relationship("KnowledgeFile", back_populates="knowledge_base")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=_new_id)
    # 侧栏会话列表按 user_id 过滤、按 updated_at 排序，删除知识库时按 knowledge_base_id
    # 反查会话（crud/knowledge_base.py）：两列都没有索引时前者全表扫描 + 临时排序，
    # 后者同样整表扫一遍（issue #176）。索引名取 SQLAlchemy 默认的 ix_<表>_<列>，
    # 与启动期补建（database.session._ensure_single_column_index）同名。
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    title = Column(String(200), nullable=False)
    memory_summary = Column(Text, default="")
    memory_summary_upto_message_id = Column(Integer, default=0)
    memory_updated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="conversations")
    knowledge_base = relationship("KnowledgeBase", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation",
                            order_by="Message.created_at",
                            cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 取一页消息按 conversation_id 过滤：没有索引时只能沿主键倒序往回走，
    # 走多少行由「这页要往回多远」决定，而不是由本会话有多少条消息决定（issue #176）。
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(10), nullable=False)
    content = Column(LONGTEXT, nullable=False)
    sources = Column(Text, default="")
    attachments = Column(Text, default="")
    ragas_status = Column(String(20), default="")
    ragas_scores = Column(Text, default="")
    ragas_error = Column(Text, default="")
    retrieval_trace = Column(LONGTEXT, default="")
    # 消息级人工反馈（issue #250）：1=赞、-1=踩、0=未反馈。NOT NULL DEFAULT 0 让存量行直接
    # 落在「未反馈」，与新建消息一致，不需要回填脚本（与 users.token_version 同款语义）。
    feedback = Column(SmallInteger, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime, server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 归属用户；历史数据为 NULL（不可见），由 scripts/backfill_knowledge_owner.py 回填。
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    # 列表与删除链路都按该列过滤，计数聚合（count_knowledge_files_by_base）也按它
    # GROUP BY：没有索引时这些查询对整个 knowledge_files 表扫一遍（issue #83 第 5 项）。
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    size = Column(Integer, nullable=False)
    content = Column(LONGTEXT, default="")
    created_at = Column(DateTime, server_default=func.now())

    knowledge_base = relationship("KnowledgeBase", back_populates="files")
    # 切片元数据行（issue #287）。删除文件时一并清掉，避免删知识库等多条删除路径漏行；
    # 批量删除路径（crud.knowledge_file.delete_knowledge_file）仍显式先删元数据，
    # 不依赖这里按行懒加载。
    chunk_metadata = relationship(
        "KnowledgeChunkMetadata",
        back_populates="knowledge_file",
        cascade="all, delete-orphan",
    )


class KnowledgeChunkMetadata(Base):
    """每个入库切片的结构元数据（issue #287）。

    Milvus collection schema 保持不变，结构信息只落 MySQL 侧：章节路径、条款编号、
    表格行列与 token 数。老文件没有对应行时检索照常走原文本路径，不要求重建；
    #289 的父子切片会复用这里的 parent_id / parent_text。
    """

    __tablename__ = "knowledge_chunk_metadata"
    __table_args__ = (
        UniqueConstraint("file_id", "chunk_id", name="uq_knowledge_chunk_metadata_file_chunk"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(
        Integer,
        ForeignKey("knowledge_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 冗余知识库归属，便于按库排查；不做外键，避免删库顺序影响清理路径。
    knowledge_base_id = Column(Integer, nullable=True, index=True)
    # 与 Milvus 主键里的 chunk 段一致（chunk_text 返回的字符串序号）。
    chunk_id = Column(String(64), nullable=False)
    # 留给 #289 的父子切片；本单恒为 NULL。
    parent_id = Column(String(64), nullable=True)
    section_path = Column(String(1000), default="")
    article_no = Column(String(64), default="")
    token_count = Column(Integer, nullable=False, default=0)
    table_id = Column(Integer, nullable=True)
    row_no = Column(Integer, nullable=True)
    parent_text = Column(LONGTEXT, nullable=True)
    metadata_json = Column(LONGTEXT, default="")
    created_at = Column(DateTime, server_default=func.now())

    knowledge_file = relationship("KnowledgeFile", back_populates="chunk_metadata")


class ChatAttachmentUpload(Base):
    """上传接口铸出的对象键的登记行：发送成功即被消费，超期未消费的由清扫任务回收。

    这张表同时是「桶里有、库里没有」的对账入口——issue #142 里完全缺失的那块：上传只把
    对象写进 OSS 就返回键，未发送的对象不在任何消息里，没有任何路径能看见它。

    这张表还是附件归属的唯一凭据（issue #233）：`user_id` 是**铸键人**，也就是这个键的
    属主。发送成功不再物理删行，而是盖上 `consumed_at`——行在，归属就在，删会话时才有
    依据判断「这把键是不是该由我回收」；行没了就只能退回「键出现在我的会话里」这种
    伪判据，任何登录用户都能借别人的会话删别人的对象。

    行还记录了清扫侧的状态（issue #238）：`claimed_at` 是租约起点，`reclaimed_at` 是
    删除意图的墓碑。三者互斥共用一张行状态机——pending（三列皆空）、deleting（claimed_at
    与 reclaimed_at 皆非空）、reclaimed（只有 reclaimed_at 非空，删除已成）、consumed
    （consumed_at 非空）。`consumed_at` 与 `reclaimed_at` 不可能同时非空：一条已被消息
    消费的对象绝不会再被判为孤儿，反之亦然。
    """

    __tablename__ = "chat_attachment_uploads"

    # 键的形态由 oss_service.SERVICE_MINTED_KEY_PATTERN 定死（约 90 个字符），255 是余量。
    object_key = Column(String(255), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    # 时间戳走应用时钟（而不是 server_default=func.now()）：清扫任务按 Python 的
    # datetime.now() 算保留窗口，两边同一个时钟才能让「超过 TTL」这个判据可预期。
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    # 被某条消息消费的时刻（issue #233）。NULL = 还没被任何消息引用，是清扫任务的候选；
    # 非 NULL = 已被引用，不再参与清扫，但行要留着当归属凭据。
    # 存量库由 database/session.py 的补列迁移补上（可空、默认 NULL），因此不需要单独的
    # 回填脚本就能让写入侧跑起来；**归属**的推定回填见 scripts/backfill_attachment_owner.py。
    consumed_at = Column(DateTime, nullable=True, default=None)
    # 租约起点（issue #238）：清扫任务把这一行「领走」的时刻，只用来给一次删除外呼计时。
    # 它的含义**不是**「发送可以通行」——恰恰相反，同一时刻写下的 reclaimed_at 才是判据。
    # NULL = 没有人在处理这一行。非 NULL 时必有 reclaimed_at 非 NULL（两者只由
    # claim_attachment_upload 在同一句 UPDATE 里一起写下），所以「claimed_at 非空」本身就
    # 蕴含「已被判为待回收」，任何人都不能靠比较它与当前时间来放行一次发送。
    claimed_at = Column(DateTime, nullable=True, default=None)
    # 删除意图已提交的时刻（issue #238，墓碑）：清扫赢下这一行之后、**碰对象之前**写下，
    # 之后才去外呼 OSS DELETE。它与 claimed_at 同一次写入，但两者生命周期不同——删除成功
    # 只清 claimed_at（行作为既成事实的墓碑留下，让这把键的发送被拒），只有墓碑过了保留期
    # 才由 GC 连行一起清掉。发送侧的拒签只读这一列，不读 claimed_at（I13）。
    # 存量库同样由 database/session.py 的补列迁移补上（可空、默认 NULL），不需要回填脚本。
    reclaimed_at = Column(DateTime, nullable=True, default=None)


class ChatTraceSession(Base):
    __tablename__ = "chat_trace_sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    conversation_id = Column(String(36), nullable=True)
    message_id = Column(Integer, nullable=True)
    status = Column(String(20), default="running")
    events = Column(LONGTEXT, default="[]")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
