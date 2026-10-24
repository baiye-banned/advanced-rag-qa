import logging

from sqlalchemy import create_engine
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from config import DATABASE_URL, MYSQL_CONNECT_ARGS

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args=MYSQL_CONNECT_ARGS, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns()
    _ensure_default_knowledge_base()


def _ensure_schema_columns():
    _ensure_mysql_utf8mb4()

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "messages" in table_names:
        columns = {column["name"] for column in inspector.get_columns("messages")}
        if "sources" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN sources TEXT"))
        if "attachments" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN attachments TEXT"))
        if "ragas_status" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN ragas_status VARCHAR(20)"))
        if "ragas_scores" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN ragas_scores TEXT"))
        if "ragas_error" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN ragas_error TEXT"))
        if "retrieval_trace" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE messages ADD COLUMN retrieval_trace LONGTEXT"))
        # 消息级反馈（issue #250）。NOT NULL DEFAULT 0 让存量消息落在「未反馈」，
        # 与新建消息一致，无需回填脚本（同 users.token_version）。
        if "feedback" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE messages ADD COLUMN feedback SMALLINT NOT NULL DEFAULT 0")
                )
        _ensure_mysql_text_column("messages", "content", "LONGTEXT", nullable=False)
        _ensure_mysql_varchar_column("messages", "role", 10, nullable=False)
        _ensure_mysql_text_column("messages", "sources", "TEXT")
        _ensure_mysql_text_column("messages", "attachments", "TEXT")
        _ensure_mysql_varchar_column("messages", "ragas_status", 20)
        _ensure_mysql_text_column("messages", "ragas_scores", "TEXT")
        _ensure_mysql_text_column("messages", "ragas_error", "TEXT")
        _ensure_mysql_text_column("messages", "retrieval_trace", "LONGTEXT")
        # conversation_id 早于本次改动就已存在，所以不能挂在上面那个「刚加列」的分支里。
        _ensure_single_column_index("messages", "conversation_id", "ix_messages_conversation_id")

    if "conversations" in table_names:
        columns = {column["name"] for column in inspector.get_columns("conversations")}
        if "knowledge_base_id" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN knowledge_base_id INTEGER"))
        if "memory_summary" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN memory_summary TEXT"))
        if "memory_summary_upto_message_id" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN memory_summary_upto_message_id INTEGER DEFAULT 0"))
        if "memory_updated_at" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN memory_updated_at DATETIME"))
        _ensure_mysql_varchar_column("conversations", "id", 36, nullable=False)
        _ensure_mysql_varchar_column("conversations", "title", 200, nullable=False)
        _ensure_mysql_text_column("conversations", "memory_summary", "TEXT")
        # 同样是先于本次改动就存在的列：老库只补列、不补索引。
        _ensure_single_column_index("conversations", "user_id", "ix_conversations_user_id")
        _ensure_single_column_index(
            "conversations", "knowledge_base_id", "ix_conversations_knowledge_base_id"
        )

    if "knowledge_files" in table_names:
        columns = {column["name"] for column in inspector.get_columns("knowledge_files")}
        if "knowledge_base_id" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN knowledge_base_id INTEGER"))
        if "user_id" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN user_id INTEGER"))
                conn.execute(text("CREATE INDEX ix_knowledge_files_user_id ON knowledge_files (user_id)"))
        # knowledge_base_id 早于本次改动就已存在，所以不能挂在上面那个「刚加列」的分支里。
        _ensure_single_column_index(
            "knowledge_files", "knowledge_base_id", "ix_knowledge_files_knowledge_base_id"
        )
        _ensure_mysql_varchar_column("knowledge_files", "name", 255, nullable=False)
        _ensure_mysql_text_column("knowledge_files", "content", "LONGTEXT")

    if "knowledge_bases" in table_names:
        columns = {column["name"] for column in inspector.get_columns("knowledge_bases")}
        if "user_id" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE knowledge_bases ADD COLUMN user_id INTEGER"))
                conn.execute(text("CREATE INDEX ix_knowledge_bases_user_id ON knowledge_bases (user_id)"))
        _ensure_mysql_varchar_column("knowledge_bases", "name", 100, nullable=False)
        _ensure_knowledge_base_owner_unique_index()

    if "users" in table_names:
        columns = {column["name"] for column in inspector.get_columns("users")}
        # 令牌世代（issue #184）。NOT NULL DEFAULT 0 让存量行直接落在世代 0，与新建用户
        # 一致，不需要单独的回填脚本。
        if "token_version" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
                )
        _ensure_mysql_varchar_column("users", "username", 50, nullable=False)
        _ensure_mysql_varchar_column("users", "password_hash", 255, nullable=False)
        _ensure_mysql_varchar_column("users", "avatar", 500)

    if "knowledge_chunk_metadata" in table_names:
        # 新表由 create_all 建；老库升级时这张表同样由 create_all 补建。这里兜住的是
        # 「表已存在但索引/列类型不对」的形态：缺索引会让按 file_id 的清理变成全表扫描，
        # 缺唯一键会让重复上传累积同一 chunk 的多行。
        _ensure_single_column_index(
            "knowledge_chunk_metadata",
            "file_id",
            "ix_knowledge_chunk_metadata_file_id",
        )
        _ensure_single_column_index(
            "knowledge_chunk_metadata",
            "knowledge_base_id",
            "ix_knowledge_chunk_metadata_knowledge_base_id",
        )
        _ensure_chunk_metadata_unique_index()
        _ensure_mysql_text_column("knowledge_chunk_metadata", "parent_text", "LONGTEXT")
        _ensure_mysql_text_column("knowledge_chunk_metadata", "metadata_json", "LONGTEXT")

    if "chat_trace_sessions" in table_names:
        columns = {column["name"] for column in inspector.get_columns("chat_trace_sessions")}
        _ensure_mysql_varchar_column("chat_trace_sessions", "id", 36, nullable=False)
        _ensure_mysql_varchar_column("chat_trace_sessions", "status", 20)
        if "events" in columns:
            _ensure_mysql_text_column("chat_trace_sessions", "events", "LONGTEXT")

    if "chat_attachment_uploads" in table_names:
        columns = {column["name"] for column in inspector.get_columns("chat_attachment_uploads")}
        # 消费时刻（issue #233）。可空、默认 NULL：存量行落成 NULL 恰好等于「还没被消费」
        # ——也就是旧代码把这行物理删掉时它在清扫任务眼里的状态，语义上严格向后兼容，
        # 因此不需要单独的回填脚本（归属的推定回填是另一件事，见
        # scripts/backfill_attachment_owner.py）。上线顺序见该 issue 的回滚方案：只要还有
        # 实例在跑旧代码，库里就不得存在 consumed_at IS NOT NULL 的行——旧代码的
        # list_pending_attachment_uploads 没有这列上的过滤，会把已消费的留存行当成孤儿候选。
        if "consumed_at" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_attachment_uploads ADD COLUMN consumed_at DATETIME"))
        # 清扫侧的租约起点与删除意图墓碑（issue #238）。同样可空、默认 NULL：存量行落成
        # 「三列皆空」＝ pending，正是新代码眼里「尚未被处理过的待回收行」，语义向后兼容，
        # 不需要回填脚本。上线顺序同理——只要还有实例在跑旧代码，库里就不得存在
        # reclaimed_at IS NOT NULL 的行：旧代码不看这列，会把已经「判过死刑」的键当成普通
        # 待回收行再签一次 DELETE（对象存储没有回收站，重复删除虽幂等，但更糟的是旧代码没有
        # 墓碑可拒签，那把键的发送会照着旧逻辑通行，落库的消息引用一个已被删掉的对象）。
        if "claimed_at" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_attachment_uploads ADD COLUMN claimed_at DATETIME"))
        if "reclaimed_at" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_attachment_uploads ADD COLUMN reclaimed_at DATETIME"))


def _ensure_mysql_utf8mb4():
    if engine.dialect.name != "mysql":
        return

    database_name = engine.url.database
    if database_name:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f"ALTER DATABASE `{database_name}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
        except Exception:
            # Existing databases may not allow ALTER DATABASE on all hosts; continue with table conversion.
            pass

    inspector = inspect(engine)
    for table_name in (
        "users",
        "knowledge_bases",
        "conversations",
        "messages",
        "knowledge_files",
        "knowledge_chunk_metadata",
        "chat_trace_sessions",
        "chat_attachment_uploads",
        "revoked_tokens",
    ):
        if table_name not in inspector.get_table_names():
            continue
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f"ALTER TABLE `{table_name}` "
                        "CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
        except Exception:
            # Table conversion can fail on partially migrated schemas; keep startup alive and
            # let the explicit column migration below handle the critical fields.
            pass


def _ensure_mysql_text_column(
    table_name: str,
    column_name: str,
    sql_type: str,
    nullable: bool | None = None,
):
    _ensure_mysql_character_column(table_name, column_name, sql_type, nullable=nullable)


def _ensure_mysql_varchar_column(
    table_name: str,
    column_name: str,
    length: int,
    nullable: bool | None = None,
):
    _ensure_mysql_character_column(table_name, column_name, f"VARCHAR({length})", nullable=nullable)


def _ensure_mysql_character_column(
    table_name: str,
    column_name: str,
    sql_type: str,
    nullable: bool | None = None,
):
    column = _get_mysql_column_info(table_name, column_name)
    if not column:
        return
    normalized_type = str(sql_type).split("(", 1)[0].lower()
    if (
        str(column.get("data_type", "")).lower() == normalized_type
        and str(column.get("character_set_name") or "").lower() == "utf8mb4"
    ):
        return
    effective_nullable = nullable
    if effective_nullable is None:
        effective_nullable = str(column.get("is_nullable") or "").upper() == "YES"
    null_clause = " NULL" if effective_nullable else " NOT NULL"
    with engine.begin() as conn:
        conn.execute(
            text(
                f"ALTER TABLE `{table_name}` "
                f"MODIFY COLUMN `{column_name}` {sql_type} "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci{null_clause}"
            )
        )


def _ensure_single_column_index(table_name: str, column_name: str, index_name: str) -> None:
    """给**已存在**的表补一条单列索引。

    模型里的 index=True 只覆盖 create_all 新建的表；早先建好的库还得显式补，
    否则升级后计数聚合仍是全表扫描。按 inspector 查到的列组合判断是否已存在，
    不依赖 CREATE INDEX IF NOT EXISTS（MySQL 不支持）。
    """
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            return
        indexed_columns = {
            tuple(index.get("column_names") or []) for index in inspector.get_indexes(table_name)
        }
        if (column_name,) in indexed_columns:
            return
        with engine.begin() as conn:
            conn.execute(text(f"CREATE INDEX `{index_name}` ON `{table_name}` ({column_name})"))
    except Exception:
        # 建索引失败不阻塞启动：这些查询退化为全表扫描，结果仍然正确。
        logger.warning(
            "Failed to ensure index %s on %s(%s)", index_name, table_name, column_name, exc_info=True
        )


def _ensure_chunk_metadata_unique_index() -> None:
    """补 (file_id, chunk_id) 唯一键（issue #287）。

    新表由模型里的 UniqueConstraint 在 create_all 时建好；这里只处理「表已在、唯一键
    丢了」的老库。不同后端经 inspector 读到的是唯一约束或唯一索引，两种都认。
    建失败不阻塞启动：重复行只会让 replace_chunk_metadata 的查询多返回几行，
    写入侧每次都会先按 file_id 清空再插入。
    """
    table_name = "knowledge_chunk_metadata"
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            return
        columns = ("file_id", "chunk_id")
        unique_sets = {
            tuple(constraint.get("column_names") or [])
            for constraint in inspector.get_unique_constraints(table_name)
        }
        unique_sets.update(
            tuple(index.get("column_names") or [])
            for index in inspector.get_indexes(table_name)
            if index.get("unique")
        )
        if columns in unique_sets:
            return
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX `uq_knowledge_chunk_metadata_file_chunk` "
                    f"ON `{table_name}` (file_id, chunk_id)"
                )
            )
    except Exception:
        logger.warning(
            "Failed to ensure the (file_id, chunk_id) unique index on %s",
            table_name,
            exc_info=True,
        )


def _ensure_knowledge_base_owner_unique_index():
    """知识库名称改为按归属用户唯一：先摘掉旧的全局唯一索引，再补 (user_id, name)。

    历史数据里 user_id 为 NULL 的行在唯一索引里互不冲突，因此回填前后都不会报错。
    """
    if engine.dialect.name != "mysql" or not _get_mysql_column_info("knowledge_bases", "user_id"):
        return
    try:
        with engine.begin() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT INDEX_NAME AS index_name,
                               GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS index_columns
                        FROM information_schema.STATISTICS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = 'knowledge_bases'
                          AND NON_UNIQUE = 0
                        GROUP BY INDEX_NAME
                        """
                    )
                )
                .mappings()
                .all()
            )
            for row in rows:
                index_columns = str(row["index_columns"] or "").lower()
                if index_columns == "name":
                    conn.execute(text(f"ALTER TABLE `knowledge_bases` DROP INDEX `{row['index_name']}`"))
                    rows = [item for item in rows if item["index_name"] != row["index_name"]]
            if not any(str(row["index_columns"] or "").lower() == "user_id,name" for row in rows):
                conn.execute(
                    text(
                        "ALTER TABLE `knowledge_bases` "
                        "ADD UNIQUE KEY `uq_knowledge_bases_user_name` (user_id, name)"
                    )
                )
    except Exception:
        # 迁移失败不阻塞启动；即使仍是旧的全局唯一索引，归属过滤依然生效。
        logger.warning("Failed to switch knowledge_bases to a per-owner unique index", exc_info=True)


def _get_mysql_column_info(table_name: str, column_name: str):
    if engine.dialect.name != "mysql":
        return None
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT DATA_TYPE AS data_type,
                       CHARACTER_SET_NAME AS character_set_name,
                       COLLATION_NAME AS collation_name,
                       IS_NULLABLE AS is_nullable
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table_name
                  AND COLUMN_NAME = :column_name
                """
            ),
            {"table_name": table_name, "column_name": column_name},
        )
        return result.mappings().first()


def _ensure_default_knowledge_base():
    if "knowledge_bases" not in inspect(engine).get_table_names():
        return
    from model.models import KnowledgeBase

    db = SessionLocal()
    try:
        default_base = db.query(KnowledgeBase).order_by(KnowledgeBase.id.asc()).first()
        if not default_base:
            default_base = KnowledgeBase(name="默认知识库")
            db.add(default_base)
            db.commit()
            db.refresh(default_base)

        # 只把悬挂引用挂到「无归属」的历史全局默认库上：knowledge_bases 里的库一旦有归属，
        # 把别人的会话/文件改绑过去就是跨用户写入，宁可留空（聊天会按用户重新解析默认库，
        # 无归属文件继续不可见并由 scripts/backfill_knowledge_owner.py 回填）。
        if default_base.user_id is not None:
            logger.warning(
                "Skip rebinding orphan rows: knowledge base %s already belongs to user %s",
                default_base.id,
                default_base.user_id,
            )
            return

        default_id = default_base.id
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE conversations SET knowledge_base_id = :kid WHERE knowledge_base_id IS NULL"),
                {"kid": default_id},
            )
            conn.execute(
                text("UPDATE knowledge_files SET knowledge_base_id = :kid WHERE knowledge_base_id IS NULL"),
                {"kid": default_id},
            )
    finally:
        db.close()
