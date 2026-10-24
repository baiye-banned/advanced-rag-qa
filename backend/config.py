import logging
import os
import secrets
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / ".env.development")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


_logger = logging.getLogger(__name__)


# MySQL
# 下面的字面量是随仓库公开的占位口令，任何人都读得到，因此绝不能成为部署真正拿去连库的口令。
# 与 SECRET_KEY 同一治理强度：未配置或仍是占位值时，ensure_mysql_password_configured() 会在
# 启动路径上拒绝启动（见 main.lifespan）。口令的解析必须发生在 DATABASE_URL 拼接之前，所以这组
# 定义落在 MySQL 段内，而 SECRET_KEY 的对应定义在文件末尾——放置位置不同，治理策略相同。
DEFAULT_MYSQL_PASSWORD = "change-me"
ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD = _env_bool("ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD", False)

MYSQL_PASSWORD_SOURCE_ENV = "env"  # scan-secrets:allow source label, not a credential
MYSQL_PASSWORD_SOURCE_INSECURE_DEV = "insecure-dev"  # scan-secrets:allow source label, not a credential
MYSQL_PASSWORD_SOURCE_UNCONFIGURED = "unconfigured"  # scan-secrets:allow source label, not a credential

MYSQL_PASSWORD_ERROR = (
    "MYSQL_PASSWORD 未配置，或仍是仓库内置的公开占位值，服务拒绝启动。\n"
    "请在环境变量或 .env 中设置一个足够长的随机口令，并把同一口令设到 MySQL 对应账号上，例如：\n"
    "  python -c \"import secrets; print(secrets.token_urlsafe(24))\"\n"
    "仅本地开发可临时设置 ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD=true 放行：该开关让进程用默认\n"
    "占位口令连库，任何拿到本仓库的人都能用同一口令连上你的数据库，禁止用于对外可访问的部署。"
)


class MysqlPasswordError(RuntimeError):
    """Raised at startup when MYSQL_PASSWORD is missing or still the public placeholder."""


def resolve_mysql_password(password, allow_insecure_default):
    """Resolve the database password into ``(password, source)``.

    The repository placeholder is never returned as a usable password: when the value
    is missing or still the placeholder, an unguessable random one is generated
    instead, so a connection made by code that skips the startup check fails outright
    rather than succeeding with a publicly known credential.
    """
    candidate = (password or "").strip()
    if candidate and candidate != DEFAULT_MYSQL_PASSWORD:
        return candidate, MYSQL_PASSWORD_SOURCE_ENV
    if allow_insecure_default:
        _logger.warning(
            "MYSQL_PASSWORD is not configured; ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD=true so "
            "this process connects with the public repository placeholder. Set a real password "
            "and never use this for a deployment reachable by others."
        )
        return DEFAULT_MYSQL_PASSWORD, MYSQL_PASSWORD_SOURCE_INSECURE_DEV
    _logger.warning(
        "MYSQL_PASSWORD is not configured or is still the public repository placeholder. "
        "The process will refuse to start; set MYSQL_PASSWORD to a random value."
    )
    return secrets.token_urlsafe(32), MYSQL_PASSWORD_SOURCE_UNCONFIGURED


def mysql_password_is_usable() -> bool:
    """True when the process has a non-placeholder password to connect with.

    That is either an explicitly configured MYSQL_PASSWORD, or the local-dev escape
    hatch. Only the unconfigured case is unusable.
    """
    return MYSQL_PASSWORD_SOURCE != MYSQL_PASSWORD_SOURCE_UNCONFIGURED


def ensure_mysql_password_configured() -> None:
    """Fail fast when the database would be reached with a publicly known password.

    Called from the application startup path, independently of the SECRET_KEY gate: a
    deployment that forgot to set MYSQL_PASSWORD must not silently serve traffic on
    credentials anyone with a copy of this repository can guess.
    """
    if not mysql_password_is_usable():
        raise MysqlPasswordError(MYSQL_PASSWORD_ERROR)


MYSQL_PASSWORD, MYSQL_PASSWORD_SOURCE = resolve_mysql_password(
    os.getenv("MYSQL_PASSWORD"), ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD
)

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_system")
MYSQL_SSL_MODE = os.getenv("MYSQL_SSL_MODE", "").strip().lower()
MYSQL_SSL_CA = os.getenv("MYSQL_SSL_CA", "").strip()

DATABASE_URL = (
    f"mysql+pymysql://{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}@"
    f"{MYSQL_HOST}:{MYSQL_PORT}/{quote_plus(MYSQL_DATABASE)}?charset=utf8mb4"
)

MYSQL_CONNECT_ARGS = {}
if MYSQL_SSL_CA:
    MYSQL_CONNECT_ARGS["ssl"] = {"ca": MYSQL_SSL_CA}
elif MYSQL_SSL_MODE in {"required", "require", "true", "1"}:
    MYSQL_CONNECT_ARGS["ssl"] = {}

# Milvus Lite / Milvus server
REBUILD_KNOWLEDGE_INDEX_ON_STARTUP = _env_bool("REBUILD_KNOWLEDGE_INDEX_ON_STARTUP", False)
MILVUS_LITE_URI = os.getenv("MILVUS_LITE_URI", "./milvus.db")
MILVUS_URI = os.getenv("MILVUS_URI", MILVUS_LITE_URI)
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "")

# SQLite checkpointer
CHECKPOINTER_DB_PATH = os.getenv("CHECKPOINTER_DB_PATH", "./checkpointer.db")

# Bootstrap accounts. When disabled, seed_default_users() creates nothing.
# Account passwords come from SEED_<USERNAME>_PASSWORD and are never hardcoded.
SEED_DEFAULT_USERS = _env_bool("SEED_DEFAULT_USERS", True)

# Knowledge base upload
KNOWLEDGE_UPLOAD_MAX_MB = _env_int("KNOWLEDGE_UPLOAD_MAX_MB", 10)

# DeepSeek OpenAI-compatible chat API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseekv4flash")

# OpenAI-compatible vision chat API for image QA.
# Defaults target Alibaba Cloud Model Studio and reuse DASHSCOPE_API_KEY.
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
VISION_API_KEY = os.getenv("VISION_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.6-plus")
VISION_OSS_URL_EXPIRES_SECONDS = _env_int("VISION_OSS_URL_EXPIRES_SECONDS", 3600)

# OpenAI-compatible text fallback for final answers when DeepSeek is unavailable.
TEXT_FALLBACK_ENABLED = _env_bool("TEXT_FALLBACK_ENABLED", True)
TEXT_FALLBACK_BASE_URL = os.getenv(
    "TEXT_FALLBACK_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
TEXT_FALLBACK_API_KEY = os.getenv("TEXT_FALLBACK_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
TEXT_FALLBACK_MODEL = os.getenv("TEXT_FALLBACK_MODEL", "qwen3.6-plus")

DEFAULT_EMBEDDING_TOKEN_SAFETY_RATIO = 0.85  # scan-secrets:allow numeric ratio, not a credential


def normalize_token_safety_ratio(value, default: float = DEFAULT_EMBEDDING_TOKEN_SAFETY_RATIO) -> float:
    """Clamp ``EMBEDDING_TOKEN_SAFETY_RATIO`` into ``(0, 1]``.

    A ratio of 0 shrinks the packing budget to a single token and fails every batch;
    a ratio above 1 removes the safety margin the provider tokenizer needs. Both are
    configuration mistakes, so warn and use a usable value instead of honoring them.
    """
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        _logger.warning(
            "EMBEDDING_TOKEN_SAFETY_RATIO value %r is not a number; using default %s", value, default
        )
        return default
    if ratio <= 0:
        _logger.warning(
            "EMBEDDING_TOKEN_SAFETY_RATIO value %s must be in (0, 1]; using the default %s "
            "instead of shrinking the packing budget to nothing",
            ratio,
            default,
        )
        return default
    if ratio > 1:
        _logger.warning(
            "EMBEDDING_TOKEN_SAFETY_RATIO value %s exceeds 1; clamping to 1.0 so requests cannot "
            "overrun the model token limit",
            ratio,
        )
        return 1.0
    return ratio


# OpenAI-compatible embedding API for semantic retrieval and RAGAS.
# Defaults target DashScope text-embedding-v4. If EMBEDDING_API_KEY is not
# set, reuse DASHSCOPE_API_KEY from Alibaba Cloud Model Studio.
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)
# Single embedding request timeout, in seconds. Retrieval issues up to 9 route
# queries per round, so a 60s default can stall the whole round; keep it at
# online-QA scale. Bulk ingestion has its own budget below and ignores this one.
EMBEDDING_TIMEOUT_SECONDS = _env_int("EMBEDDING_TIMEOUT_SECONDS", 10)

# Bulk ingestion packs a whole document into several embedding requests. The provider
# (DashScope text-embedding-v4) accepts at most 10 inputs per request and at most 8192
# tokens per input. Batching therefore has to respect both the item cap and a token
# budget; the old 64-item / 60000-character defaults produced requests the provider
# rejects with 400 (issue #286).
#
# Token counts come from tiktoken's cl100k_base encoding, a deterministic local
# estimate -- not character counts dressed up as tokens. The provider's tokenizer is
# not public, so the budget is scaled by a safety ratio to leave headroom.
EMBEDDING_MAX_BATCH_ITEMS = _env_int("EMBEDDING_MAX_BATCH_ITEMS", 10)
EMBEDDING_MAX_REQUEST_TOKENS = _env_int("EMBEDDING_MAX_REQUEST_TOKENS", 8192)
EMBEDDING_TOKEN_SAFETY_RATIO = normalize_token_safety_ratio(
    _env_float("EMBEDDING_TOKEN_SAFETY_RATIO", 0.85)
)
# tiktoken's BPE cache directory for cl100k_base. Empty means tiktoken's default
# (`<tempdir>/data-gym-cache`). A clean or network-restricted deployment must point
# this at a directory pre-populated with cl100k_base.tiktoken; otherwise the first
# ingest tries to download the BPE file and fails. Startup performs a non-fatal
# prewarm and logs the effective directory (see rag/milvus_client.py).
TIKTOKEN_CACHE_DIR = os.getenv("TIKTOKEN_CACHE_DIR", "").strip()
# Extra attempts after the initial request, for 429/5xx only. Permanent errors such as
# 400 are not retried. Retries stay inside the whole-document budget enforced by
# KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS, so they cannot turn a timeout into success.
EMBEDDING_MAX_RETRIES = _env_int("EMBEDDING_MAX_RETRIES", 2)

# Legacy knobs, kept so existing deployments keep working. The effective item cap is
# min(EMBEDDING_MAX_BATCH_ITEMS, EMBEDDING_INGEST_BATCH_SIZE), so a legacy value can
# only tighten -- never exceed -- the model capability. The legacy character cap is an
# additional constraint when positive; 0 disables it. Neither default reproduces the
# 64-item / 60000-character request that the provider rejects.
EMBEDDING_INGEST_BATCH_SIZE = _env_int("EMBEDDING_INGEST_BATCH_SIZE", EMBEDDING_MAX_BATCH_ITEMS)
EMBEDDING_INGEST_BATCH_MAX_CHARS = _env_int("EMBEDDING_INGEST_BATCH_MAX_CHARS", 0)

# Bulk ingestion keeps its own timeout instead of sharing the online-QA one above.
# A batch now carries at most EMBEDDING_MAX_BATCH_ITEMS chunks (10 by default), so the
# per-request body is much smaller than the old 64-chunk shape; 60s still covers a slow
# round trip. Ingestion runs on a worker thread, so a slow batch does not freeze the
# event loop. Raising it only extends how long one stuck batch holds a pool slot.
EMBEDDING_INGEST_TIMEOUT_SECONDS = _env_int("EMBEDDING_INGEST_TIMEOUT_SECONDS", 60)

# Ingestion runs on a thread pool of its own rather than the asyncio default
# executor that retrieval shares. A round of retrieval fans out up to nine
# concurrent recalls through asyncio.to_thread; without the split, uploads that
# fill the default pool queue every recall behind them. Keep this pool small --
# it bounds how many uploads can hold an embedding connection at once.
KNOWLEDGE_INDEX_MAX_WORKERS = _env_int("KNOWLEDGE_INDEX_MAX_WORKERS", 4)

# Whole-document budget for one upload's ingest. EMBEDDING_INGEST_TIMEOUT_SECONDS
# above only caps a single batch; a document is split into several batches, so
# without a total budget one upload can hold an ingest slot indefinitely. The
# budget is measured from the start of the upload and shared by every step of
# that document (parse, chunk, embed), not restarted per step.
KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS = _env_int("KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS", 300)

# Dedicated reranker for retrieved chunks.
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "dashscope")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
RERANK_BASE_URL = os.getenv(
    "RERANK_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
)
RERANK_API_KEY = os.getenv("RERANK_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
RERANK_TIMEOUT_SECONDS = _env_int("RERANK_TIMEOUT_SECONDS", 30)
RERANK_LLM_FALLBACK_ENABLED = _env_bool("RERANK_LLM_FALLBACK_ENABLED", True)

# Online RAGAS evaluation.
RAGAS_ENABLED = _env_bool("RAGAS_ENABLED", False)
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", DEEPSEEK_MODEL)
RAGAS_TIMEOUT_SECONDS = _env_int("RAGAS_TIMEOUT_SECONDS", 180)
RAGAS_METRIC_TIMEOUT_SECONDS = _env_int("RAGAS_METRIC_TIMEOUT_SECONDS", 60)
RAGAS_MAX_CONTEXTS = _env_int("RAGAS_MAX_CONTEXTS", 3)
RAGAS_MAX_CONTEXT_CHARS = _env_int("RAGAS_MAX_CONTEXT_CHARS", 1500)
RAGAS_MAX_ANSWER_CHARS = _env_int("RAGAS_MAX_ANSWER_CHARS", 2000)

# Multi-route retrieval.
RETRIEVAL_ROUTE_TOP_K = _env_int("RETRIEVAL_ROUTE_TOP_K", 8)
RETRIEVAL_RERANK_TOP_N = _env_int("RETRIEVAL_RERANK_TOP_N", 5)

# BM25 keyword ranking (issue #288). When enabled, the keyword route is ordered by
# BM25 instead of the legacy hit-count score; the legacy score is still returned in
# every candidate for final-selection compatibility. The DF corpus is every keyword
# window of the knowledge base, so it can be large: BM25_MAX_DOCS caps how many
# windows may be indexed. Exceeding the cap degrades to the legacy ordering and is
# recorded in the retrieval trace; it never quietly uses the candidate-set DF.
BM25_ENABLED = _env_bool("BM25_ENABLED", True)
BM25_K1 = _env_float("BM25_K1", 1.5)
BM25_B = _env_float("BM25_B", 0.75)
BM25_MAX_DOCS = _env_int("BM25_MAX_DOCS", 20000)

# Conversation memory.
MEMORY_WINDOW_TURNS = _env_int("MEMORY_WINDOW_TURNS", 4)
MEMORY_SUMMARY_MAX_CHARS = _env_int("MEMORY_SUMMARY_MAX_CHARS", 15000)
MEMORY_RECENT_MAX_CHARS = _env_int("MEMORY_RECENT_MAX_CHARS", 8000)

# Learning/debug trace.
LEARNING_TRACE_ENABLED = _env_bool("LEARNING_TRACE_ENABLED", True)
LEARNING_TRACE_MAX_TEXT_CHARS = _env_int("LEARNING_TRACE_MAX_TEXT_CHARS", 1200)

# Aliyun OSS for chat image attachments
OSS_ACCESS_KEY_ID = os.getenv("oss_access_key_id") or os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("oss_access_key_secret") or os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET = os.getenv("oss_bucket") or os.getenv("OSS_BUCKET", "")
OSS_ENDPOINT = os.getenv("oss_endpoint") or os.getenv("OSS_ENDPOINT", "")

# 聊天附件回收（issue #142）。上传即登记一条「待确认」行，发送成功时与消息行**同一次提交**
# 被消费；超过这个时长仍未被消费的登记行，由清扫任务实际删除对应对象。
# 取 24 小时是为了让「挑好图先放在输入框里、过一会儿再发」远在窗口之内：清扫一旦提前动手，
# 删掉的就是一条马上要发出去的消息所引用的对象，而对象存储没有回收站。
CHAT_ATTACHMENT_PENDING_TTL_SECONDS = _env_int("CHAT_ATTACHMENT_PENDING_TTL_SECONDS", 24 * 60 * 60)
# 单次清扫最多处理多少个对象。DELETE 是串行外呼、每个最坏要等到连接超时，不设上限时
# 一次积压就能让清扫一直占着线程（清扫挂在后台线程上，但也该有明确的工作量上界）。
CHAT_ATTACHMENT_SWEEP_BATCH_LIMIT = _env_int("CHAT_ATTACHMENT_SWEEP_BATCH_LIMIT", 200)
# 两轮清扫之间的间隔。只在启动时扫一次是不够的：一个跑几个月不重启的进程永远等不到第二轮，
# 孤儿会一直攒着，那正是 issue #142 要消灭的形态。默认 6 小时——回收是尽力而为的维护动作，
# 迟一点没有正确性代价，而每轮都要串行外呼 OSS。
CHAT_ATTACHMENT_SWEEP_INTERVAL_SECONDS = _env_int("CHAT_ATTACHMENT_SWEEP_INTERVAL_SECONDS", 6 * 60 * 60)
# 一次删除外呼的租约时长（issue #238）。清扫领走一行时写下租约起点，这些秒数只用来判断
# 「上一个清扫进程是不是已经死在半路上」：超过它，这一行可以被下一轮清扫**重新领取**并重试
# 删除。它**不是**发送侧的放行条——发送能不能通行只看墓碑（reclaimed_at），与租约是否过期
# 无关；把租约过期当成「清扫放弃了、可以发送了」正是这次修复要消灭的误判。
# 默认 300 秒：够一次 OSS DELETE 往返（连接超时量级），又远小于清扫间隔（6 小时），
# 于是一轮清扫里几乎不会出现「刚领的行还没做完就被自己重领」。
CHAT_ATTACHMENT_LEASE_SECONDS = _env_int("CHAT_ATTACHMENT_LEASE_SECONDS", 300)
# 墓碑保留期（issue #238）。删除成功后，登记行作为「这把键已被回收」的事实留下，好让这把
# 键的发送被立刻拒绝（否则消息会落库并永久引用一个已被删掉的对象）。保留期一过，行没有
# 别的用处了，由清扫末尾的 GC 连行一起清掉——清掉之后这把键退回「表里查无此行」的语义，
# 发送不再被拒（那正是 #142 之前就存在的老键形态，按 deny-by-default 无处可拒）。
# 默认 7 天：比 24 小时的待回收窗口长得多，给足「清扫已判死、用户又想把图发出去」的纠错面。
# **置 0 = 关闭 GC**：墓碑永不清理（用于需要长期保留删除记录的场景），代价是这张表只增不减。
CHAT_ATTACHMENT_TOMBSTONE_TTL_SECONDS = _env_int("CHAT_ATTACHMENT_TOMBSTONE_TTL_SECONDS", 7 * 24 * 60 * 60)

# 请求限流（issue #183）。计数器在进程内存里（见 service/rate_limit.py），多 worker / 多实例
# 部署时每个进程各持一份，真实上限 = 配置值 × 进程数。登录侧按「账号 + 来源地址」计数：
# 阈值内的正确口令不受影响，超阈值后连正确口令一起拒到窗口滑出（否则每次拒绝都要先跑一遍
# bcrypt，攻击者照样能烧 CPU），窗口滑出即自动恢复，不做永久锁定。
LOGIN_RATE_LIMIT_MAX_FAILURES = _env_int("LOGIN_RATE_LIMIT_MAX_FAILURES", 5)
LOGIN_RATE_LIMIT_WINDOW_SECONDS = _env_int("LOGIN_RATE_LIMIT_WINDOW_SECONDS", 300)
# 同一进程内同时在跑的 SSE 聊天流上限；超出的请求立刻 429，不排队等上游超时。
CHAT_STREAM_MAX_CONCURRENCY = _env_int("CHAT_STREAM_MAX_CONCURRENCY", 8)

# JWT
# The literal below is a public placeholder shipped with the repository; anyone can
# read it, so it must never be used as an actual HS256 signing key.
DEFAULT_SECRET_KEY = "change-this-secret-key-in-production"
ALLOW_INSECURE_DEFAULT_SECRET = _env_bool("ALLOW_INSECURE_DEFAULT_SECRET", False)

SECRET_KEY_SOURCE_ENV = "env"  # scan-secrets:allow source label, not a credential
SECRET_KEY_SOURCE_INSECURE_DEV = "insecure-dev"  # scan-secrets:allow source label, not a credential
SECRET_KEY_SOURCE_UNCONFIGURED = "unconfigured"  # scan-secrets:allow source label, not a credential

SECRET_KEY_ERROR = (
    "SECRET_KEY 未配置，或仍是仓库内置的公开占位值，服务拒绝启动。\n"
    "请在环境变量或 .env 中设置一个足够长的随机值，例如：\n"
    "  python -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
    "仅本地开发可临时设置 ALLOW_INSECURE_DEFAULT_SECRET=true 放行：该开关让进程使用一次性\n"
    "随机密钥启动，每次重启都会使已签发的 token 全部失效，禁止用于任何对外可访问的部署。"
)


class SecretKeyError(RuntimeError):
    """Raised at startup when SECRET_KEY is missing or still the public placeholder."""


def resolve_secret_key(secret_key, allow_insecure_default):
    """Resolve the JWT signing key into ``(key, source)``.

    The repository placeholder is never returned as a usable key. When the value is
    missing or still the placeholder, an ephemeral random key is generated instead, so
    the process can never sign or verify with a publicly known secret -- not even when
    it is imported by a script or test that skips the startup check.
    """
    candidate = (secret_key or "").strip()
    if candidate and candidate != DEFAULT_SECRET_KEY:
        return candidate, SECRET_KEY_SOURCE_ENV
    if allow_insecure_default:
        _logger.warning(
            "SECRET_KEY is not configured; ALLOW_INSECURE_DEFAULT_SECRET=true so this "
            "process runs on an ephemeral random key. Never use it for a deployment "
            "reachable by others, and expect every restart to invalidate issued tokens."
        )
        return secrets.token_urlsafe(48), SECRET_KEY_SOURCE_INSECURE_DEV
    _logger.warning(
        "SECRET_KEY is not configured or is still the public repository placeholder. "
        "The process will refuse to start; set SECRET_KEY to a long random value."
    )
    return secrets.token_urlsafe(48), SECRET_KEY_SOURCE_UNCONFIGURED


def secret_key_is_usable() -> bool:
    """True when the process has a non-placeholder key to sign and verify with.

    That is either an explicitly configured SECRET_KEY, or the local-dev escape hatch
    (ephemeral random key). Only the unconfigured case is unusable.
    """
    return SECRET_KEY_SOURCE != SECRET_KEY_SOURCE_UNCONFIGURED


def ensure_secret_key_configured() -> None:
    """Fail fast when the process would run on an unpredictable-but-unusable config.

    Called from the application startup path: a deployment that forgot to set
    SECRET_KEY must not silently serve traffic.
    """
    if not secret_key_is_usable():
        raise SecretKeyError(SECRET_KEY_ERROR)


SECRET_KEY, SECRET_KEY_SOURCE = resolve_secret_key(
    os.getenv("SECRET_KEY"), ALLOW_INSECURE_DEFAULT_SECRET
)
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24)
