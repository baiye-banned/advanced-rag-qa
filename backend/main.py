import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from config import (
    REBUILD_KNOWLEDGE_INDEX_ON_STARTUP,
    ensure_mysql_password_configured,
    ensure_secret_key_configured,
)
from database.session import init_db
from rag.milvus_client import prewarm_token_encoding
from rag.retrieval import schedule_bm25_prewarm
from router.auth import router as auth_router
from router.chat import router as chat_router
from router.knowledge import router as knowledge_router
from router.uploads import router as uploads_router
from router.user import router as user_router
from service.chat_service import schedule_orphan_attachment_sweep
from service.knowledge_service import rebuild_existing_knowledge_index, run_ingest_step
from service.user_service import seed_default_users


logger = logging.getLogger(__name__)

INTEGRITY_CONFLICT_MESSAGE = "数据冲突：本次操作与当前数据状态不一致，请重试。"

# 分词表预热最多等这么久：离线环境里 tiktoken 的下载连接可能长时间挂起，
# 预热本身是锦上添花，不能把服务就绪时间绑在它的网络超时上。
PREWARM_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_secret_key_configured()
    ensure_mysql_password_configured()
    # 分词表预热是非致命的：tiktoken 首次使用会下载 cl100k_base 的 BPE 文件，干净或
    # 受限网络部署会在用户第一次上传时才暴露失败。这里提前检查并把实际缓存目录写进
    # 日志；失败只记 WARNING，不阻止启动（见 rag.milvus_client.prewarm_token_encoding）。
    # 放线程里执行：离线环境下载失败前的连接等待不应卡住启动时的其它协程。
    # 再加一道显式超时：线程里的阻塞不会因为 wait_for 取消而结束，但启动流程不必
    # 等它——超时只记 WARNING，后续初始化照常进行，首次上传时再暴露/重试即可。
    try:
        await asyncio.wait_for(
            asyncio.to_thread(prewarm_token_encoding),
            timeout=PREWARM_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "Embedding tokenizer prewarm timed out after %.0fs; continuing startup. "
            "The first ingest may block on tokenizer initialization.",
            PREWARM_TIMEOUT_SECONDS,
        )
    init_db()
    seed_default_users()
    if REBUILD_KNOWLEDGE_INDEX_ON_STARTUP:
        # 重建对每个历史文件同步向量化并写库，跑在**入库专用**线程池上；启动顺序不变
        # （重建完成才接流量），但重建期间事件循环仍可调度其它协程。不设总时限：
        # 重建按文件数逐个跑，没有「一份文档」的预算语义，逐文件的失败隔离已由
        # rebuild_existing_knowledge_index 自己保证。
        await run_ingest_step(rebuild_existing_knowledge_index)
    # BM25 语料预热（issue #288 返工轮 medium-1）：整库扫描放在后台守护线程上，启动不等它，
    # 第一问也不必在请求路径里首次扫描。上传/删除路径也会按知识库单独排一次。
    schedule_bm25_prewarm()
    # 回收「上传了但从未被发送」的聊天附件（issue #142）：上传即登记一条待确认行，发送成功
    # 时被消费，超过保留窗口还没被消费的由清扫删掉对应对象。挂后台守护线程而不是 await：
    # DELETE 是串行外呼，积压一批能把就绪时间拖成分钟级；清扫只碰超过保留窗口的登记行，
    # 与启动时正在上传的对象没有交集，晚一点跑没有正确性代价。
    #
    # 启动只是第一轮：线程里是个循环，之后按间隔继续扫。只扫启动那一轮的话，跑几个月不重启
    # 的进程永远等不到第二轮，孤儿会一直攒着——issue #142 的形态会以「服务没重启」为由复活。
    schedule_orphan_attachment_sweep()
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)
# CORS 收口（issue #235）：`allow_origins=["*"]` 必须与「不允许凭据」绑定，不能搭配
# `allow_credentials=True`。后者一旦为真，`*` 与 `Access-Control-Allow-Credentials` 在浏览器里
# 无法共存，Starlette 会改为**回显请求里的任意 Origin** 并附上允许凭据的头——等于给任意站点
# 发放「可携带凭据跨域读写」的许可。
#
# 而本服务今天没有任何凭据可被这样带走：身份是 localStorage 里的 Bearer token，由请求拦截器
# 手写成 Authorization 头（`src/api/request.js:11-17`），全仓没有 cookie 会话、也没有
# withCredentials；生产（`docs/DEPLOYMENT.md` 的 nginx 同 host 托管 dist 与 `/api/`）与开发
# （`vite.config.js` 的 server.proxy）都是同源部署，唯一的跨源消费者是 CI e2e 套件，它同样走
# Authorization 头——而 Authorization 不受 Allow-Credentials 管辖，靠的是下面的 allow_headers。
# 所以 `allow_credentials=True` 在这里收益为零，只是凭空多出一条「任意源 + 允许凭据」的配置。
#
# 升级触发条件：将来若引入 cookie 会话，**必须同时改为显式来源白名单**，而不是把
# allow_credentials 改回 True。那时凭据真实存在，`*` 与它共存就从潜伏项变成可利用的漏洞。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# `/uploads` 曾经是一整棵目录的匿名静态挂载（issue #186）。目录下只有 `avatars/`，所以这里换成
# 一条带鉴权的路由：语义上仍然是「按 URL 取头像」，但请求方必须是这条头像的主人。
# 挂载与路由不能并存——Mount 会先匹配掉整个前缀，后面那条路由永远轮不到，匿名读就又回来了。
app.include_router(uploads_router)


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    """写路径未单独处理的唯一/外键冲突兜底成 409，不再冒成 500（issue #60 验收标准第 4 条）。

    用户可见的知识库重名竞态已在 service 层翻译成带具体文案的 400，这里只兜住漏网者
    （例如并发删除后仍被引用的写入），保证响应是可读的 4xx 而不是 Starlette 默认的 500。
    """
    logger.warning("Unhandled integrity error: %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=409, content={"detail": INTEGRITY_CONFLICT_MESSAGE})


@app.get("/")
def root():
    return {"status": "ok", "service": "RAG API"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(user_router)
app.include_router(chat_router)
app.include_router(knowledge_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        reload_excludes=[
            "uploads/*",
            "uploads/**",
            "backend/uploads/*",
            "backend/uploads/**",
            "*.db",
            "backend/*.db",
            "*.sqlite",
            "*.sqlite3",
            "backend/*.sqlite",
            "backend/*.sqlite3",
        ],
    )
