
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated, Any, Callable

from fastapi import Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from config import KNOWLEDGE_INDEX_MAX_WORKERS, KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS
from rag.milvus_client import EmbeddingBackendError, add_chunks, delete_file_chunks
from rag.retrieval import schedule_bm25_prewarm
from crud import knowledge_base as crud_knowledge_base
from crud import knowledge_file as crud_knowledge_file
from crud.pagination import LIST_DEFAULT_LIMIT, LIST_MAX_LIMIT
from database.session import SessionLocal, get_db
from model.models import KnowledgeFile, User
from schema.schemas import KnowledgeBaseRequest
from service.auth_service import get_current_user
from service.pagination_service import resolve_list_limit
from service.utils_service import (
    KNOWLEDGE_UPLOAD_MAX_BYTES,
    KNOWLEDGE_UPLOAD_TYPE_ERROR_MESSAGE,
    knowledge_upload_too_large_message,
    resolve_knowledge_upload_type,
)


logger = logging.getLogger(__name__)

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
# multipart 的 content-length 含边界和表单字段等开销，预检留出余量，避免误伤刚好达标的文件。
MULTIPART_OVERHEAD_ALLOWANCE_BYTES = 4096

# 入库专用线程池。解析、分块、向量化外呼与 Milvus 写入都是同步实现，之前两类走
# asyncio.to_thread —— 那是 asyncio 的**默认** executor，检索侧一轮多路召回也用同一个，
# 于是并发上传占满默认池时整轮检索排在入库后面（issue #83 第 4 项）。
_INDEX_EXECUTOR = ThreadPoolExecutor(
    max_workers=KNOWLEDGE_INDEX_MAX_WORKERS,
    thread_name_prefix="knowledge-index",
)

# 清理专用线程池。超时/中止后的「删向量 + 删元数据行」都是同步阻塞外呼：
# ① 不能落回事件循环线程（返工轮 major-2：total budget 耗尽时 work.cancel() 让 future
#    立刻 done，add_done_callback 于是在事件循环线程上就地执行整段清理，循环停摆 300ms）；
# ② 也不该排进上面的入库池——入库池只有 KNOWLEDGE_INDEX_MAX_WORKERS 个槽位，
#    被卡死的写入线程占满时清理会无限期排队（返工轮 minor-1，实测 1.5s 未完成）。
_CLEANUP_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="knowledge-cleanup")

INGEST_TIMEOUT_MESSAGE = "知识文件上传失败：入库超过总时限，请稍后重试。"


class KnowledgeIngestTimeout(Exception):
    """整份文档入库超过总时限。

    携带 future 是给清理用的：超时只是放弃**等待**，工作线程还在跑，
    清理必须排在它之后（见 defer_ingest_cleanup）。
    """

    def __init__(self, message: str, future=None):
        super().__init__(message)
        self.future = future


def ingest_deadline() -> float:
    """一次上传的入库截止时刻（time.monotonic 时钟）。整份文档只算一次，各步骤共享。

    用 time.monotonic 而不是事件循环时钟：入库步骤跑在 ThreadPoolExecutor 的工作线程里，
    该线程需要拿同一个截止时刻去限制 Retry-After 退避（见 milvus_client._embed_documents）。
    """
    return time.monotonic() + KNOWLEDGE_INDEX_TOTAL_TIMEOUT_SECONDS


def _submit_on(executor: ThreadPoolExecutor, step: Callable[..., Any], args: tuple) -> tuple:
    """把一步同步工作提交到指定线程池，返回 (底层 concurrent future, asyncio future)。

    自己 submit + wrap_future，而不是用 loop.run_in_executor：后者只返回 asyncio
    那一层，而 asyncio.wait_for 超时时会立刻取消并**完成**它——哪怕工作线程还在跑。
    要判断「工作线程真的跑完了」，必须持有底层 concurrent future，
    超时清理正是挂在它上面（见 defer_ingest_cleanup）。
    """
    work = executor.submit(partial(step, *args))
    return work, asyncio.wrap_future(work)


def submit_ingest_work(step: Callable[..., Any], args: tuple) -> tuple:
    """把一步入库工作提交到入库池，返回 (底层 concurrent future, asyncio future)。"""
    return _submit_on(_INDEX_EXECUTOR, step, args)


async def _run_step_on(
    executor: ThreadPoolExecutor,
    step: Callable[..., Any],
    args: tuple,
    deadline: float | None,
) -> Any:
    """在线程池上执行一步同步工作，deadline 非 None 时给整份文档的总预算。

    入库与清理共用这一份超时语义：两条路径各写一遍，迟早会漂移成两种行为。
    """
    work, future = _submit_on(executor, step, args)
    if deadline is None:
        return await future

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        # 还没轮到执行：可以真的取消掉这个工作项（cancel 只对未开始的任务生效）。
        work.cancel()
        raise KnowledgeIngestTimeout(INGEST_TIMEOUT_MESSAGE, work)
    try:
        return await asyncio.wait_for(future, timeout=remaining)
    except (asyncio.TimeoutError, TimeoutError):
        # 两个都写：3.11 起 asyncio.TimeoutError 就是内建 TimeoutError 的别名，
        # 而 3.10 上它是另一个类（继承自 Exception），只写内建那个接不住——
        # 超时会落到下面的通用失败分支：既不报「超过总时限」，也走不到排在写入线程
        # 之后的清理，等于把这一项的两个修复一起绕过去。CI 的 Python 3.10 实测过。
        raise KnowledgeIngestTimeout(INGEST_TIMEOUT_MESSAGE, work)


async def run_ingest_step(step: Callable[..., Any], /, *args: Any, deadline: float | None = None) -> Any:
    """在入库专用线程池上执行一步同步入库工作。

    与 asyncio.to_thread 的两点差别都是 issue #83 要求的：

    ① 走 _INDEX_EXECUTOR 而不是默认池：入库与检索不再互相排队。
    ② deadline 给的是整份文档的截止时刻。EMBEDDING_INGEST_TIMEOUT_SECONDS 只约束
       单批外呼，一份文档要发多批，没有总预算时一次上传可以无限期占住一个槽位。
       调用方在入库开始时算一次 deadline，解析/分块/向量化共享它。

    deadline=None 表示不设总时限（启动重建按文件数逐个跑，没有「一份文档」的语义）。

    超时只放弃等待：asyncio 取消不了已经在跑的线程，工作项会继续跑完。
    因此超时抛出的 KnowledgeIngestTimeout 会带上那个 future，
    调用方必须把清理排在它之后，否则删掉的向量会被还在跑的批次重新写回来。
    """
    return await _run_step_on(_INDEX_EXECUTOR, step, args, deadline)


async def run_cleanup_step(step: Callable[..., Any], /, *args: Any, deadline: float | None = None) -> Any:
    """在清理专用线程池上执行一步同步清理工作（删向量）。

    返工轮 minor-1：失败分支原本走 run_ingest_step，即排进只有
    KNOWLEDGE_INDEX_MAX_WORKERS 个槽位的入库池——被卡死的写入线程占满时这一步会
    无限期排队（对抗评审实测 1.5s 未完成），而且 deadline=None 时 `await future`
    本身没有上限，等于把已经失败的请求再挂死一次。这里换成独立池 + 可选总时限。

    deadline 由调用方给（上传链路用 ingest_deadline()，与入库共用同一把尺子）。
    超时同样只放弃等待：工作项已经提交、会在后台跑完，留痕交由调用方。
    """
    return await _run_step_on(_CLEANUP_EXECUTOR, step, args, deadline)


def defer_ingest_cleanup(work, entry_id: int, user_id: int) -> None:
    """把「删向量 + 删元数据行」挂到入库**工作线程真正结束**之后执行。

    work 必须是底层 concurrent future（submit_ingest_work 的第一个返回值）：
    asyncio 那一层在 wait_for 超时时就已经完成，挂在它上面等于没有排序。

    超时后写入线程不会被取消，仍在往向量库写。若像其它失败路径那样立刻清理，
    清理先删、写回后到，就留下了一批指向已删文件的向量：列表里没有它，
    检索却还能命中，答案里会出现点开就 404 的来源（issue #83 第 4 项对抗评审实测）。
    早先没有总时限时不会出现这条路径（请求会一直等到线程结束才清理），
    所以这里必须自己保证顺序。

    用独立 SessionLocal 而不是请求级 Session：清理跑在线程池里，
    而且请求早已返回，请求级 Session 可能已经关闭。

    这条路径的代价，写在这里备查：真正卡死（永不返回）的写入线程会让清理永不执行，
    于是元数据行留在库里——文件出现在列表中、检索命中的是残缺向量，但用户已收到 500。
    这是有意选的失败态：反过来先删行再等线程，留下的是「列表里没有、检索却命中」
    的孤儿向量，那条路径不会自愈（行没了，启动重建也不会再碰它），
    而留下行的这条会被 rebuild_existing_knowledge_index 在下次启动时重建索引。
    """
    def _cleanup(_finished_future) -> None:
        # 回调体只做「派活」这一件非阻塞的事，真正的阻塞外呼全部交给清理专用池。
        #
        # 返工轮 major-2：future 已经完成时，concurrent.futures 是在**调用线程**上
        # 就地执行回调的，而这里的调用线程常常就是事件循环线程——总预算在工作项还
        # 排队（PENDING）时耗尽，work.cancel() 成功、future 立刻 done，
        # upload_knowledge 紧接着调本函数，整段 delete_file_chunks 外呼 + 一次删行
        # 就回到了事件循环上（对抗评审实测循环停摆 300.5ms；asyncio.wait_for 那条
        # 超时分支把取消传导到底层 future，同样使其 done，形态一致）。
        # 回调在线程池工作线程里被调用时也不能就地做阻塞活：那会占住入库槽位。
        try:
            _CLEANUP_EXECUTOR.submit(_delete_file_vectors_and_row, entry_id, user_id)
        except RuntimeError as exc:
            # 解释器退出、线程池已关时 submit 会抛 RuntimeError：留痕即可，
            # 让异常从 done_callback 里冒出去只会污染线程池的工作线程。
            logger.warning("Failed to schedule ingest cleanup: file_id=%s error=%s", entry_id, exc, exc_info=True)

    # 线程池的 done_callback 由**执行该工作项的线程**在返回后调用，天然落在写入之后；
    # 派活也因此在写入结束之后才发生，清理依旧排在写入线程之后。
    work.add_done_callback(_cleanup)


def _keep_knowledge_file_row(entry_id: int, *, reason: str) -> None:
    """向量没删干净时保留元数据行，等下次启动重建把它修好。

    这是本文件统一的失败态取舍（见 defer_ingest_cleanup）：删向量失败还照样删行，
    留下的是「列表里没有、检索却命中」的孤儿向量，而且**不会自愈**——行没了，
    rebuild_existing_knowledge_index 只遍历数据库里的行，再也不会碰到这批向量，
    占用的向量库存储也没有任何流程会回收。行留着，下次启动重建就会用 add_chunks
    （整体替换语义）盖掉它残留的向量。
    """
    logger.warning("Keeping knowledge file row for the startup rebuild: file_id=%s reason=%s", entry_id, reason)


def _delete_file_vectors_and_row(entry_id: int, user_id: int) -> None:
    """删向量 + 删元数据行。只在清理专用池的线程里跑，不在事件循环线程上跑。

    只有向量**确实删干净**才允许删元数据行——与单文件删除路径（delete_knowledge
    先删向量、失败即抛 500 并保留行）同一条判据：删向量失败还照样删行，就会留下
    上面 _keep_knowledge_file_row 记的那种不自愈的孤儿向量。
    """
    db = SessionLocal()
    try:
        try:
            delete_file_chunks(entry_id)
        except Exception as exc:
            logger.warning("Failed to clean up timed-out ingest vectors: file_id=%s error=%s", entry_id, exc, exc_info=True)
            _keep_knowledge_file_row(entry_id, reason="vector delete failed")
            # issue #287：向量没删干净时文件行仍留着等启动重建，但本次切片的结构元数据
            # 不该留下——它描述的向量已经不完整，清掉后按当前文本路径降级即可。
            _delete_chunk_metadata_best_effort(db, entry_id)
            return
        # 复用既有 CRUD：归属过滤与「行可能已被并发删掉」都走它的语义；
        # delete_knowledge_file 同一次提交里也会清掉切片元数据。
        crud_knowledge_file.delete_knowledge_file(db, entry_id, user_id)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning("Failed to remove knowledge file after ingest timeout: file_id=%s error=%s", entry_id, exc, exc_info=True)
    finally:
        db.close()


def _delete_chunk_metadata_best_effort(db: Session, file_id: int) -> None:
    """失败清理路径上的元数据删除：失败只留痕，不遮蔽原始错误。

    读的是任意 ``Session`` 替身/半关闭会话时，失败可能不是 ``SQLAlchemyError``
    （例如替身没有实现 ``delete``）。清理路径本身不允许再把异常抛回去。
    先 rollback：上一步（replace/add_chunks）失败后会话可能处于待回滚状态，
    直接 delete 会以 PendingRollbackError 失败，旧行反而清不掉。
    """
    try:
        db.rollback()
    except Exception:
        pass
    try:
        crud_knowledge_file.delete_chunk_metadata(db, file_id)
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "Failed to remove chunk metadata: file_id=%s error=%s", file_id, exc, exc_info=True
        )


def upload_body_exceeds_limit(content_length: str | None, max_bytes: int) -> bool:
    """content-length 预检：只在声明长度明显超过上限时才判定超限。"""
    try:
        declared_length = int(content_length)
    except (TypeError, ValueError):
        return False
    return declared_length > max_bytes + MULTIPART_OVERHEAD_ALLOWANCE_BYTES


async def read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    """分块读取并累计计量，超限立刻中断，避免把超大文件整体读进内存。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(400, knowledge_upload_too_large_message())
        chunks.append(chunk)
    return b"".join(chunks)


def get_default_knowledge_base(db: Session, user_id: int):
    knowledge_base = crud_knowledge_base.get_default_knowledge_base(db, user_id)
    if knowledge_base:
        return knowledge_base
    try:
        return crud_knowledge_base.create_knowledge_base(db, "默认知识库", user_id)
    except IntegrityError:
        # 并发首次访问时可能同时创建默认知识库，唯一约束冲突后回读即可。
        db.rollback()
        knowledge_base = crud_knowledge_base.get_default_knowledge_base(db, user_id)
        if knowledge_base:
            return knowledge_base
        raise


def resolve_knowledge_base(db: Session, knowledge_base_id: int | None, user_id: int):
    knowledge_base = crud_knowledge_base.resolve_knowledge_base(db, knowledge_base_id, user_id)
    if knowledge_base:
        return knowledge_base
    if knowledge_base_id:
        # 传了 id 却查不到（不存在或不属于当前用户）统一按 404 处理，不暴露资源存在性。
        raise HTTPException(404, "知识库不存在")
    return get_default_knowledge_base(db, user_id)


def rebuild_existing_knowledge_index():
    db = SessionLocal()
    try:
        for entry in db.query(KnowledgeFile).filter(KnowledgeFile.knowledge_base_id.isnot(None)).all():
            chunks = crud_knowledge_file.chunk_text(entry.content or "", entry.id)
            _warn_on_low_chunk_coverage(entry, entry.content or "", chunks, scope="Knowledge index rebuild")
            try:
                add_chunks(chunks, entry.id, entry.name, entry.knowledge_base_id)
            except Exception as exc:
                # 单个文件失败不阻塞启动：旧索引保持原样，可修复向量化服务后再次重建。
                logger.warning("Knowledge index rebuild failed: file_id=%s error=%s", entry.id, exc, exc_info=True)
                # issue #287：重建失败时 #289 的检索侧不能读到与旧向量/本次分块不一致的
                # 结构元数据；清掉比留着陈旧值更安全（缺失时按文本路径降级）。
                _delete_chunk_metadata_best_effort(db, entry.id)
                continue
            try:
                # 重建成功（整体替换语义的 add_chunks）后同步刷新切片元数据，
                # 让 #289 的章节/条款/表格信息跟当前切片文本一致。
                crud_knowledge_file.replace_chunk_metadata(
                    db, entry.id, entry.knowledge_base_id, chunks
                )
            except Exception as exc:
                # 元数据刷新失败不推翻已经完成的向量重建；清掉旧行避免读到更早的切片。
                logger.warning(
                    "Knowledge chunk metadata rebuild failed: file_id=%s error=%s",
                    entry.id,
                    exc,
                    exc_info=True,
                )
                _delete_chunk_metadata_best_effort(db, entry.id)
        # 启动重建完成后把 BM25 语料也预热一遍；重建路径本来就要遍历整库，这里不该让
        # 第一问再独自扫描一次。
        schedule_bm25_prewarm()
    finally:
        db.close()


# 知识库重名唯一键在各后端/各版本的报错特征。模型侧的名字是 uq_knowledge_bases_user_name；
# SQLite 没有具名唯一键，只能从「哪几列冲突」反推。
# 后三条覆盖 9b34e8a 之前的历史 schema（name 单列唯一）：那个索引在 MySQL 上由
# _ensure_knowledge_base_owner_unique_index 换掉、且它的失败被吞掉，所以换库失败时
# 仍可能带着旧索引跑；不认这三条的话，重名会被误判成「非重名冲突」而回 409，
# 前端拿不到「知识库已存在」的提示。
_KNOWLEDGE_BASE_NAME_UNIQUE_MARKERS = (
    "uq_knowledge_bases_user_name",
    "knowledge_bases.user_id, knowledge_bases.name",
    "UNIQUE constraint failed: knowledge_bases.name",
    "for key 'knowledge_bases.name'",
    "for key 'name'",
)


def _is_knowledge_base_name_conflict(exc: IntegrityError) -> bool:
    """这个 IntegrityError 是不是「同一用户已有同名知识库」这条唯一键引起的。

    issue #83 第 6 项：`knowledge_bases` 上不止这一条约束（user_id 外键同样会抛
    IntegrityError）。不区分来源就一律报「知识库名称已存在」，用户拿到的是与真实
    原因无关的诊断——提示改名会继续失败——同时让 main.py 新增的全局 409 兜底在这两条
    写路径上永远不可达。
    """
    detail = str(getattr(exc, "orig", None) or exc)
    return any(marker in detail for marker in _KNOWLEDGE_BASE_NAME_UNIQUE_MARKERS)


def _index_failure_detail(exc: Exception) -> str:
    """向量化失败时把可读原因透出给前端，其余写入异常保持原有提示。"""
    if isinstance(exc, EmbeddingBackendError):
        return f"知识文件上传失败：{exc}"
    if isinstance(exc, KnowledgeIngestTimeout):
        return INGEST_TIMEOUT_MESSAGE
    return "知识文件上传失败，向量库写入异常。"


def _warn_on_low_chunk_coverage(entry, text: str, chunks: list[dict], *, scope: str) -> None:
    """分块覆盖率异常时留痕：入库文本可能远少于原文，不允许静默成功。"""
    coverage = crud_knowledge_file.chunk_coverage_ratio(text, chunks)
    if coverage >= crud_knowledge_file.KNOWLEDGE_INDEX_MIN_COVERAGE_RATIO:
        return
    logger.warning(
        "%s chunk coverage too low: file_id=%s filename=%s chunks=%s covered_chars=%s source_chars=%s coverage=%.1f%%",
        scope,
        entry.id,
        entry.name,
        len(chunks),
        sum(len(chunk.get("text") or "") for chunk in chunks),
        len(text or ""),
        coverage * 100,
    )


def _delete_file_vectors_or_500(file_id: int, *, scope: str, detail: str) -> None:
    try:
        delete_file_chunks(file_id)
    except Exception as exc:
        logger.warning("%s vector cleanup failed: file_id=%s error=%s", scope, file_id, exc, exc_info=True)
        raise HTTPException(500, detail)


def list_knowledge_bases(
    limit: Annotated[int, Query(ge=1, le=LIST_MAX_LIMIT)] = LIST_DEFAULT_LIMIT,
    after_id: Annotated[int | None, Query(ge=1)] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """知识库列表，按创建顺序（旧 -> 新）分页返回。

    游标 after_id 取上一次响应里最后一条的 id，返回它之后的那一页（键集游标，
    不是 offset）；不传时返回第一页，页大小默认 LIST_DEFAULT_LIMIT。
    """
    rows = crud_knowledge_base.list_knowledge_bases(
        db,
        user.id,
        limit=resolve_list_limit(limit),
        after_id=after_id,
    )
    # 文件数用一条 GROUP BY 聚合取回，避免逐库懒加载（那条语句会把 LONGTEXT content 整列读出来）。
    # 上限落地后这里的 IN 列表长度也跟着有了界：页大小多大，这条聚合的参数列表就多长，
    # 不再随知识库总数增长（issue #191）。
    file_counts = crud_knowledge_base.count_knowledge_files_by_base(db, [item.id for item in rows])
    return [
        crud_knowledge_base.serialize_knowledge_base(item, file_counts.get(item.id, 0))
        for item in rows
    ]


def create_knowledge_base(body: KnowledgeBaseRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "知识库名称不能为空")
    if crud_knowledge_base.knowledge_base_name_exists(db, name, user.id):
        raise HTTPException(400, "知识库名称已存在")
    try:
        entry = crud_knowledge_base.create_knowledge_base(db, name, user.id)
    except IntegrityError as exc:
        # 先回滚失败事务，避免该 Session 残留失败事务状态。
        db.rollback()
        if not _is_knowledge_base_name_conflict(exc):
            # 非重名冲突（例如请求在途时该 user 行被并发删除，user_id 外键失效）：
            # 不能谎称重名——提示改名会一直失败。重新抛出交给全局兜底回 409。
            logger.warning(
                "Knowledge base create failed on a non-name constraint: name=%s user_id=%s",
                name, user.id, exc_info=True,
            )
            raise
        # 预检查与写入之间被并发请求抢先提交了同名知识库，唯一约束兜底：
        # 翻译成与串行一致的 400。
        logger.warning("Knowledge base create conflict: name=%s user_id=%s", name, user.id, exc_info=True)
        raise HTTPException(400, "知识库名称已存在")
    # 刚落库的知识库名下不可能已有文件，直接给 0，省一次计数查询。
    return crud_knowledge_base.serialize_knowledge_base(entry, 0)


def rename_knowledge_base(kid: int, body: KnowledgeBaseRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entry = crud_knowledge_base.get_knowledge_base(db, kid, user.id)
    if not entry:
        raise HTTPException(404, "知识库不存在")
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "知识库名称不能为空")
    if crud_knowledge_base.knowledge_base_name_exists(db, name, user.id, exclude_id=kid):
        raise HTTPException(400, "知识库名称已存在")
    try:
        entry = crud_knowledge_base.rename_knowledge_base(db, kid, name, user.id)
    except IntegrityError as exc:
        db.rollback()
        if not _is_knowledge_base_name_conflict(exc):
            # 与 create 同源：只把重名唯一键翻译成 400，其余交给全局 409 兜底。
            logger.warning(
                "Knowledge base rename failed on a non-name constraint: kid=%s name=%s user_id=%s",
                kid, name, user.id, exc_info=True,
            )
            raise
        # 两个知识库同时被改成同一个名字时同样只有一方能提交成功，后到者按同名处理。
        logger.warning("Knowledge base rename conflict: kid=%s name=%s user_id=%s", kid, name, user.id, exc_info=True)
        raise HTTPException(400, "知识库名称已存在")
    return crud_knowledge_base.serialize_knowledge_base(entry, crud_knowledge_base.count_knowledge_files(db, kid))


def delete_knowledge_base(kid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entry = crud_knowledge_base.get_knowledge_base(db, kid, user.id)
    if not entry:
        raise HTTPException(404, "知识库不存在")
    if crud_knowledge_base.count_knowledge_bases(db, user.id) <= 1:
        raise HTTPException(400, "至少保留一个知识库")

    target = crud_knowledge_base.get_fallback_knowledge_base(db, kid, user.id)
    file_ids = [file_entry.id for file_entry in crud_knowledge_base.list_files_for_knowledge_base(db, kid)]
    for file_id in file_ids:
        _delete_file_vectors_or_500(
            file_id,
            scope="Knowledge base",
            detail="知识库删除失败，向量库清理异常，请稍后重试。",
        )
    crud_knowledge_base.delete_knowledge_base_with_files(db, kid, target.id, user.id)
    return {"message": "ok", "fallback_knowledge_base_id": target.id}


def list_knowledge(
    knowledge_base_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=LIST_MAX_LIMIT)] = LIST_DEFAULT_LIMIT,
    before_id: Annotated[int | None, Query(ge=1)] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """知识库文件列表，按上传时间倒序（新 -> 旧）分页返回。

    游标 before_id 取上一次响应里最后一条的 id，返回它之前（更早）的那一页，与消息接口
    同族；不传时返回最新一页，页大小默认 LIST_DEFAULT_LIMIT。
    """
    knowledge_base = resolve_knowledge_base(db, knowledge_base_id, user.id)
    files = crud_knowledge_file.list_knowledge_files(
        db,
        knowledge_base.id,
        user.id,
        limit=resolve_list_limit(limit),
        before_id=before_id,
    )
    return [crud_knowledge_file.serialize_knowledge_file(item) for item in files]


async def upload_knowledge(request: Request, file: UploadFile = File(...), knowledge_base_id: int | None = Form(None), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    knowledge_base = resolve_knowledge_base(db, knowledge_base_id, user.id)

    # 类型校验在读取之前完成：白名单外的文件既不解码也不落库。
    if resolve_knowledge_upload_type(file.content_type, file.filename) is None:
        raise HTTPException(400, KNOWLEDGE_UPLOAD_TYPE_ERROR_MESSAGE)
    # content-length 预检 + 分块读取时二次核验，双保险拦截超限文件。
    if upload_body_exceeds_limit(request.headers.get("content-length"), KNOWLEDGE_UPLOAD_MAX_BYTES):
        raise HTTPException(400, knowledge_upload_too_large_message())

    content = await read_upload_within_limit(file, KNOWLEDGE_UPLOAD_MAX_BYTES)
    # 整份文档的入库预算从这里起算，解析、分块、向量化共享同一个截止时刻。
    deadline = ingest_deadline()
    # 解析（含 .docx/.pdf 解码）是同步 CPU 活：直接调用会独占事件循环，
    # 大文件上传期间同进程其它请求（含健康检查）全部停摆。
    try:
        text = await run_ingest_step(
            crud_knowledge_file.extract_file_text, file.filename or "", content, deadline=deadline
        )
    except KnowledgeIngestTimeout:
        # 此时既没落库也没写向量，无需清理，直接按超时回 500。
        raise HTTPException(500, INGEST_TIMEOUT_MESSAGE)

    # 抽取成功但结果为空（扫描件、图片型 PDF、空文本文件）不是入库成功：空文本会被
    # chunk_text 静默转成 []，再由 add_chunks 的 replace_empty 分支正常返回，最后落下一行
    # 「出现在列表里、向量库 0 条、原文不落盘」的记录——它永远检索不到，也没有任何回填入口。
    # 放在落库之前：此时既没有元数据行也没有向量，失败无需任何清理（与上面的超时分支同址）。
    if not (text or "").strip():
        logger.warning(
            "Knowledge file upload rejected for empty extracted text: filename=%s",
            file.filename,
        )
        raise HTTPException(400, crud_knowledge_file.EMPTY_EXTRACTED_TEXT_MESSAGE)

    try:
        entry = crud_knowledge_file.create_knowledge_file(
            db,
            knowledge_base_id=knowledge_base.id,
            name=file.filename or "unknown",
            size=len(content),
            content=text,
            user_id=user.id,
        )
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning("Knowledge file metadata save failed: filename=%s error=%s", file.filename, exc, exc_info=True)
        raise HTTPException(500, crud_knowledge_file.knowledge_file_save_error_message(exc))

    try:
        chunks = await run_ingest_step(
            crud_knowledge_file.chunk_text, text, entry.id, deadline=deadline
        )
    except KnowledgeIngestTimeout:
        # 分块超时发生在写向量之前，只回滚刚落下的元数据行。
        try:
            crud_knowledge_file.delete_knowledge_file(db, entry.id, user.id)
        except SQLAlchemyError as cleanup_commit_exc:
            db.rollback()
            logger.warning("Failed to remove knowledge file after ingest timeout: file_id=%s error=%s", entry.id, cleanup_commit_exc, exc_info=True)
        raise HTTPException(500, INGEST_TIMEOUT_MESSAGE)

    _warn_on_low_chunk_coverage(entry, text, chunks, scope="Knowledge file upload")
    # 切片结构元数据（issue #287）先于向量写入落库：后续任何一步失败都能按 file_id
    # 找到并清掉它；老文件没有这张表的行时检索照常按文本路径降级。
    try:
        crud_knowledge_file.replace_chunk_metadata(db, entry.id, knowledge_base.id, chunks)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning(
            "Knowledge chunk metadata save failed: file_id=%s error=%s", entry.id, exc, exc_info=True
        )
        try:
            crud_knowledge_file.delete_knowledge_file(db, entry.id, user.id)
        except SQLAlchemyError as cleanup_commit_exc:
            db.rollback()
            logger.warning(
                "Failed to rollback knowledge file after metadata save failure: file_id=%s error=%s",
                entry.id,
                cleanup_commit_exc,
                exc_info=True,
            )
        raise HTTPException(500, "知识文件上传失败，切片元数据写入异常。")
    try:
        # 向量化外呼与 Milvus 写入都是同步的，放进入库专用线程池执行：既不让事件循环
        # 被独占，也不与检索侧抢 asyncio 默认池。deadline 同时传给 add_chunks：它不只是
        # 等待上限，还限制重试退避——上游 Retry-After 再大也不能让线程睡过整份文档预算。
        await run_ingest_step(
            partial(add_chunks, deadline=deadline),
            chunks,
            entry.id,
            entry.name,
            knowledge_base.id,
            deadline=deadline,
        )
    except KnowledgeIngestTimeout as exc:
        # 写入线程还在跑，不能在这里清理：此时删掉的向量会被它重新写回来，
        # 留下「列表里没有、检索却命中」的孤儿向量。客户端先拿到 500，
        # 清理挂到工作项完成之后（顺序由线程池的 done_callback 保证）。
        logger.warning("Knowledge file ingest timed out: file_id=%s error=%s", entry.id, exc, exc_info=True)
        defer_ingest_cleanup(exc.future, entry.id, user.id)
        raise HTTPException(500, _index_failure_detail(exc))
    except Exception as exc:
        logger.warning("Knowledge file indexing failed: file_id=%s error=%s", entry.id, exc, exc_info=True)
        # 与超时清理同一条判据（见 _delete_file_vectors_and_row）：向量清理确认成功
        # 才删元数据行。add_chunks 可能已经写过一部分，清理失败还删行就是孤儿向量。
        vectors_removed = False
        try:
            # 清理走清理专用池并自带总时限（minor-1）：入库池只有 4 个槽位，被卡死的
            # 写入线程占满时这一步会无限期排队；不设上限还会把已经失败的请求再挂死。
            await run_cleanup_step(delete_file_chunks, entry.id, deadline=ingest_deadline())
            vectors_removed = True
        except KnowledgeIngestTimeout as cleanup_exc:
            # 超时只放弃等待：工作项已提交、会在后台跑完，删成没删成无从得知，
            # 因此按「没删干净」处理（下面保留行），而不是赌它成功。
            logger.warning(
                "Vector cleanup timed out after indexing failure, abandoned the wait: file_id=%s error=%s",
                entry.id,
                cleanup_exc,
                exc_info=True,
            )
        except Exception as cleanup_exc:
            logger.warning("Failed to clean partially indexed chunks: file_id=%s error=%s", entry.id, cleanup_exc, exc_info=True)
        if vectors_removed:
            try:
                crud_knowledge_file.delete_knowledge_file(db, entry.id, user.id)
            except SQLAlchemyError as cleanup_commit_exc:
                db.rollback()
                logger.warning("Failed to rollback partially indexed knowledge file: file_id=%s error=%s", entry.id, cleanup_commit_exc, exc_info=True)
        else:
            _keep_knowledge_file_row(entry.id, reason="indexing failed and vector cleanup did not confirm success")
            # issue #287：文件行留给启动重建，但本次已写入的切片元数据要清掉，
            # 不对应任何完整向量的元数据没有保留价值。
            _delete_chunk_metadata_best_effort(db, entry.id)
        raise HTTPException(500, _index_failure_detail(exc))

    # 行已提交、向量已写入：立刻在后台重建该知识库的 BM25 语料，让上传后的第一次提问
    # 不必自己承担整库扫描（issue #288 返工轮 medium-1）。预热是尽力而为，失败不影响上传。
    schedule_bm25_prewarm(knowledge_base.id)
    return crud_knowledge_file.serialize_knowledge_file(entry)


def delete_knowledge(fid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entry = crud_knowledge_file.get_knowledge_file(db, fid, user.id)
    if not entry:
        # 文件不存在或不属于当前用户，统一 404，不暴露资源存在性。
        raise HTTPException(404, "文件不存在")
    _delete_file_vectors_or_500(
        fid,
        scope="Knowledge file",
        detail="知识文件删除失败，向量库清理异常，请稍后重试。",
    )
    knowledge_base_id = getattr(entry, "knowledge_base_id", None)
    crud_knowledge_file.delete_knowledge_file(db, fid, user.id)
    # 删除同样改变了该知识库的 BM25 语料：后台重建，避免下一问仍扫描到已删内容。
    if knowledge_base_id is not None:
        schedule_bm25_prewarm(knowledge_base_id)
    return {"message": "ok"}


def get_knowledge_detail(fid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entry = crud_knowledge_file.get_knowledge_file(db, fid, user.id)
    if not entry:
        raise HTTPException(404, "文件不存在")
    return crud_knowledge_file.serialize_knowledge_file(entry)


def get_knowledge_content(fid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entry = crud_knowledge_file.get_knowledge_file(db, fid, user.id)
    if not entry:
        raise HTTPException(404, "文件不存在")
    return crud_knowledge_file.get_knowledge_content(db, fid, user.id)
