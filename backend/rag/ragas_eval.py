import asyncio
import concurrent.futures
import json
import logging
import time

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RAGAS_API_KEY,
    RAGAS_BASE_URL,
    RAGAS_ENABLED,
    RAGAS_LLM_MODEL,
    RAGAS_MAX_ANSWER_CHARS,
    RAGAS_MAX_CONTEXT_CHARS,
    RAGAS_MAX_CONTEXTS,
    RAGAS_METRIC_TIMEOUT_SECONDS,
    RAGAS_TIMEOUT_SECONDS,
)
from database.session import SessionLocal
from rag.learning_trace import append_trace_event, summarize_text
from model.models import Message
from rag.llm import normalize_deepseek_model, openai_base_url
from service.utils_service import _internal_error_detail


logger = logging.getLogger(__name__)

# RAGAS 失败回给用户的固定文案：`_friendly_error` 的结果会经 `Message.ragas_error`
# 渲染到前端，异常原文（驱动报错、文件路径、上游地址）只进 logger。
RAGAS_EMBEDDING_FAILED_MESSAGE = "Embedding 调用失败，请检查 RAGAS 所用的 embedding 服务配置"
RAGAS_LLM_FAILED_MESSAGE = "DeepSeek 调用失败，请检查 RAGAS 所用的模型服务配置"
RAGAS_EVALUATION_FAILED_MESSAGE = "评测过程发生未预期的错误"


async def evaluate_message_async(
    message_id: int,
    question: str,
    answer: str,
    contexts: list[str],
    trace_id: str | None = None,
):
    if not RAGAS_ENABLED:
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "ragas_disabled",
            "evaluate_message_async",
            result={"message_id": message_id},
            note="RAGAS 已关闭，系统不会执行本轮质量评估。",
        )
        _mark_message(message_id, "failed", {}, "RAGAS 已关闭")
        return

    logger.info("RAGAS evaluation started: message_id=%s contexts=%s", message_id, len(contexts or []))
    await asyncio.to_thread(
        append_trace_event,
        trace_id,
        "ragas_running",
        "evaluate_message_async",
        params={
            "message_id": message_id,
            "question": question,
            "answer": summarize_text(answer, 500),
            "contexts_count": len(contexts or []),
        },
        note="RAGAS 后台任务开始运行，消息状态从 pending 进入 running。",
    )
    _mark_message(message_id, "running", {}, "")
    try:
        prepared_answer = _truncate_text(answer, RAGAS_MAX_ANSWER_CHARS)
        prepared_contexts = _prepare_contexts(contexts)
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "ragas_inputs_prepared",
            "evaluate_message_async",
            creates={
                "prepared_answer_chars": len(prepared_answer),
                "prepared_contexts_count": len(prepared_contexts),
                "max_contexts": RAGAS_MAX_CONTEXTS,
            },
            note="RAGAS 会先裁剪回答和上下文，避免评估请求过长。",
        )
        scores = await asyncio.wait_for(
            asyncio.to_thread(_evaluate_message_sync, question, prepared_answer, prepared_contexts, trace_id),
            timeout=RAGAS_TIMEOUT_SECONDS,
        )
        errors = scores.pop("_errors", {})
        if scores:
            error_text = _format_metric_errors(errors)
            _mark_message(message_id, "done", scores, error_text)
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "ragas_done",
                "evaluate_message_async",
                result={"scores": scores, "errors": errors},
                status="done",
                note="RAGAS 至少有一个指标评估成功，结果已写回 assistant 消息。",
            )
            logger.info("RAGAS evaluation done: message_id=%s scores=%s errors=%s", message_id, scores, errors)
        else:
            error_text = _format_metric_errors(errors) or "RAGAS 评测失败：所有指标均未返回结果"
            _mark_message(message_id, "failed", {}, error_text)
            await asyncio.to_thread(
                append_trace_event,
                trace_id,
                "ragas_failed",
                "evaluate_message_async",
                result={"errors": errors, "message": error_text},
                status="done",
                note="RAGAS 所有指标都失败了；这不会影响已生成回答，但 UI 会展示评估失败。",
            )
            logger.warning("RAGAS evaluation failed: message_id=%s errors=%s", message_id, errors)
    except asyncio.TimeoutError:
        message = f"RAGAS 评测超时：超过 {RAGAS_TIMEOUT_SECONDS} 秒未完成，请稍后重试"
        logger.warning("RAGAS evaluation timeout: message_id=%s timeout=%s", message_id, RAGAS_TIMEOUT_SECONDS)
        _mark_message(message_id, "failed", {}, message)
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "ragas_timeout",
            "evaluate_message_async",
            result={"timeout_seconds": RAGAS_TIMEOUT_SECONDS},
            status="done",
            note="RAGAS 总耗时超过上限，系统将评估状态置为 failed。",
        )
    except Exception as exc:
        # 算一次复用：`_friendly_error` 每次都生成新的错误编号，调用两次会让消息里
        # 的编号与轨迹事件里的对不上，用户报了也没法在日志里定位。
        friendly = _friendly_error(exc)
        logger.warning("RAGAS evaluation failed: message_id=%s error=%s", message_id, exc, exc_info=True)
        _mark_message(message_id, "failed", {}, f"RAGAS 评测失败：{friendly}")
        await asyncio.to_thread(
            append_trace_event,
            trace_id,
            "ragas_failed",
            "evaluate_message_async",
            result={"error": friendly},
            status="done",
            note="RAGAS 评估异常，错误信息已写回消息。",
        )


def schedule_ragas_evaluation(
    message_id: int,
    question: str,
    answer: str,
    contexts: list[str],
    trace_id: str | None = None,
):
    try:
        asyncio.create_task(evaluate_message_async(message_id, question, answer, contexts, trace_id))
    except RuntimeError:
        _mark_message(message_id, "failed", {}, "当前事件循环不可用，无法启动 RAGAS 评测")
        append_trace_event(
            trace_id,
            "ragas_schedule_failed",
            "schedule_ragas_evaluation",
            result={"message_id": message_id},
            note="当前事件循环不可用，RAGAS 后台任务没有成功创建。",
        )


def _ragas_llm_client_config() -> dict:
    """RAGAS evaluator 使用的 OpenAI 客户端参数（不构造客户端，便于离线断言）。"""
    return {
        "api_key": RAGAS_API_KEY,
        "base_url": openai_base_url(RAGAS_BASE_URL),
    }


def _evaluate_message_sync(question: str, answer: str, contexts: list[str], trace_id: str | None = None) -> dict:
    if not RAGAS_API_KEY:
        raise RuntimeError("RAGAS API Key 未配置")
    if not EMBEDDING_BASE_URL or not EMBEDDING_API_KEY:
        raise RuntimeError("Embedding API 未配置")

    from openai import OpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    embedding_client = OpenAI(
        api_key=EMBEDDING_API_KEY,
        base_url=openai_base_url(EMBEDDING_BASE_URL),
    )
    embeddings = _ResponseRelevancyEmbeddingsAdapter(
        OpenAIEmbeddings(client=embedding_client, model=EMBEDDING_MODEL)
    )

    llm_client = OpenAI(**_ragas_llm_client_config())
    llm = _make_ragas_llm(llm_factory, llm_client)
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts or [],
    )

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision_without_reference": LLMContextPrecisionWithoutReference(llm=llm),
        "response_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
    }

    scores = {}
    errors = {}
    for key, metric in metrics.items():
        started_at = time.monotonic()
        try:
            scores[key] = _score_metric_with_timeout(metric, sample, RAGAS_METRIC_TIMEOUT_SECONDS)
            append_trace_event(
                trace_id,
                "ragas_metric_done",
                "_evaluate_message_sync",
                result={"metric": key, "score": scores[key], "seconds": round(time.monotonic() - started_at, 2)},
                note=f"RAGAS 指标 {key} 评估成功。",
            )
            logger.info("RAGAS metric done: metric=%s seconds=%.2f", key, time.monotonic() - started_at)
        except Exception as exc:
            errors[key] = _friendly_error(exc)
            append_trace_event(
                trace_id,
                "ragas_metric_failed",
                "_evaluate_message_sync",
                result={"metric": key, "error": errors[key], "seconds": round(time.monotonic() - started_at, 2)},
                note=f"RAGAS 指标 {key} 评估失败，其他指标仍会继续。",
            )
            logger.warning(
                "RAGAS metric failed: metric=%s seconds=%.2f error=%s",
                key,
                time.monotonic() - started_at,
                exc,
                exc_info=True,
            )

    if errors:
        scores["_errors"] = errors
    return scores


def _make_ragas_llm(llm_factory, client):
    model = normalize_deepseek_model(RAGAS_LLM_MODEL)
    return llm_factory(model=model, client=client, max_tokens=4096, temperature=0)


class _ResponseRelevancyEmbeddingsAdapter:
    """Expose the LangChain-style methods expected by RAGAS 0.4.2."""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_query(self, text: str):
        return self.embeddings.embed_text(text, dimensions=EMBEDDING_DIM)

    def embed_documents(self, texts: list[str]):
        return self.embeddings.embed_texts(texts, dimensions=EMBEDDING_DIM)


def _score_metric(metric, sample):
    score = metric.single_turn_score(sample)
    return round(float(score), 4)


def _score_metric_with_timeout(metric, sample, timeout_seconds: int):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_score_metric, metric, sample)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"metric timeout after {timeout_seconds}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _mark_message(message_id: int, status: str, scores: dict, error: str):
    db = SessionLocal()
    try:
        message = db.query(Message).filter_by(id=message_id).first()
        if not message:
            return
        message.ragas_status = status
        message.ragas_scores = json.dumps(scores, ensure_ascii=False) if scores else ""
        message.ragas_error = error or ""
        db.commit()
    finally:
        db.close()


def _format_metric_errors(errors: dict) -> str:
    if not errors:
        return ""
    details = "；".join(f"{name}: {message}" for name, message in errors.items())
    return f"部分 RAGAS 指标评测失败：{details}"


def _friendly_error(exc: Exception) -> str:
    """把异常归类成用户可读的原因；异常原文只落日志，不拼进返回值。

    返回值会写进 `Message.ragas_error`（`src/views/Chat.vue` 原样渲染），并进
    `ragas_*` 轨迹事件供用户回查，所以与本 PR 对 SSE 帧 / HTTP detail 的判据一致：
    驱动报错、文件路径、上游地址这类内部文本不外泄，改用固定文案 + 可上报的编号。
    """
    text = str(exc) or exc.__class__.__name__
    lowered = text.lower()
    if "max_tokens" in text or "length limit" in lowered or "incomplete" in lowered:
        return "DeepSeek 结构化输出被截断，请调高 RAGAS evaluator max_tokens 或缩短回答/上下文"
    if "timed out" in lowered or "timeout" in lowered:
        return "RAGAS 评测超时"
    if "embedding" in lowered:
        return _internal_error_detail(RAGAS_EMBEDDING_FAILED_MESSAGE, "ragas_embedding", exc)
    if "deepseek" in lowered or "connection" in lowered or "connect" in lowered:
        return _internal_error_detail(RAGAS_LLM_FAILED_MESSAGE, "ragas_llm", exc)
    return _internal_error_detail(RAGAS_EVALUATION_FAILED_MESSAGE, "ragas_evaluation", exc)


def _prepare_contexts(contexts: list[str]) -> list[str]:
    prepared = []
    for item in contexts or []:
        text = _truncate_text(item, RAGAS_MAX_CONTEXT_CHARS)
        if text:
            prepared.append(text)
        if len(prepared) >= RAGAS_MAX_CONTEXTS:
            break
    return prepared


def _truncate_text(text: str, max_chars: int) -> str:
    value = _text_value(text).strip()
    if not value or max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n...(已截断)"


def _text_value(value) -> str:
    return "" if value is None else str(value)
