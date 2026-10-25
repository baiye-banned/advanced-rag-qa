import json
import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    ContentFilterFinishReasonError,
    PermissionDeniedError,
    RateLimitError,
)

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    TEXT_FALLBACK_API_KEY,
    TEXT_FALLBACK_BASE_URL,
    TEXT_FALLBACK_ENABLED,
    TEXT_FALLBACK_MODEL,
)
from service.utils_service import _internal_error_detail


logger = logging.getLogger(__name__)

# 生成失败分支回给用户的固定文案：异常原文（上游地址、驱动报错、密钥配置）只进 logger。
# 轨迹事件里的 error 同样是固定文案——它既编进 SSE 帧，也随 retrieval_trace 落库回查。
LLM_GENERATION_FAILED_MESSAGE = "DeepSeek 生成失败"
LLM_TEXT_FALLBACK_FAILED_MESSAGE = "文本后备模型生成失败"
ANSWER_GENERATION_FAILED_MESSAGE = "回答生成失败，请稍后重试"

# 生成失败按原因给固定前缀：缺 Key 不能再说成网络故障，网络故障也不能被当成配置问题。
# 异常原文只进 logger，这些文案才进 SSE error 事件（前端原样显示）。
DEEPSEEK_API_KEY_MISSING_MESSAGE = "DeepSeek API Key 未配置，请在 .env 中配置 DEEPSEEK_API_KEY"  # scan-secrets:allow 用户提示文案，不是凭据值
DEEPSEEK_API_KEY_INVALID_MESSAGE = "DeepSeek API Key 无效或被拒绝，请检查 .env 中的 DEEPSEEK_API_KEY"  # scan-secrets:allow 用户提示文案，不是凭据值
DEEPSEEK_TRANSPORT_FAILED_MESSAGE = "DeepSeek 网络/上游请求失败"
DEEPSEEK_CONTENT_FILTERED_MESSAGE = "DeepSeek 模型拒绝生成该内容，请调整提问后重试"
# 后备模型的拦截判定与 DeepSeek 无关，文案里不能出现 DeepSeek 字样，否则用户会误以为是主模型的问题。
TEXT_FALLBACK_CONTENT_FILTERED_MESSAGE = "模型拒绝生成该内容，请调整提问后重试"
TEXT_FALLBACK_DISABLED_HINT = "；文本后备模型未启用，请设置 TEXT_FALLBACK_ENABLED=true。"
TEXT_FALLBACK_UNCONFIGURED_HINT = "；文本后备模型未配置，请配置 TEXT_FALLBACK_API_KEY 或 DASHSCOPE_API_KEY。"

# DeepSeek 对内容风险既可能回 finish_reason=content_filter（流式），也可能直接回 400
# 「Content Exists Risk」（非流式/网关层）。两种形态都归入同一用户文案，避免空流静默结束。
_CONTENT_FILTER_MARKERS = ("content_filter", "content exists risk", "content risk")


class LLMCredentialError(RuntimeError):
    """本地缺少模型凭据；单独成型以便与网络/上游失败分流。

    继承 RuntimeError，保持既有调用方与测试的异常契约不变。`stage` 只在异常
    确实由阶段独立配置（该阶段 env 有覆盖）触发时携带，供用户文案指向正确的变量。
    """

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message)
        self.stage = stage


def _is_transport_error(exc: Exception) -> bool:
    """网络断开、超时、限流与上游 5xx 归入传输类；本地缺 Key 与模型拒绝另行分流。

    401（`AuthenticationError`）与 403（`PermissionDeniedError`）虽然也是 4xx，但语义是
    凭据缺失/被拒绝，由 `_classify_deepseek_failure` 单独归入 invalid_credentials；其余
    4xx（400/404/422）不是传输故障，留给 generation 兜底。
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _is_content_filter_error(exc: Exception) -> bool:
    """识别模型侧内容过滤：SDK 的 content_filter 异常，或 DeepSeek 的 400 风险提示。"""
    if isinstance(exc, ContentFilterFinishReasonError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 400:
        text = str(exc).lower()
        return any(marker in text for marker in _CONTENT_FILTER_MARKERS)
    return False


def _chunk_finish_reason(chunk: Any) -> str:
    """从流式 chunk 里取 finish_reason；不同版本/供应商放在 response_metadata 或 additional_kwargs。"""
    for attr in ("response_metadata", "additional_kwargs"):
        metadata = getattr(chunk, attr, None)
        if isinstance(metadata, dict):
            reason = metadata.get("finish_reason")
            if reason:
                return str(reason)
    return ""


def _message_finish_reason(message: Any) -> str | None:
    """从非流式响应里取 finish_reason；没有该元数据时返回 None。"""
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict):
        reason = metadata.get("finish_reason")
        if reason:
            return str(reason)
    return None


def _is_content_filter_chunk(chunk: Any) -> bool:
    """流式 chunk 的终态是否由内容过滤触发。

    这条检查是必要的：`ContentFilterFinishReasonError` 只在结构化输出（带 response_format）
    的解析路径上抛；普通 `astream` 只会在最后一个 chunk 的元数据里带 finish_reason，
    不在这里拦下就会变成「空流正常结束」。
    """
    return "content_filter" in _chunk_finish_reason(chunk).lower()


def _chunk_text(chunk: Any) -> str:
    """统一取出流式文本；测试替身可能直接 yield 字符串。"""
    if isinstance(chunk, str):
        return chunk
    return _message_text(getattr(chunk, "content", ""))


def _classify_deepseek_failure(exc: Exception, stage: str | None = None) -> tuple[str, str]:
    """把生成异常映射为 (日志分类, 用户固定文案前缀)；异常原文只交给 logger。

    `stage` 命中已知阶段（或异常自身携带 stage）时，凭据文案指向该阶段的变量，
    例如 ANSWER_API_KEY；未命中时保持原有 DeepSeek 文案，兼容完全继承的部署。
    """
    resolved_stage = getattr(exc, "stage", None) or stage
    if isinstance(exc, LLMCredentialError):
        if resolved_stage in _STAGE_ENV:
            return "missing_credentials", _stage_missing_key_message(resolved_stage)
        return "missing_credentials", DEEPSEEK_API_KEY_MISSING_MESSAGE
    # OpenAI SDK 把 401 映射为 AuthenticationError、403 映射为 PermissionDeniedError，
    # 二者是兄弟类型（都继承 APIStatusError），必须一起识别，否则 403 会掉进 generation 兜底。
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        if resolved_stage in _STAGE_ENV:
            return "invalid_credentials", _stage_invalid_key_message(resolved_stage)
        return "invalid_credentials", DEEPSEEK_API_KEY_INVALID_MESSAGE
    if _is_content_filter_error(exc):
        return "content_filtered", DEEPSEEK_CONTENT_FILTERED_MESSAGE
    if _is_transport_error(exc):
        return "transport", DEEPSEEK_TRANSPORT_FAILED_MESSAGE
    return "generation", LLM_GENERATION_FAILED_MESSAGE


def normalize_deepseek_model(model: str) -> str:
    aliases = {
        "deepseekv4flash": "deepseek-v4-flash",
        "deepseekv4pro": "deepseek-v4-pro",
    }
    return aliases.get((model or "").lower(), model)


def openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def deepseek_chat_url() -> str:
    return f"{openai_base_url(DEEPSEEK_BASE_URL)}/chat/completions"


def openai_chat_url(base_url: str) -> str:
    return f"{openai_base_url(base_url)}/chat/completions"


def get_deepseek_model(*, streaming: bool = False, temperature: float = 0.1, max_tokens: int | None = None):
    if not DEEPSEEK_API_KEY:
        raise LLMCredentialError("DeepSeek API Key 未配置")
    kwargs = {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": openai_base_url(DEEPSEEK_BASE_URL),
        "model": normalize_deepseek_model(DEEPSEEK_MODEL),
        "temperature": temperature,
        "streaming": streaming,
        "timeout": 60,
        "max_retries": 0,
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def get_text_fallback_model(*, streaming: bool = False, temperature: float = 0.1, max_tokens: int | None = None):
    if not TEXT_FALLBACK_API_KEY:
        raise LLMCredentialError("文本后备模型 API Key 未配置")
    kwargs = {
        "api_key": TEXT_FALLBACK_API_KEY,
        "base_url": openai_base_url(TEXT_FALLBACK_BASE_URL),
        "model": TEXT_FALLBACK_MODEL,
        "temperature": temperature,
        "streaming": streaming,
        "timeout": 60,
        "max_retries": 0,
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


# 每阶段独立覆盖用的环境变量。继承关系在调用时逐字段解析：空值（或缺失）的字段
# 读取该阶段来源的当前值，显式覆盖的字段不再跟随来源变化。RAGAS 的真实客户端由
# `ragas_eval._evaluate_message_sync` 自行构造，这里不保留死配置。
_STAGE_ENV = {
    "router": ("ROUTER_API_KEY", "ROUTER_BASE_URL", "ROUTER_MODEL"),
    "planner": ("PLANNER_API_KEY", "PLANNER_BASE_URL", "PLANNER_MODEL"),
    "answer": ("ANSWER_API_KEY", "ANSWER_BASE_URL", "ANSWER_MODEL"),
    "memory": ("MEMORY_API_KEY", "MEMORY_BASE_URL", "MEMORY_MODEL"),
    "rerank_llm": ("RERANK_LLM_API_KEY", "RERANK_LLM_BASE_URL", "RERANK_LLM_MODEL"),
    "text_fallback": ("TEXT_FALLBACK_API_KEY", "TEXT_FALLBACK_BASE_URL", "TEXT_FALLBACK_MODEL"),
}

# 阶段来源：router 继承文本后备模型，其余阶段继承 DeepSeek。
_STAGE_SOURCE = {
    "router": "text_fallback",
    "planner": "deepseek",
    "answer": "deepseek",
    "memory": "deepseek",
    "rerank_llm": "deepseek",
    "text_fallback": "text_fallback",
}

# DeepSeek 别名（deepseekv4flash 等）只在 DeepSeek 家族的阶段里归一化；路由和文本
# 后备是 DashScope 模型，保持配置原文。
_DEEPSEEK_FAMILY_STAGES = {"planner", "answer", "memory", "rerank_llm"}

# 用户文案里的阶段名称，避免把非 DeepSeek 的阶段也说成 DeepSeek。
_STAGE_LABELS = {
    "router": "路由模型",
    "planner": "规划模型",
    "answer": "回答模型",
    "memory": "记忆模型",
    "rerank_llm": "重排模型",
    "text_fallback": "文本后备模型",
}


def _stage_env_value(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _stage_has_override(stage: str) -> bool:
    """该阶段是否有任一环境变量被显式设置为非空值。"""
    return any(_stage_env_value(name) for name in _STAGE_ENV[stage])


def _source_values(source: str) -> tuple[str, str, str]:
    """读取来源的当前模块常量，运行期替换（含测试替身）能逐字段生效。"""
    if source == "text_fallback":
        return TEXT_FALLBACK_API_KEY, TEXT_FALLBACK_BASE_URL, TEXT_FALLBACK_MODEL
    return DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def _stage_values(stage: str) -> tuple[str, str, str]:
    """逐字段解析阶段配置：显式覆盖优先，否则跟随来源的当前值。"""
    source_key, source_base_url, source_model = _source_values(_STAGE_SOURCE[stage])
    api_key_name, base_url_name, model_name = _STAGE_ENV[stage]
    return (
        _stage_env_value(api_key_name) or source_key,
        _stage_env_value(base_url_name) or source_base_url,
        _stage_env_value(model_name) or source_model,
    )


def _stage_missing_key_message(stage: str) -> str:
    return (
        f"{_STAGE_LABELS[stage]} API Key 未配置，"
        f"请在 .env 中配置 {_STAGE_ENV[stage][0]}"
    )


def _stage_invalid_key_message(stage: str) -> str:
    return (
        f"{_STAGE_LABELS[stage]} API Key 无效或被拒绝，"
        f"请检查 .env 中的 {_STAGE_ENV[stage][0]}"
    )


def get_stage_model(
    stage: str,
    *,
    streaming: bool = False,
    temperature: float = 0.1,
    max_tokens: int | None = None,
):
    """按 RAG 阶段取模型，未配置的字段在调用时继承该阶段的默认来源。

    阶段与来源：router -> TEXT_FALLBACK_*，其余阶段 -> DEEPSEEK_*。阶段没有任何
    环境变量覆盖时复用原 getter，保持既有监控和测试替身的行为；只要显式覆盖了任一
    字段，就按「空值继承」逐字段组装客户端。
    """
    if stage not in _STAGE_ENV:
        raise ValueError(f"未知的 RAG 阶段：{stage}")

    # 未显式给预算时不要向下传 max_tokens=None：原 getter 的默认值就是 None，
    # 省略才能让调用方（及其测试替身）看到与改动前完全一致的参数。
    getter_kwargs = {"streaming": streaming, "temperature": temperature}
    if max_tokens is not None:
        getter_kwargs["max_tokens"] = max_tokens

    if stage == "text_fallback":
        return get_text_fallback_model(**getter_kwargs)

    if not _stage_has_override(stage):
        try:
            if _STAGE_SOURCE[stage] == "text_fallback":
                return get_text_fallback_model(**getter_kwargs)
            return get_deepseek_model(**getter_kwargs)
        except LLMCredentialError:
            # 继承来的 Key 为空时，报阶段名比报来源模型名更容易定位到该改哪个变量；
            # 不携带 stage：真正缺的是来源变量，指向阶段变量会误导用户。
            raise LLMCredentialError(f"{stage} API Key 未配置") from None

    api_key, base_url, model = _stage_values(stage)
    if not api_key:
        # 只有 MODEL/BASE_URL 被覆盖、来源 Key 也为空时，提示仍落在该阶段的变量上。
        raise LLMCredentialError(f"{stage} API Key 未配置", stage=stage)
    if stage in _DEEPSEEK_FAMILY_STAGES:
        model = normalize_deepseek_model(model)
    kwargs = {
        "api_key": api_key,
        "base_url": openai_base_url(base_url),
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
        "timeout": 60,
        "max_retries": 0,
    }
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


async def call_chat_text_with_metadata(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 900,
    stage: str = "planner",
) -> tuple[str, str | None]:
    """返回文本与 finish_reason，供调用方识别被 token 上限截断的输出。"""
    model = get_stage_model(
        stage, streaming=False, temperature=0.1, max_tokens=max_tokens
    )
    message = await model.ainvoke([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    return _message_text(message.content), _message_finish_reason(message)


async def call_chat_text(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 900,
    stage: str = "planner",
) -> str:
    content, _ = await call_chat_text_with_metadata(
        system_prompt, user_prompt, max_tokens=max_tokens, stage=stage
    )
    return content


async def call_chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 800,
    stage: str = "planner",
) -> dict:
    content = await call_chat_text(
        system_prompt, user_prompt, max_tokens=max_tokens, stage=stage
    )
    parsed = parse_json_object(content)
    if isinstance(parsed, dict) and parsed.get("error") and "content" in parsed:
        raise ValueError("chat model returned invalid JSON")
    if not isinstance(parsed, dict):
        raise ValueError("chat model returned non-object JSON")
    return parsed


async def call_router_json(payload: dict) -> dict:
    model = get_stage_model("router", streaming=False, temperature=0, max_tokens=300)
    system_prompt = (
        "你是企业知识库 RAG 路由器，只判断当前问题是否需要检索企业知识库。"
        "只输出 JSON，不要输出 Markdown。字段必须是："
        "need_rag(boolean), route(string: rag/direct), confidence(number:0-1), reason(string), source(string)。"
        "如果问题询问公司制度、规章、流程、文件、知识库或上传资料内容，need_rag=true。"
        "如果只是数学、常识、闲聊、翻译、润色、代码解释或当前会话统计，need_rag=false。"
        "图片附件已经被转成文本；如果只需根据图片描述回答，不需要知识库，need_rag=false。"
        "只要不确定，need_rag=true。"
    )
    message = await model.ainvoke([
        ("system", system_prompt),
        ("user", json.dumps(payload, ensure_ascii=False)),
    ])
    content = _message_text(message.content)
    parsed = parse_json_object(content)
    if isinstance(parsed, dict) and parsed.get("error") and "content" in parsed:
        raise ValueError("router model returned invalid JSON")
    if not isinstance(parsed, dict):
        raise ValueError("router model returned non-object JSON")
    return parsed


async def stream_answer_events(
    question: str,
    context: str,
    memory_context: str = "",
    trace: Any = None,
    use_rag: bool = True,
):
    messages = build_answer_messages(question, context, memory_context, use_rag)
    await _trace_add(
        trace,
        "langchain_generation_prompt_built",
        "LangChain ChatOpenAI",
        creates={
            "system_prompt": messages[0][1],
            "user_prompt": messages[1][1],
            "mode": "rag" if use_rag else "direct",
        },
        note="LangChain 生成链构造最终 messages，记忆和知识库上下文分区进入 prompt。",
    )

    try:
        async for chunk in _stream_model_chunks(get_stage_model("answer", streaming=True, temperature=0.1), messages):
            # 内容过滤可能只体现在最后一个 chunk 的 finish_reason 上（普通 astream 不会抛
            # ContentFilterFinishReasonError），必须在这里显式转为 error，否则空流会被当成
            # 正常结束。过滤不是传输故障，不切换后备模型——换个模型继续生成会绕过内容策略。
            if _is_content_filter_chunk(chunk):
                logger.warning("DeepSeek stream generation stopped by content filter")
                await _trace_add(
                    trace,
                    "langchain_generation_failed",
                    "ChatOpenAI.astream",
                    result={
                        "provider": "DeepSeek",
                        "failure_kind": "content_filtered",
                        "error": DEEPSEEK_CONTENT_FILTERED_MESSAGE,
                    },
                    note="DeepSeek 流式生成被内容过滤终止；不切换后备模型，也不保存失败回答。",
                )
                yield {"type": "error", "message": DEEPSEEK_CONTENT_FILTERED_MESSAGE}
                return
            text = _chunk_text(chunk)
            if text:
                yield text
        return
    except Exception as exc:
        # 阶段文案只在 answer 显式覆盖时启用：完全继承 DeepSeek 的部署保持原有
        # DEEPSEEK_API_KEY 文案，避免把继承来的凭据问题说成阶段变量。
        failure_stage = "answer" if _stage_has_override("answer") else None
        failure_kind, failure_prefix = _classify_deepseek_failure(exc, stage=failure_stage)
        logger.warning(
            "DeepSeek stream generation failed (%s): %s",
            failure_kind,
            exc,
            exc_info=True,
        )
        await _trace_add(
            trace,
            "langchain_generation_failed",
            "ChatOpenAI.astream",
            result={
                "provider": "DeepSeek",
                "failure_kind": failure_kind,
                "error": failure_prefix,
            },
            note="DeepSeek 流式生成失败；如果文本后备模型可用，LangChain 会切换到后备模型继续生成。",
        )
        if failure_kind == "content_filtered":
            # 内容过滤与网络/配置故障不同：后备模型不会改变上游的内容裁决。
            yield {"type": "error", "message": DEEPSEEK_CONTENT_FILTERED_MESSAGE}
            return
        if not TEXT_FALLBACK_ENABLED:
            yield {
                "type": "error",
                "message": failure_prefix + TEXT_FALLBACK_DISABLED_HINT,
            }
            return
        if not TEXT_FALLBACK_API_KEY:
            yield {
                "type": "error",
                "message": failure_prefix + TEXT_FALLBACK_UNCONFIGURED_HINT,
            }
            return

    # 后备模型是从头重新生成整段回答，而不是接着 DeepSeek 的半截话往下写。
    # 因此在下发后备模型第一个 chunk 之前，必须先下发 reset 事件作废已输出的内容，
    # 否则消费方（前端渲染、chat_service 落库）会把两段回答首尾拼接成重复文本。
    reset_event = {
        "type": "reset",
        "reason": "text_fallback",
        "message": "DeepSeek 生成中断，已切换到文本后备模型重新生成本轮回答。",
    }
    await _trace_add(
        trace,
        "langchain_stream_reset",
        "ChatOpenAI.astream",
        params={"reason": reset_event["reason"], "model": TEXT_FALLBACK_MODEL},
        note="下发 reset 事件作废已流式输出的 DeepSeek 内容，消费方需清空缓冲后重新累积后备模型输出。",
    )
    yield reset_event

    try:
        await _trace_add(
            trace,
            "langchain_text_fallback_started",
            "ChatOpenAI.astream",
            params={"model": TEXT_FALLBACK_MODEL},
            note="LangChain 切换到 OpenAI 兼容文本后备模型继续本轮回答。",
        )
        async for chunk in _stream_model_chunks(get_stage_model("text_fallback", streaming=True, temperature=0.1), messages):
            if _is_content_filter_chunk(chunk):
                logger.warning("text fallback stream generation stopped by content filter")
                await _trace_add(
                    trace,
                    "langchain_text_fallback_failed",
                    "ChatOpenAI.astream",
                    result={"failure_kind": "content_filtered", "error": TEXT_FALLBACK_CONTENT_FILTERED_MESSAGE},
                    note="文本后备模型同样被内容过滤终止，本轮回答返回错误事件。",
                )
                yield {"type": "error", "message": TEXT_FALLBACK_CONTENT_FILTERED_MESSAGE}
                return
            text = _chunk_text(chunk)
            if text:
                yield text
    except Exception as exc:
        await _trace_add(
            trace,
            "langchain_text_fallback_failed",
            "ChatOpenAI.astream",
            result={"error": LLM_TEXT_FALLBACK_FAILED_MESSAGE},
            note="文本后备模型也生成失败，本轮回答返回错误事件。",
        )
        # 这条 error 事件的 message 前端会原样显示（src/stores/chat.js 直接取 error.message），
        # 因此只给固定文案 + 编号；原文由 _internal_error_detail 带 exc_info 落日志。
        yield {
            "type": "error",
            "message": _internal_error_detail(ANSWER_GENERATION_FAILED_MESSAGE, "text_fallback", exc),
        }


def build_answer_messages(question: str, context: str, memory_context: str, use_rag: bool) -> list[tuple[str, str]]:
    if use_rag:
        system_prompt = (
            "你是企业知识库智能问答助手。请优先基于提供的企业知识库上下文回答用户问题；"
            "如果知识库信息不足，请明确说明不足之处，并给出可验证的建议。"
            "回答使用中文，结构清晰，避免编造未在上下文中出现的事实。"
            "信息优先级为：企业知识库上下文 > 当前用户问题 > 会话记忆。"
            "会话记忆只能用于理解指代、延续任务和用户偏好，不能替代知识库事实依据。"
        )
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"会话记忆：\n{memory_context or '（无）'}\n\n"
            f"企业知识库上下文：\n{context or '（未检索到相关知识库内容）'}"
        )
    else:
        system_prompt = (
            "你是企业对话助手。当前问题不需要知识库检索，请直接回答。"
            "你可以结合会话记忆理解指代和上下文，但不要声称自己检索到了知识库。"
            "如果信息不足，请明确说明，而不是编造知识库事实。"
        )
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"会话记忆：\n{memory_context or '（无）'}\n\n"
            "回答要求：直接回答当前问题，不要引用知识库上下文，也不要提到检索过程。"
        )
    return [("system", system_prompt), ("user", user_prompt)]


async def _stream_model_chunks(model: ChatOpenAI, messages: list[tuple[str, str]]):
    """原样产出 provider 的流式 chunk，让调用方能读到 finish_reason 等元数据。"""
    async for chunk in model.astream(messages):
        yield chunk


def parse_json_object(content: str) -> dict:
    if isinstance(content, (dict, list, int, float, bool)):
        return content
    content = _text_value(content).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    match = re.search(r"\{.*\}", content, flags=re.S)
    if match:
        content = match.group(0)
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "content": content}


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(_text_value(item.get("text")))
                else:
                    parts.append(_text_value(item.get("content")))
            else:
                parts.append(_text_value(item))
        return "".join(parts)
    return _text_value(content)


def _text_value(value) -> str:
    return "" if value is None else str(value)


async def _trace_add(trace: Any, *args, **kwargs) -> None:
    """记录一条生成侧事件；失败只进 debug 日志，绝不影响本轮回答。

    `TraceRecorder.add` 是协程（写库交给工作线程，issue #201），这里 await 它——调用点都在
    `stream_answer_events` 这个异步生成器里，也就是事件循环线程上。
    """
    if not trace:
        return
    try:
        await trace.add(*args, **kwargs)
    except Exception:
        logger.debug("Trace add failed", exc_info=True)
