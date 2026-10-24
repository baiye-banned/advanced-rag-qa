import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
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
DEEPSEEK_TRANSPORT_FAILED_MESSAGE = "DeepSeek 网络/上游请求失败"
DEEPSEEK_CONTENT_FILTERED_MESSAGE = "DeepSeek 模型拒绝生成该内容，请调整提问后重试"
TEXT_FALLBACK_DISABLED_HINT = "；文本后备模型未启用，请设置 TEXT_FALLBACK_ENABLED=true。"
TEXT_FALLBACK_UNCONFIGURED_HINT = "；文本后备模型未配置，请配置 TEXT_FALLBACK_API_KEY 或 DASHSCOPE_API_KEY。"


class LLMCredentialError(RuntimeError):
    """本地缺少模型凭据；单独成型以便与网络/上游失败分流。

    继承 RuntimeError，保持既有调用方与测试的异常契约不变。
    """


def _is_transport_error(exc: Exception) -> bool:
    """网络断开、超时、限流与上游 5xx 归入传输类；本地缺 Key 与模型拒绝另行分流。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _classify_deepseek_failure(exc: Exception) -> tuple[str, str]:
    """把生成异常映射为 (日志分类, 用户固定文案前缀)；异常原文只交给 logger。"""
    if isinstance(exc, LLMCredentialError):
        return "missing_credentials", DEEPSEEK_API_KEY_MISSING_MESSAGE
    if isinstance(exc, ContentFilterFinishReasonError):
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


async def call_chat_text(system_prompt: str, user_prompt: str, *, max_tokens: int = 900) -> str:
    model = get_deepseek_model(streaming=False, temperature=0.1, max_tokens=max_tokens)
    message = await model.ainvoke([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    return _message_text(message.content)


async def call_chat_json(system_prompt: str, user_prompt: str, *, max_tokens: int = 800) -> dict:
    content = await call_chat_text(system_prompt, user_prompt, max_tokens=max_tokens)
    parsed = parse_json_object(content)
    if isinstance(parsed, dict) and parsed.get("error") and "content" in parsed:
        raise ValueError("chat model returned invalid JSON")
    if not isinstance(parsed, dict):
        raise ValueError("chat model returned non-object JSON")
    return parsed


async def call_router_json(payload: dict) -> dict:
    if not TEXT_FALLBACK_API_KEY:
        raise RuntimeError("路由模型 API Key 未配置")
    model = get_text_fallback_model(streaming=False, temperature=0, max_tokens=300)
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
        async for chunk in _stream_model_chunks(get_deepseek_model(streaming=True, temperature=0.1), messages):
            yield chunk
        return
    except Exception as exc:
        failure_kind, failure_prefix = _classify_deepseek_failure(exc)
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
            result={"provider": "DeepSeek", "error": LLM_GENERATION_FAILED_MESSAGE},
            note="DeepSeek 流式生成失败；如果文本后备模型可用，LangChain 会切换到后备模型继续生成。",
        )
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
        async for chunk in _stream_model_chunks(get_text_fallback_model(streaming=True, temperature=0.1), messages):
            yield chunk
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
    async for chunk in model.astream(messages):
        text = _message_text(chunk.content)
        if text:
            yield text


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
