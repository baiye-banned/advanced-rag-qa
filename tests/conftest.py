"""测试共享替身与流式响应辅助（issue #18：消除各测试文件里的重复定义）。

只收敛被多个测试文件共用的替身（FakeQuery/FakeDb/FakeUser/FakeKnowledgeBase/TraceRecorder 替身）
以及 SSE 收集/解析辅助；单文件专用的替身（如 test_milvus_client 的 _FakeHttpClient）仍留在各自文件里。
"""

import asyncio
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# 阶段覆盖变量（issue #317）。采用者会在 .env / 宿主环境里配置它们，而 conftest 的路由、
# 记忆、重排等用例都假设没有阶段覆盖：一旦宿主机带值，`llm.get_stage_model` 会绕过被替换
# 的原 getter 并可能真实外呼。用法例级隔离保证测试结果不取决于运行机器的环境。
# `RAGAS_LLM_MODEL` 是既有变量，不属于阶段覆盖路径，保持原样。
STAGE_ENV_NAMES = (
    "ROUTER_API_KEY", "ROUTER_BASE_URL", "ROUTER_MODEL",
    "PLANNER_API_KEY", "PLANNER_BASE_URL", "PLANNER_MODEL",
    "ANSWER_API_KEY", "ANSWER_BASE_URL", "ANSWER_MODEL",
    "MEMORY_API_KEY", "MEMORY_BASE_URL", "MEMORY_MODEL",
    "RERANK_LLM_API_KEY", "RERANK_LLM_BASE_URL", "RERANK_LLM_MODEL",
    "RAGAS_API_KEY", "RAGAS_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_stage_env(monkeypatch):
    """每条用例从「无阶段覆盖」开始；用例内的 monkeypatch.setenv 仍可覆盖。"""
    for name in STAGE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class FakeQuery:
    """SQLAlchemy Query 的最小替身：只实现被测代码真正用到的 filter_by/first 组合。"""

    def __init__(self, rows):
        self.rows = rows

    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDb:
    """Session 替身：记录写入与事务状态，供用例断言「被测代码对数据库做了什么」。"""

    def __init__(self, user=None):
        self.user = user
        self.added = []
        self.commits = 0
        self.closed = False
        self.rolled_back = False

    def query(self, model):
        # 延迟导入：conftest 在收集阶段不应该把整个应用拉起来
        from service import chat_service

        if model is chat_service.User:
            return FakeQuery([self.user] if self.user is not None else [])
        return FakeQuery([])

    def add(self, item):
        self.added.append(item)

    def commit(self):
        self.commits += 1

    def refresh(self, item):
        if getattr(item, "id", None) is None:
            item.id = 1

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True

    def added_by_role(self, role):
        return [item for item in self.added if getattr(item, "role", "") == role]


class FakeUser:
    id = 7
    username = "alice"


class FakeKnowledgeBase:
    id = 3
    name = "制度库"


class FakeTraceRecorder:
    """TraceRecorder 替身：真实实现把写库交给工作线程，测试只保留内存事件与 SSE 游标。

    `add` / `attach` / `finish` 与真实实现同样是协程——调用方一律 `await`，替身若留在同步
    形态，`await None` 会直接炸，反而看不清是哪条断言坏了。
    """

    instances: list["FakeTraceRecorder"] = []

    def __init__(self, user_id=None):
        self.user_id = user_id
        self.trace_id = "trace-test"
        self.status = "running"
        self.events = []
        self.attachments = {}
        self.finished = {}
        self._cursor = 0
        FakeTraceRecorder.instances.append(self)

    async def add(self, stage, function, **payload):
        event = {"index": len(self.events) + 1, "stage": stage, "function": function, **payload}
        self.events.append(event)
        return event

    async def attach(self, conversation_id=None, message_id=None, status=None):
        if status:
            self.status = status
        self.attachments = {"conversation_id": conversation_id, "message_id": message_id}

    async def finish(self, status="done", **extra):
        self.status = status
        self.finished = {"status": status, **extra}

    def snapshot(self):
        return {"trace_id": self.trace_id, "status": self.status, "events": self.events}

    def drain_sse_payloads(self):
        payloads = []
        while self._cursor < len(self.events):
            payloads.append({
                "type": "trace",
                "trace_id": self.trace_id,
                "event": self.events[self._cursor],
            })
            self._cursor += 1
        return payloads

    def stages(self):
        return [event["stage"] for event in self.events]

    def event(self, stage):
        return next(event for event in self.events if event["stage"] == stage)


@pytest.fixture
def fake_user():
    return FakeUser()


@pytest.fixture
def fake_db(fake_user):
    return FakeDb(fake_user)


@pytest.fixture
def fake_knowledge_base():
    return FakeKnowledgeBase()


@pytest.fixture
def trace_recorder_cls():
    """返回已清空实例登记的 TraceRecorder 替身类，供 monkeypatch.setattr 使用。"""
    FakeTraceRecorder.instances = []
    return FakeTraceRecorder


def collect_stream(iterator):
    """收集 StreamingResponse.body_iterator 的全部 chunk 并拼成 SSE 文本。"""

    async def _run():
        chunks = []
        async for chunk in iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    return asyncio.run(_run())


def parse_sse_frames(body):
    """把 SSE 文本拆成 payload 列表，跳过非 data 行与终止帧 [DONE]。"""
    frames = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        payload = block[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        frames.append(json.loads(payload))
    return frames


def frames_of_type(frames, frame_type):
    return [frame for frame in frames if frame.get("type") == frame_type]


def streamed_content(frames):
    """按前端协议拼接 content 帧（reset 帧会清空已累积内容）。"""
    text = ""
    for frame in frames:
        if frame.get("type") == "reset":
            text = ""
            continue
        if "content" in frame:
            text += frame["content"]
    return text
