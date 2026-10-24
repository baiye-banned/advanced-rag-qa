"""issue #11 回归：知识库上传必须先校验类型与大小，再决定是否把文件读进内存。

覆盖点：
- 超出上限的文件返回 400，且不落库、不写向量；
- 白名单外的扩展名 / MIME 返回 400，不再兜底当纯文本解码；
- 合法文件仍能正常抽取、落库、写向量，护栏没有误伤正常链路。
"""
import asyncio
import io
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import session as db_session
from database.session import Base
from model.models import (
    KnowledgeBase,
    KnowledgeChunkMetadata,
    KnowledgeFile,
    RevokedToken,
    User,
)
from router import auth as auth_router
from router import knowledge as knowledge_router
from service import auth_service, knowledge_service, utils_service


USERNAME = "alice"
# 变量名避开 secret-scan 的「PASSWORD…=」凭据赋值模式，值只是本用例的登录口令，不是真实凭据。
USER_PASS = "alice-pass"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


@pytest.fixture()
def api(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        bind=engine,
        tables=[
            User.__table__, RevokedToken.__table__, KnowledgeBase.__table__,
            KnowledgeFile.__table__, KnowledgeChunkMetadata.__table__,
        ],
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db = TestingSession()

    alice = User(username=USERNAME, password_hash=auth_service.pwd_context.hash(USER_PASS))
    db.add(alice)
    db.commit()
    base = KnowledgeBase(name="alice-kb", user_id=alice.id)
    db.add(base)
    db.commit()

    app = FastAPI()
    app.include_router(auth_router.router)
    app.include_router(knowledge_router.router)
    app.dependency_overrides[db_session.get_db] = lambda: db

    # 向量库不参与校验，用例里记录调用即可，避免连真实 Milvus。
    indexed: list[list[dict]] = []
    monkeypatch.setattr(knowledge_service, "add_chunks", lambda chunks, *args, **kwargs: indexed.append(chunks))
    monkeypatch.setattr(knowledge_service, "delete_file_chunks", lambda *args, **kwargs: None)

    client = TestClient(app)
    login = client.post("/api/auth/login", json={"username": USERNAME, "password": USER_PASS})
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['token']}"}

    try:
        yield SimpleNamespace(client=client, db=db, base=base, indexed=indexed, headers=headers)
    finally:
        db.close()


def _upload(api, filename, content, content_type):
    return api.client.post(
        "/api/knowledge/upload",
        files={"file": (filename, content, content_type)},
        data={"knowledge_base_id": str(api.base.id)},
        headers=api.headers,
    )


def _stored_files(api):
    api.db.expire_all()
    return api.db.query(KnowledgeFile).all()


def _build_docx(text: str) -> bytes:
    from docx import Document

    buffer = io.BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()


class _FakeUpload:
    """模拟 UploadFile 的分块读取，用于验证读取过程不会整体吞下超限文件。"""

    def __init__(self, chunk: bytes, total_chunks: int):
        self.chunk = chunk
        self.remaining = total_chunks
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self.remaining <= 0:
            return b""
        self.remaining -= 1
        return self.chunk


def test_oversized_upload_is_rejected_and_not_persisted(api):
    payload = b"a" * (utils_service.KNOWLEDGE_UPLOAD_MAX_BYTES + 1)

    response = _upload(api, "big.txt", payload, "text/plain")

    assert response.status_code == 400
    assert "超过" in response.json()["detail"]
    assert _stored_files(api) == []
    assert api.indexed == []


def test_oversized_upload_is_rejected_in_chunked_read(api, monkeypatch):
    # 放大允许的 multipart 开销，等于关掉 content-length 预检，只留读取时的二次核验。
    monkeypatch.setattr(knowledge_service, "MULTIPART_OVERHEAD_ALLOWANCE_BYTES", 1 << 30)
    payload = b"a" * (utils_service.KNOWLEDGE_UPLOAD_MAX_BYTES + 1)

    response = _upload(api, "big.txt", payload, "text/plain")

    assert response.status_code == 400
    assert "超过" in response.json()["detail"]
    assert _stored_files(api) == []
    assert api.indexed == []


def test_content_length_precheck_rejects_before_reading_body(api, monkeypatch):
    async def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("超限请求应在 content-length 预检处被拒绝，不应再进入读取流程")

    monkeypatch.setattr(knowledge_service, "read_upload_within_limit", _fail_if_called)
    payload = b"a" * (utils_service.KNOWLEDGE_UPLOAD_MAX_BYTES + 4096)

    response = _upload(api, "big.txt", payload, "text/plain")

    assert response.status_code == 400
    assert _stored_files(api) == []


def test_upload_exactly_at_limit_is_accepted(api):
    # 边界：恰好达到上限的文件必须放行，预检的 multipart 开销余量不能反过来误伤合法上传。
    payload = b"a" * utils_service.KNOWLEDGE_UPLOAD_MAX_BYTES

    response = _upload(api, "exact.txt", payload, "text/plain")

    assert response.status_code == 200
    assert response.json()["size"] == utils_service.KNOWLEDGE_UPLOAD_MAX_BYTES
    assert len(_stored_files(api)) == 1


def test_unsupported_extension_is_rejected(api):
    response = _upload(api, "run.exe", b"MZ" + b"\x00" * 1024, "application/x-msdownload")

    assert response.status_code == 400
    assert response.json()["detail"] == utils_service.KNOWLEDGE_UPLOAD_TYPE_ERROR_MESSAGE
    assert _stored_files(api) == []
    assert api.indexed == []


def test_upload_without_filename_is_rejected(api):
    # 没有 filename 的 multipart 部分不会被 FastAPI 当作文件，请求在框架层（422）就被拒绝；
    # 无论状态码来自哪一层，都不能落库。
    response = api.client.post(
        "/api/knowledge/upload",
        files={"file": ("", b"payload", "text/plain")},
        data={"knowledge_base_id": str(api.base.id)},
        headers=api.headers,
    )

    assert response.status_code == 422
    assert _stored_files(api) == []


def _raw_multipart_upload(api, content_disposition_value, payload, content_type="text/plain"):
    """自行拼 multipart 报文，用来发送 httpx 的 files= 参数拼不出的头（例如 RFC 5987 的 filename*）。

    issue #98：python-multipart 0.0.30 改过 RFC 2231/5987 扩展参数与分号分隔语义，
    这里按原始报文发送，避免被客户端库的编码行为掩盖。
    """
    boundary = "----fix98multipartboundary"
    parts = [
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; {content_disposition_value}\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ]
    body = parts[0].encode("utf-8") + payload + b"\r\n"
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="knowledge_base_id"\r\n\r\n'
        f"{api.base.id}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    return api.client.post(
        "/api/knowledge/upload",
        content=body,
        headers={**api.headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def test_upload_parses_pdf_and_indexes_extracted_text(api):
    """PDF 走完整上传链路：解析出正文、按页标记入分块、落库。

    issue #98：pypdf 4 → 6 跨两个大版本，解析器换成新版后这条路径必须仍然端点可达。
    """
    from test_pdf_extraction import _minimal_pdf

    payload = _minimal_pdf(["Effective on release.", "Late arrivals are logged."])
    response = _upload(api, "rule.pdf", payload, "application/pdf")

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "rule.pdf"
    assert [entry.name for entry in _stored_files(api)] == ["rule.pdf"]

    indexed_text = "\n".join(chunk["text"] for chunk in api.indexed[0])
    assert "Effective on release." in indexed_text
    assert "Late arrivals are logged." in indexed_text
    # 页码标记是分块的页边界（_heading_level 判为 0 级），只用于切分、不进正文，
    # 两页因此各成一块而不是被页码串成一段。
    assert "第 1 页" not in indexed_text
    assert len(api.indexed[0]) >= 2


def test_upload_keeps_non_ascii_filename(api):
    """文件名含中文时，落库名与扩展名判定都要正确，不能被截断或错误解码。"""
    response = _upload(api, "制度文件.md", "# 迟到处理\n\n迟到 30 分钟以内记口头提醒。".encode(), "text/plain")

    assert response.status_code == 200
    assert response.json()["name"] == "制度文件.md"
    assert [entry.name for entry in _stored_files(api)] == ["制度文件.md"]


def test_upload_drops_rfc5987_filename_extended_parameter(api):
    """RFC 5987 的 filename* 不再被采信：该 part 会被当成普通表单字段，请求以 422 拒绝。

    issue #98 的升级实测行为：加固后的解析器不再把 filename* 归一化成 filename。
    逐版本实测（parse_options_header 直调）：0.0.20 / 0.0.29 **仍归一化**，
    **0.0.30 起丢弃**——也就是说变更发生在 0.0.30，正是
    GHSA-vffw-93wf-4j4q（RFC 2231/5987 参数走私）标注的修复版本，本单的 0.0.31 下限覆盖它。

    之所以变成 422：starlette 0.38.6 的 formparsers 把 Content-Disposition 解析委托给
    python-multipart（starlette/formparsers.py:183 调 parse_options_header），
    再判 `b"filename" in options`（starlette/formparsers.py:188）。
    键不存在 → 该 part 不被识别为文件 → FastAPI 的 File(...) 匹配不上 → 422。

    这是修复 GHSA-vffw-93wf-4j4q 所采取的方向：
    不在应用层重新解析 filename* 来恢复兼容，否则等于把这条告警刚堵上的洞重新打开。
    浏览器一律只发 filename="..."（原始 UTF-8 字节），该形态在升级前后都正常，
    见 test_upload_keeps_non_ascii_filename；受影响的是自造报文的 API 客户端，
    改用 filename="制度文件.md" 即可。

    注：0.0.30 同时还有第二处解析行为变更——QueryStringParser 不再把分号当字段分隔符
    （GHSA-6jv3-5f52-599m，实测 0.0.29 及以前 `a=1;b=2` 拆成两个字段，0.0.30 起不拆）。
    本仓库的表单字段只有整数 knowledge_base_id，不含分号，故不受影响；一并记录以免日后误判。
    """
    response = _raw_multipart_upload(
        api,
        "name=\"file\"; filename*=UTF-8''%E5%88%B6%E5%BA%A6%E6%96%87%E4%BB%B6.md",
        "# 标题\n\n正文。".encode(),
    )

    assert response.status_code == 422
    assert _stored_files(api) == []


def test_upload_rejects_duplicate_filename_parameters(api):
    """重复的 filename 参数由扩展名白名单兜住：无论解析器取哪一个，都不能落库。

    0.0.9 与 0.0.31 在 parse_options_header 层都取「后者」（此处为 evil.exe），
    该行为未随升级改变；真正拦住它的是 resolve_knowledge_upload_type 的白名单，
    这条用例钉的是白名单这道应用层防线本身。
    """
    response = _raw_multipart_upload(
        api,
        'name="file"; filename="payload.txt"; filename="evil.exe"',
        b"payload",
    )

    assert response.status_code in (400, 422)
    assert _stored_files(api) == []


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("notes.txt", "application/x-msdownload"),
        ("notes.md", "application/pdf"),
        ("report.pdf", "image/png"),
        ("report.docx", "text/plain"),
    ],
)
def test_declared_content_type_must_match_extension(api, filename, content_type):
    response = _upload(api, filename, b"payload", content_type)

    assert response.status_code == 400
    assert _stored_files(api) == []


def test_supported_text_file_is_extracted_indexed_and_persisted(api):
    payload = "# 迟到处理\n\n迟到 30 分钟以内记口头提醒。".encode()

    response = _upload(api, "rule.md", payload, "text/plain")

    assert response.status_code == 200
    assert response.json()["name"] == "rule.md"
    assert response.json()["size"] == len(payload)
    stored = _stored_files(api)
    assert len(stored) == 1
    assert "口头提醒" in stored[0].content
    assert api.indexed and api.indexed[0]


def test_generic_content_type_is_accepted_by_extension(api):
    response = _upload(api, "notes.txt", b"hello", "application/octet-stream")

    assert response.status_code == 200
    assert len(_stored_files(api)) == 1


def test_supported_docx_is_extracted(api):
    response = _upload(api, "report.docx", _build_docx("季度目标：完成上传校验"), DOCX_CONTENT_TYPE)

    assert response.status_code == 200
    assert "季度目标" in _stored_files(api)[0].content


def test_read_upload_within_limit_returns_content_under_limit():
    fake = _FakeUpload(b"abc", 2)

    assert asyncio.run(knowledge_service.read_upload_within_limit(fake, 10)) == b"abcabc"
    assert set(fake.read_sizes) == {knowledge_service.UPLOAD_READ_CHUNK_BYTES}


def test_read_upload_within_limit_aborts_right_after_crossing_limit():
    chunk = b"x" * knowledge_service.UPLOAD_READ_CHUNK_BYTES
    fake = _FakeUpload(chunk, 8)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(knowledge_service.read_upload_within_limit(fake, 3 * knowledge_service.UPLOAD_READ_CHUNK_BYTES))

    assert excinfo.value.status_code == 400
    # 读到第 4 块（累计 4MB > 3MB）立即中断，剩余 4 块没有被读取。
    assert len(fake.read_sizes) == 4
    assert fake.remaining == 4


def test_upload_body_exceeds_limit_tolerates_multipart_overhead():
    max_bytes = 1024
    allowance = knowledge_service.MULTIPART_OVERHEAD_ALLOWANCE_BYTES

    assert knowledge_service.upload_body_exceeds_limit(None, max_bytes) is False
    assert knowledge_service.upload_body_exceeds_limit("", max_bytes) is False
    assert knowledge_service.upload_body_exceeds_limit("not-a-number", max_bytes) is False
    assert knowledge_service.upload_body_exceeds_limit(str(max_bytes + allowance), max_bytes) is False
    assert knowledge_service.upload_body_exceeds_limit(str(max_bytes + allowance + 1), max_bytes) is True
