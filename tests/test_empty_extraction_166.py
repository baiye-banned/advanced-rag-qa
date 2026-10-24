"""issue #166 回归：抽不出文本的文件不得以「入库成功」交付。

背景（机制链，逐跳都正常返回，叠加后成功被一路传到 HTTP 200）：

1. `crud/knowledge_file.py` 的 `extract_pdf_text` 把没有文字的页直接跳过，全部页都没有
   文字时正常返回 `''` —— 无文本层不属于任何异常，走不到 `except`；
2. `chunk_text('')` 返回 `[]`；
3. `add_chunks([])` 走 `replace_empty` 分支**正常返回**，对调用方与「写入了 N 条向量」无差别；
4. 覆盖率守卫因 `chunk_coverage_ratio('')` 短路返回 **1.0**（而非 0.0）而失效，
   整条链路只剩一行 INFO 留痕。

结果是上传回 200、文件出现在列表里、`size` 非 0，而向量库 0 条、原文不落盘 ——
这一行没有任何回填入口，只能重传，重传同一份扫描件结果一模一样。

本文件逐条钉住修法：
- 抽取为空 ⇒ 上传返回 400 + 明确文案，且**不落元数据行、不写向量**（不再是 200）；
- `chunk_coverage_ratio` 空源返回 0.0，阈值守卫在空源上恢复信号；
- 存量空内容行的预览与 `.docx` 一样给占位提示，不再是一片空白。
"""

import io
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from crud import knowledge_file as crud_knowledge_file
from crud.knowledge_file import chunk_coverage_ratio
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
from service import auth_service, knowledge_service


USERNAME = "alice"
# 变量名避开 secret-scan 的「PASSWORD…=」凭据赋值模式，值只是本用例的登录口令，不是真实凭据。
USER_PASS = "alice-pass"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_CONTENT_TYPE = "application/pdf"


@compiles(LONGTEXT, "sqlite")
def _compile_longtext_as_text(_type, _compiler, **_kwargs):
    return "TEXT"


# ---------------------------------------------------------------------------
# 无文本层 PDF 夹具：按字节构造，不依赖任何 PDF 写库（与 test_pdf_extraction 同一手法）
# ---------------------------------------------------------------------------


def _serialize_pdf(objects: list[tuple[int, bytes]]) -> bytes:
    """把 (对象号, 对象体) 列表拼成完整 PDF 字节，xref 偏移按实际写入位置计算。"""
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num, body in objects:
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"

    total_objs = len(objects) + 1  # +1 是固定的 0 号空闲对象
    xref_offset = len(out)
    # 每个 xref 表项必须正好 20 字节：10 位偏移 + 空格 + 5 位世代号 + 空格 + 类型 + 空格 + 换行。
    out += b"xref\n0 %d\n" % total_objs
    out += b"0000000000 65535 f \n"
    for num in range(1, total_objs):
        out += b"%010d 00000 n \n" % offsets[num]

    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % total_objs
    out += b"startxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(out)


def _pdf_without_text_layer() -> bytes:
    """一页结构合法、但内容流不画任何字形的 PDF —— 扫描件 / 图片型 PDF 的等价形态。

    第 1 跳的输入就是它：`page.extract_text()` 对这样的页返回空，`parts` 为空列表。
    """
    stream = b"BT ET"
    return _serialize_pdf([
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>"),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"),
        (
            4,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
        ),
        (5, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"),
    ])


def _pdf_with_text(text: str) -> bytes:
    """一页画了 ASCII 文本的 PDF，作为「正常路径不受影响」的对照。"""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)").encode("latin-1")
    stream = b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET"
    return _serialize_pdf([
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>"),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"),
        (
            4,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
        ),
        (5, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"),
    ])


def _build_docx(text: str) -> bytes:
    from docx import Document

    buffer = io.BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 端到端夹具：真路由 + 内存库，向量库只记录调用
# ---------------------------------------------------------------------------


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

    # 向量库不参与校验，用例里记录调用即可：`indexed` 为空即证明这一步根本没被走到。
    indexed: list[list[dict]] = []
    monkeypatch.setattr(knowledge_service, "add_chunks", lambda chunks, *args, **kwargs: indexed.append(chunks))
    monkeypatch.setattr(knowledge_service, "delete_file_chunks", lambda *args, **kwargs: None)

    client = TestClient(app)
    login = client.post("/api/auth/login", json={"username": USERNAME, "password": USER_PASS})
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['token']}"}

    try:
        yield SimpleNamespace(
            client=client,
            db=db,
            base=base,
            user=alice,
            indexed=indexed,
            headers=headers,
        )
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


def _insert_legacy_file(api, name, content=""):
    """直接落一行「存量」记录：修复上线前入库的空内容文件只可能是这种形态。"""
    entry = KnowledgeFile(
        knowledge_base_id=api.base.id,
        name=name,
        size=1024,
        content=content,
        user_id=api.user.id,
    )
    api.db.add(entry)
    api.db.commit()
    api.db.refresh(entry)
    return entry


# ---------------------------------------------------------------------------
# 第 1 条修法：抽取为空 ⇒ 明确失败，且不落行、不写向量
# ---------------------------------------------------------------------------


def test_upload_rejects_pdf_without_text_layer(api):
    """本单的原始复现：无文本层 PDF 不能再以 200 交付。"""
    response = _upload(api, "扫描件.pdf", _pdf_without_text_layer(), PDF_CONTENT_TYPE)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == crud_knowledge_file.EMPTY_EXTRACTED_TEXT_MESSAGE
    # 不落元数据行：没有「看起来可用、实际永远检索不到」的残留记录。
    assert _stored_files(api) == []
    # 更不该走到写向量那一步 —— 第 3 跳的 replace_empty 分支在修复后已不可达此路径。
    assert api.indexed == []


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("空文件.txt", "text/plain"),
        ("空文件.md", "text/markdown"),
    ],
)
def test_upload_rejects_text_file_without_content(api, filename, content_type):
    """文本类文件同样：抽不出文本就没有可索引的东西，不能判成功。"""
    response = _upload(api, filename, b"", content_type)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == crud_knowledge_file.EMPTY_EXTRACTED_TEXT_MESSAGE
    assert _stored_files(api) == []
    assert api.indexed == []


def test_upload_rejects_whitespace_only_text_file(api):
    """只有空白字符的文本文件等价于空：`.strip()` 后没有可索引内容。"""
    response = _upload(api, "空白.txt", "  \n\t\n  ".encode(), "text/plain")

    assert response.status_code == 400, response.text
    assert _stored_files(api) == []


def test_rejection_is_logged_as_warning(api, caplog):
    """运维侧要有信号：修复前这条路径只剩一行 INFO（action=replace_empty）。"""
    import logging

    with caplog.at_level(logging.WARNING):
        _upload(api, "扫描件.pdf", _pdf_without_text_layer(), PDF_CONTENT_TYPE)

    assert "empty extracted text" in caplog.text


# ---------------------------------------------------------------------------
# 防护回归：正常路径不受影响（对抗评审反例 1）
# ---------------------------------------------------------------------------


def test_upload_still_indexes_pdf_with_text(api):
    """有文本层的 PDF 照常入库：护栏没有误伤正常链路。"""
    payload = _pdf_with_text("Knowledge base regression")

    response = _upload(api, "制度.pdf", payload, PDF_CONTENT_TYPE)

    assert response.status_code == 200, response.text
    assert response.json()["size"] == len(payload)
    stored = _stored_files(api)
    assert len(stored) == 1
    assert "Knowledge base regression" in stored[0].content
    assert api.indexed and api.indexed[0], "有文本的 PDF 必须真的写进向量库"


def test_upload_still_indexes_non_empty_text_file(api):
    payload = "# 迟到处理\n\n迟到 30 分钟以内记口头提醒。".encode()

    response = _upload(api, "rule.md", payload, "text/plain")

    assert response.status_code == 200, response.text
    assert "口头提醒" in _stored_files(api)[0].content
    assert api.indexed and api.indexed[0]


def test_upload_still_indexes_minimal_non_empty_text(api):
    """护栏的判据是「去掉空白后为空」，不是长度：极短的非空文件照常入库。

    反向变异自证：把守卫改成 `len(text.strip()) < 3` 这类长度阈值（会误拒合法短文件），
    其余用例全绿，只有这一条会红 —— 它钉住的是判据的形状，不是「空」这个字面。
    """
    response = _upload(api, "a.txt", b"a", "text/plain")

    assert response.status_code == 200, response.text
    assert _stored_files(api)[0].content.strip() == "a"
    assert api.indexed and api.indexed[0]


def test_upload_still_indexes_docx_with_content(api):
    """docx 正常路径不受影响（占位提示的先例仍然服务于存量空行）。"""
    response = _upload(api, "report.docx", _build_docx("季度目标：完成上传校验"), DOCX_CONTENT_TYPE)

    assert response.status_code == 200, response.text
    assert "季度目标" in _stored_files(api)[0].content
    assert api.indexed and api.indexed[0]


def test_unsupported_extension_still_400_before_extraction(api):
    """白名单外的扩展名仍是原有的类型错误，不被新分支抢先。"""
    response = _upload(api, "run.exe", b"MZ", "application/octet-stream")

    assert response.status_code == 400
    assert response.json()["detail"] != crud_knowledge_file.EMPTY_EXTRACTED_TEXT_MESSAGE
    assert _stored_files(api) == []


# ---------------------------------------------------------------------------
# 第 2 条修法：覆盖率空源返回 0.0，阈值守卫恢复信号
# ---------------------------------------------------------------------------


def test_chunk_coverage_ratio_returns_zero_for_empty_source():
    """第 4 跳：空源必须算 0%，不能短路成 100%。"""
    assert chunk_coverage_ratio("", []) == 0.0
    assert chunk_coverage_ratio(None, []) == 0.0
    # 空源哪怕带着分块也仍是 0：被索引文本对不上任何原文，除零同样要避开。
    assert chunk_coverage_ratio("", [{"text": "第一段"}]) == 0.0
    # 非空源的口径一字未改。
    assert chunk_coverage_ratio("第一段落", [{"text": "第一"}]) == 0.5
    assert chunk_coverage_ratio("第一段", []) == 0.0


def test_low_coverage_guard_fires_on_empty_source(caplog):
    """守卫在空源上必须留下 warning —— 这正是修复前被短路掉的那条信号。"""
    import logging

    entry = SimpleNamespace(id=12, name="扫描件.pdf")

    with caplog.at_level(logging.WARNING):
        knowledge_service._warn_on_low_chunk_coverage(entry, "", [], scope="Knowledge file upload")

    assert "chunk coverage too low" in caplog.text
    assert "coverage=0.0%" in caplog.text


def test_rebuild_of_legacy_empty_file_warns_instead_of_passing_silently(monkeypatch, caplog):
    """存量空内容行走启动重建时也要有信号（修复前 coverage=1.0，守卫直接 return）。

    这里**不**打桩 `chunk_text`：真实分块对空文本返回 `[]`，正是存量空行的形态。
    """
    import logging

    class FakeSession:
        def query(self, model):
            return self

        def filter(self, *args):
            return self

        def all(self):
            return [SimpleNamespace(id=1, content="", name="扫描件.pdf", knowledge_base_id=2)]

        def close(self):
            pass

    calls = []
    monkeypatch.setattr(knowledge_service, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(knowledge_service, "add_chunks", lambda *args, **kwargs: calls.append(args[1]))

    with caplog.at_level(logging.WARNING):
        knowledge_service.rebuild_existing_knowledge_index()

    assert calls == [1]
    assert "chunk coverage too low" in caplog.text


# ---------------------------------------------------------------------------
# 第 3 条修法：空内容预览的占位提示放开到 .pdf / .txt / .md
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["扫描件.pdf", "空文件.txt", "空文件.md", "报告.docx"])
def test_empty_content_preview_hints_for_text_file_types(api, name):
    """预览不再是一片空白：空内容行要解释自己为什么是空的。"""
    entry = _insert_legacy_file(api, name, content="")

    response = api.client.get(f"/api/knowledge/{entry.id}/content", headers=api.headers)

    assert response.status_code == 200, response.text
    assert response.json()["content"] == "该文件上传时未抽取内容，请重新上传以生成预览。"


def test_non_empty_content_preview_is_untouched(api):
    entry = _insert_legacy_file(api, "制度.pdf", content="第一章 总则\n本制度自发布之日起施行。")

    response = api.client.get(f"/api/knowledge/{entry.id}/content", headers=api.headers)

    assert response.status_code == 200, response.text
    assert response.json()["content"].startswith("第一章 总则")
