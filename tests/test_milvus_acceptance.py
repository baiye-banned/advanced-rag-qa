"""Milvus acceptance goal (docs/MAINTENANCE_GOAL_CLOSURE.md, goal 2).

Unlike tests/test_milvus_client.py (which drives a FakeClient), this file exercises a
real Milvus Lite database created in a temporary directory: upload -> query -> delete
round trips, knowledge-base isolation, and the index rebuild path.

Only the embedding function is replaced, with a deterministic offline implementation,
so the suite never touches the network and identical text always produces identical
vectors.

Milvus Lite serves the temporary database on a loopback port, which the gRPC client
behind pymilvus would otherwise route through HTTP_PROXY/HTTPS_PROXY when the process
has one configured; every test removes those variables for its own duration (see
`isolate_from_environment_proxy`), so the module behaves the same with and without a
proxy in the environment.

The suite is offline by default. Two opt-in variables point it at a deployed server for
issue #178 acceptance:

- ``MILVUS_ACCEPTANCE_URI`` (e.g. ``http://127.0.0.1:19530``) uses that server instead of
  the temporary Lite file. The instance must be a **disposable test instance** -- the
  tests write and delete rows in it. Never set this for production or shared data.
- ``MILVUS_ACCEPTANCE_LIVE_EMBEDDING=1`` keeps the project's real embedding provider
  instead of the deterministic offline bigram embedding.
"""

import hashlib
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from crud import knowledge_file as crud_knowledge_file
from rag import milvus_client
from service import knowledge_service

ROOT = Path(__file__).resolve().parents[1]
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")
PROXY_ENV_NAMES = {name.lower() for name in PROXY_VARIABLES + NO_PROXY_VARIABLES}
LOOPBACK_NO_PROXY = "localhost,127.0.0.1,::1"
DEAD_PROXY = "http://127.0.0.1:9"
ACCEPTANCE_URI_ENV = "MILVUS_ACCEPTANCE_URI"
LIVE_EMBEDDING_ENV = "MILVUS_ACCEPTANCE_LIVE_EMBEDDING"

CHUNKS = [
    {"id": "0", "text": "员工迟到一次罚款50元，当月累计三次以上加倍处罚。"},
    {"id": "1", "text": "迟到超过30分钟视为旷工半天。"},
    {"id": "2", "text": "报销流程需要在费用发生后30天内提交。"},
]
# Every file_id this module writes, so a server-backed run can start from the same state
# as a fresh Lite database without dropping a collection it does not own.
ACCEPTANCE_FILE_IDS = (1101, 1201, 1301, 1401, 1501, 1502, 1601, 1701)


def _bigram_embedding(texts):
    """Deterministic offline embedding: unit vectors of character-bigram counts.

    Identical text yields identical vectors, so a search for a stored chunk returns it
    with cosine similarity 1, and texts sharing bigrams still rank above unrelated ones.
    """
    vectors = []
    for text in texts:
        value = str(text)
        vector = [0.0] * milvus_client.EMBEDDING_DIM
        for index in range(max(len(value) - 1, 1)):
            gram = value[index : index + 2] or value
            slot = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16) % milvus_client.EMBEDDING_DIM
            vector[slot] += 1.0
        norm = math.sqrt(sum(component * component for component in vector)) or 1.0
        vectors.append([component / norm for component in vector])
    return vectors


class _StrongConsistencyClient:
    """Serve reads with Strong consistency when pointed at a deployed Milvus.

    Milvus Lite answers a read immediately after the write in the same process; a
    deployed server answers search/query with Bounded consistency by default, so a
    chunk inserted or deleted a moment earlier can still be missing from the next read.
    The acceptance flows below assert read-after-write, so only the read methods are
    overridden to ask for Strong consistency; every other call is forwarded to the real
    client.
    """

    def __init__(self, client):
        self._client = client

    def search(self, *args, **kwargs):
        kwargs.setdefault("consistency_level", "Strong")
        return self._client.search(*args, **kwargs)

    def query(self, *args, **kwargs):
        kwargs.setdefault("consistency_level", "Strong")
        return self._client.query(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _remove_acceptance_rows(client) -> None:
    """Delete the rows this module owns, leaving every other row in the collection.

    A server keeps its data between runs, so without this the previous run's rows would
    make the "empty store" and "unfiltered query" assertions fail on a second run. Only
    the file_ids above are touched: the collection (and anything else in it) stays.
    """
    if not client.has_collection(milvus_client.COLLECTION_NAME):
        return
    client.delete(
        collection_name=milvus_client.COLLECTION_NAME,
        filter=f"file_id in {list(ACCEPTANCE_FILE_IDS)}",
    )


def _release_lite_server(uri: str) -> None:
    """Milvus Lite serves a file-backed database from a background thread.

    The file handle stays open until the server for that path is released, which
    otherwise makes temporary-directory cleanup fail on Windows.
    """
    try:
        from milvus_lite.server_manager import server_manager_instance
    except ImportError:  # pragma: no cover - milvus-lite is a declared dependency
        return
    try:
        server_manager_instance.release_server(uri)
    except Exception:  # pragma: no cover - release is best effort cleanup
        pass


@pytest.fixture(autouse=True)
def isolate_from_environment_proxy(monkeypatch):
    """Keep a globally configured proxy from hijacking the loopback connection.

    gRPC resolves proxies from the process environment and applies them even to
    127.0.0.1, so with a dead proxy configured every connection here is sent to that
    proxy and fails after a ~10 s channel timeout instead of reaching Milvus Lite.
    Removing the proxy variables (and excluding loopback via NO_PROXY as a second line of
    defence) makes the tests pass at local-connection speed in both environments.

    Function-scoped on purpose: pymilvus creates its channel lazily inside the test body,
    so the environment only has to be clean for the duration of each test.
    """
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name in NO_PROXY_VARIABLES:
        monkeypatch.setenv(name, LOOPBACK_NO_PROXY)


@pytest.fixture
def acceptance_store(tmp_path, monkeypatch):
    """Point the module at a disposable Milvus instance and restore it afterwards.

    Without ``MILVUS_ACCEPTANCE_URI`` this is a throwaway Milvus Lite database under
    ``tmp_path`` (fully offline). With it, the module talks to that server instead and
    only closes the client on teardown: the instance is assumed to be a one-off test
    instance, so neither its collection nor its rows are dropped wholesale. Before each
    test the rows this module wrote to certain file_ids are deleted (see
    ``_remove_acceptance_rows``), so the suite can be re-run against the same server.

    ``MILVUS_ACCEPTANCE_LIVE_EMBEDDING=1`` leaves the project's real embedding provider
    in place; otherwise the deterministic offline bigram embedding is installed.
    """
    saved = (milvus_client.MILVUS_URI, milvus_client._client, milvus_client._embedding_fn)
    server_uri = os.environ.get(ACCEPTANCE_URI_ENV, "").strip()
    if server_uri:
        milvus_client.MILVUS_URI = server_uri
        milvus_client._client = None
        # The client is still created lazily by the module's first call, but through a
        # wrapper that asks for Strong consistency on reads: unlike Lite, a server may
        # not show a row inserted or deleted a moment ago.
        client_class = milvus_client.MilvusClient
        monkeypatch.setattr(
            milvus_client,
            "MilvusClient",
            lambda **kwargs: _StrongConsistencyClient(client_class(**kwargs)),
        )
        # Without the live opt-in, install the deterministic offline embedding; with it,
        # leave the project's real embedding provider in place.
        if os.environ.get(LIVE_EMBEDDING_ENV, "").strip() != "1":
            milvus_client._embedding_fn = _bigram_embedding
        try:
            _remove_acceptance_rows(milvus_client._connect_milvus())
            yield server_uri
        finally:
            client, milvus_client._client = milvus_client._client, None
            milvus_client.MILVUS_URI, _, milvus_client._embedding_fn = saved
            if client is not None:
                client.close()
        return

    uri = str(tmp_path / "milvus_acceptance.db")
    milvus_client.MILVUS_URI = uri
    milvus_client._client = None
    milvus_client._embedding_fn = _bigram_embedding
    try:
        yield uri
    finally:
        client, milvus_client._client = milvus_client._client, None
        milvus_client.MILVUS_URI, _, milvus_client._embedding_fn = saved
        if client is not None:
            client.close()
        _release_lite_server(uri)
        shutil.rmtree(str(tmp_path), ignore_errors=True)


@pytest.fixture
def lite_store(acceptance_store):
    """Backward-compatible alias for :func:`acceptance_store`."""
    return acceptance_store


def _count_rows(filter_expr: str) -> int:
    rows = milvus_client._connect_milvus().query(
        collection_name=milvus_client.COLLECTION_NAME,
        filter=filter_expr,
        output_fields=["count(*)"],
    )
    return int(rows[0]["count(*)"]) if rows else 0


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def filter_by(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def query(self, _model):
        return _FakeQuery(self._rows)

    def close(self):
        self.closed = True


class _KnowledgeFileRow:
    def __init__(self, file_id, name, content, knowledge_base_id):
        self.id = file_id
        self.name = name
        self.content = content
        self.knowledge_base_id = knowledge_base_id


def test_acceptance_upload_query_delete_round_trip(acceptance_store):
    milvus_client.add_chunks(CHUNKS, file_id=1101, file_name="考勤制度.txt", knowledge_base_id=11)

    hits = milvus_client.query_vectors("迟到超过30分钟视为旷工半天。", top_k=5, knowledge_base_id=11)

    assert [hit["chunk_id"] for hit in hits][0] == "1"
    # milvus-lite 3.2 reports the raw COSINE score in this field (higher = nearer, 1.0
    # for an identical vector) where the 3.0 line reported the 1 - similarity distance,
    # so an exact match scores 1.0 rather than 0.0. Ordering is unaffected: the same
    # chunk still ranks first and still carries the highest score of the three.
    assert len(hits) == 3
    assert hits[0]["distance"] == pytest.approx(1.0, abs=1e-6)
    assert all(hit["distance"] < hits[0]["distance"] for hit in hits[1:])
    assert hits[0]["content"] == "迟到超过30分钟视为旷工半天。"
    assert hits[0]["file_name"] == "考勤制度.txt"
    assert hits[0]["file_id"] == 1101
    assert hits[0]["route"] == "vector"
    assert set(hits[0]) == {"id", "chunk_id", "content", "file_name", "file_id", "route", "distance"}
    assert _count_rows("file_id == 1101") == 3

    milvus_client.delete_file_chunks(1101)

    assert _count_rows("file_id == 1101") == 0
    assert milvus_client.query_vectors("迟到超过30分钟视为旷工半天。", top_k=5, knowledge_base_id=11) == []


def test_acceptance_knowledge_base_isolation(acceptance_store):
    milvus_client.add_chunks(
        [{"id": "0", "text": "A库专属条款：迟到一次罚款50元。"}],
        file_id=1201,
        file_name="A考勤.txt",
        knowledge_base_id=12,
    )
    milvus_client.add_chunks(
        [{"id": "0", "text": "B库专属条款：报销必须附发票。"}],
        file_id=1301,
        file_name="B报销.txt",
        knowledge_base_id=13,
    )

    hits_a = milvus_client.query_vectors("A库专属条款 迟到 罚款", top_k=10, knowledge_base_id=12)
    hits_b = milvus_client.query_vectors("B库专属条款 报销 发票", top_k=10, knowledge_base_id=13)

    assert {hit["file_id"] for hit in hits_a} == {1201}
    assert {hit["file_id"] for hit in hits_b} == {1301}
    assert all("B库专属" not in hit["content"] for hit in hits_a)
    assert all("A库专属" not in hit["content"] for hit in hits_b)

    unfiltered = milvus_client.query_vectors("迟到 报销", top_k=10)
    assert {hit["file_id"] for hit in unfiltered} == {1201, 1301}


def test_acceptance_reupload_replaces_previous_chunks(acceptance_store):
    milvus_client.add_chunks(CHUNKS, file_id=1401, file_name="考勤制度.txt", knowledge_base_id=14)
    assert _count_rows("file_id == 1401") == 3

    milvus_client.add_chunks(
        [{"id": "0", "text": "更新后的制度：迟到罚款200元。"}],
        file_id=1401,
        file_name="考勤制度.txt",
        knowledge_base_id=14,
    )

    assert _count_rows("file_id == 1401") == 1
    hits = milvus_client.query_vectors("迟到超过30分钟视为旷工半天。", top_k=5, knowledge_base_id=14)
    assert [hit["content"] for hit in hits] == ["更新后的制度：迟到罚款200元。"]
    assert hits[0]["id"] == "1401_0"


def test_acceptance_delete_removes_only_target_file(acceptance_store):
    milvus_client.add_chunks(
        [{"id": "0", "text": "文件一：迟到罚款50元。"}], file_id=1501, file_name="一.txt", knowledge_base_id=15
    )
    milvus_client.add_chunks(
        [{"id": "0", "text": "文件二：迟到罚款200元。"}], file_id=1502, file_name="二.txt", knowledge_base_id=15
    )

    milvus_client.delete_file_chunks(1501)

    assert _count_rows("file_id == 1501") == 0
    assert _count_rows("knowledge_base_id == 15") == 1
    hits = milvus_client.query_vectors("文件二：迟到罚款200元。", top_k=5, knowledge_base_id=15)
    assert [hit["file_id"] for hit in hits] == [1502]


def test_acceptance_index_rebuild_restores_recall_without_duplicates(acceptance_store, monkeypatch):
    # Plain body sentences only: lines like "第1条：..." are parsed as headings and would
    # collapse back into a single chunk instead of producing a multi-chunk document.
    sentence = "员工迟到一次罚款{amount}元，当月累计三次以上加倍处罚，并记入绩效档案。"
    content = "\n\n".join(sentence.format(amount=index * 10) * 2 for index in range(1, 25))
    row = _KnowledgeFileRow(1601, "考勤制度.txt", content, 16)
    session = _FakeSession([row])
    monkeypatch.setattr(knowledge_service, "SessionLocal", lambda: session)

    knowledge_service.rebuild_existing_knowledge_index()

    expected_chunks = crud_knowledge_file.chunk_text(content, 1601)
    assert len(expected_chunks) > 1
    assert _count_rows("knowledge_base_id == 16") == len(expected_chunks)
    assert session.closed is True
    query = "员工迟到一次罚款50元，当月累计三次以上加倍处罚"
    first_recall = [
        hit["chunk_id"] for hit in milvus_client.query_vectors(query, top_k=20, knowledge_base_id=16)
    ]
    assert first_recall

    milvus_client.delete_file_chunks(1601)
    assert _count_rows("knowledge_base_id == 16") == 0

    knowledge_service.rebuild_existing_knowledge_index()

    assert _count_rows("knowledge_base_id == 16") == len(expected_chunks)
    second_recall = [
        hit["chunk_id"] for hit in milvus_client.query_vectors(query, top_k=20, knowledge_base_id=16)
    ]
    assert second_recall == first_recall


def test_acceptance_query_window_and_missing_collection(acceptance_store):
    # No collection yet: the query must report an empty result, not bootstrap the store.
    assert milvus_client.query_vectors("迟到", top_k=5, knowledge_base_id=17) == []

    milvus_client.add_chunks(CHUNKS, file_id=1701, file_name="考勤制度.txt", knowledge_base_id=17)

    assert len(milvus_client.query_vectors("迟到", top_k=2, knowledge_base_id=17)) == 2
    assert len(milvus_client.query_vectors("迟到", top_k=0, knowledge_base_id=17)) == 1
    assert len(milvus_client.query_vectors("迟到", top_k=99, knowledge_base_id=17)) == 3
    assert milvus_client.query_vectors("   ", top_k=5, knowledge_base_id=17) == []
    assert milvus_client.query_vectors("迟到", top_k=5, knowledge_base_id=99) == []


def test_acceptance_round_trip_survives_a_dead_proxy_environment():
    """Re-run one case in a child process whose proxy variables point at a dead port.

    Without the isolation above, gRPC sends the loopback connection to that proxy and the
    child fails after a ~10 s channel timeout; the whole module then reports failures in
    an environment whose only peculiarity is a proxy setting. The run must stay green.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in PROXY_ENV_NAMES and key != "PYTEST_ADDOPTS"
    }
    env.update({name: DEAD_PROXY for name in PROXY_VARIABLES})
    # Keep the opt-in target in the child too: the re-run must exercise the same store as
    # the parent (the env copy already carries them, this states the dependency).
    for name in (ACCEPTANCE_URI_ENV, LIVE_EMBEDDING_ENV):
        if name in os.environ:
            env[name] = os.environ[name]
    node = "tests/test_milvus_acceptance.py::test_acceptance_upload_query_delete_round_trip"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        # The child writes UTF-8 (PYTHONIOENCODING), while text=True would decode with the
        # Windows locale (GBK) and lose the failure output on the first Chinese character.
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
