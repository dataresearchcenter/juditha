import pytest

from juditha import io
from juditha.rpc.client import ApiStore
from juditha.rpc.server import make_server
from juditha.store import Store, get_build_store, get_store

TEXT = "The European Parliament met the European Commission today."


@pytest.fixture()
def served(fixtures_path, tmp_path):
    """A built store, a gRPC server on an ephemeral port, and a client for it.

    Yields `(local, remote)` so every test can assert wire parity against the
    same corpus. Port 0 lets the OS pick, so parallel runs can't collide.
    """
    uri = str(tmp_path / "juditha")
    build = get_build_store(uri)
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", build)
    build.build()

    local = Store(uri)
    server, port = make_server(local, "localhost", 0, 2)
    server.start()
    remote = get_store(f"grpc://localhost:{port}")
    try:
        yield local, remote
    finally:
        remote.close()
        server.stop(0).wait()
        get_store.cache_clear()
        get_build_store.cache_clear()


def unpack(result):
    """Everything but `took`, which is a server-side timing and never matches."""
    return result.model_dump(exclude={"took"})


def test_rpc_search_parity(served):
    local, remote = served
    name = "European Parliament"
    res = remote.search(name)
    assert res is not None
    assert res.query == name
    assert name in res.names
    assert "PublicBody" in res.schemata
    assert res.common_schema == "PublicBody"
    assert res.caption == name
    assert unpack(res) == unpack(local.search(name))


def test_rpc_search_not_found(served):
    _, remote = served
    assert remote.search("xyzzyplugh gibberish") is None


def test_rpc_search_threshold(served):
    local, remote = served
    # unset optional fields mean "use the server's Settings default"
    assert remote.search("European Parlment") is None
    res = remote.search("European Parlment", threshold=0.5)
    assert res is not None
    assert 0.5 < res.score < 1
    assert unpack(res) == unpack(local.search("European Parlment", threshold=0.5))


def test_rpc_search_schemata(served):
    _, remote = served
    assert remote.search("European Parliament", schemata=("PublicBody",)) is not None
    assert remote.search("European Parliament", schemata=("Person",)) is None


def test_rpc_search_limit(served):
    local, remote = served
    res = remote.search("European Parliament", limit=1)
    assert res is not None
    assert unpack(res) == unpack(local.search("European Parliament", limit=1))


def test_rpc_extract_parity(served):
    local, remote = served
    mentions = remote.extract(TEXT)
    assert "European Parliament" in [m.text for m in mentions]
    assert mentions == local.extract(TEXT)
    # schema_ survives the pydantic alias round-trip
    assert {m.schema_ for m in mentions} == {"PublicBody"}


def test_rpc_extract_no_match(served):
    _, remote = served
    assert remote.extract("Nothing to see here.") == []


def test_rpc_percolate_parity(served):
    local, remote = served
    mentions = remote.percolate(TEXT)
    assert "European Parliament" in [m.text for m in mentions]
    assert mentions == local.percolate(TEXT)


def test_rpc_percolate_slop(served):
    local, remote = served
    text = "The European (sic) Parliament met today."
    assert remote.percolate(text) == local.percolate(text)
    assert remote.percolate(text, slop=1) == local.percolate(text, slop=1)


def test_rpc_get_store_dispatch(served):
    _, remote = served
    assert isinstance(remote, ApiStore)
    assert remote.uri.startswith("grpc://")
    assert remote.target == remote.uri.removeprefix("grpc://")


def test_rpc_build_store_rejects_remote():
    with pytest.raises(ValueError):
        get_build_store("grpc://localhost:50051")
