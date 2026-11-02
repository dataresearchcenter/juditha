"""FTM `topics` flow: aggregator -> tantivy / automaton -> search, extract,
percolate, and across the gRPC wire."""

import pytest
from typer.testing import CliRunner

from juditha import io
from juditha.cli import cli
from juditha.extraction import AhoExtractor
from juditha.normalizer import name_key
from juditha.rpc.server import make_server
from juditha.store import Store, get_build_store, get_store, lookup

TEXT = (
    "Jane Doe Example met Offshore Holdings Example and "
    "Plain Organisation Example yesterday."
)


@pytest.fixture()
def topics_store(fixtures_path, store):
    io.load_proxies(fixtures_path / "topics.ftm.json", store)
    store.build()
    return store


def by_text(mentions):
    return {m.text: m for m in mentions}


def test_topics_aggregated(fixtures_path, store):
    io.load_proxies(fixtures_path / "topics.ftm.json", store)
    doc = store.aggregator.get_doc(name_key("Jane Doe Example"))
    assert doc is not None
    # two entities share the cluster: their topics are unioned
    assert doc.topics == {"role.pep", "sanction"}
    plain = store.aggregator.get_doc(name_key("Plain Organisation Example"))
    assert plain is not None
    assert plain.topics == set()


def test_topics_search(topics_store):
    res = topics_store.search("Jane Doe Example")
    assert res is not None
    assert res.topics == {"role.pep", "sanction"}
    assert res.model_dump()["topics"] == {"role.pep", "sanction"}
    res = topics_store.search("Offshore Holdings Example")
    assert res is not None
    assert res.topics == {"corp.offshore"}
    res = topics_store.search("Plain Organisation Example")
    assert res is not None
    assert res.topics == set()


def test_topics_search_filter(topics_store):
    q = "Jane Doe Example"
    assert topics_store.search(q, topics=("role.pep",)) is not None
    assert topics_store.search(q, topics=("sanction",)) is not None
    # any-of semantics, like schemata
    assert topics_store.search(q, topics=("corp.offshore", "sanction")) is not None
    assert topics_store.search(q, topics=("corp.offshore",)) is None
    # combines with the schemata filter
    assert topics_store.search(q, schemata=("Person",), topics=("role.pep",))
    assert topics_store.search(q, schemata=("Company",), topics=("role.pep",)) is None
    # clusters without topics never match a topics filter
    assert (
        topics_store.search("Plain Organisation Example", topics=("role.pep",)) is None
    )


def test_topics_lookup_filter(topics_store):
    get_store.cache_clear()
    assert lookup("Jane Doe Example", uri=topics_store.uri, topics=("role.pep",))
    assert lookup("Jane Doe Example", uri=topics_store.uri, topics=("crime",)) is None
    get_store.cache_clear()


def test_topics_cli_filters(fixtures_path, tmp_path, monkeypatch):
    get_store.cache_clear()
    get_build_store.cache_clear()
    lookup.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", str(tmp_path / "juditha"))
    runner = CliRunner()
    runner.invoke(cli, ["load-entities", "-i", str(fixtures_path / "topics.ftm.json")])
    runner.invoke(cli, ["build"])

    res = runner.invoke(cli, ["lookup", "Jane Doe Example", "--topics", "role.pep"])
    assert res.exit_code == 0
    assert "Jane Doe Example" in res.output
    # repeatable, any-of
    res = runner.invoke(
        cli,
        ["lookup", "Jane Doe Example", "--topics", "crime", "--topics", "sanction"],
    )
    assert res.exit_code == 0
    assert "Jane Doe Example" in res.output
    res = runner.invoke(cli, ["lookup", "Jane Doe Example", "--topics", "crime"])
    assert res.exit_code == 0
    assert "not found" in res.output
    res = runner.invoke(cli, ["lookup", "Jane Doe Example", "--schemata", "Company"])
    assert res.exit_code == 0
    assert "not found" in res.output
    res = runner.invoke(
        cli,
        ["lookup", "Jane Doe Example", "--schemata", "Person", "--topics", "role.pep"],
    )
    assert res.exit_code == 0
    assert "Jane Doe Example" in res.output
    get_store.cache_clear()
    get_build_store.cache_clear()


def test_topics_extract(topics_store):
    mentions = by_text(topics_store.extract(TEXT))
    assert mentions["Jane Doe Example"].topics == {"role.pep", "sanction"}
    assert mentions["Offshore Holdings Example"].topics == {"corp.offshore"}
    assert mentions["Plain Organisation Example"].topics == set()
    # JSON surface carries topics alongside the aliased `schema`
    data = mentions["Jane Doe Example"].model_dump(mode="json")
    assert data["schema"] == "Person"
    assert sorted(data["topics"]) == ["role.pep", "sanction"]


def test_topics_extract_alias(topics_store):
    # aliases carry the cluster's topics too
    mentions = by_text(topics_store.extract("Call Janie Example now."))
    assert mentions["Janie Example"].topics == {"role.pep", "sanction"}


def test_topics_automaton_roundtrip(tmp_path, topics_store):
    # topics survive save / load of the patterns file
    path = tmp_path / "automaton.txt"
    topics_store.extractor.save(path)
    ext = AhoExtractor()
    assert ext.load(path)
    assert by_text(ext.extract(TEXT)) == by_text(topics_store.extract(TEXT))


def test_topics_automaton_legacy_file(tmp_path):
    # a patterns file written before topics existed loads with empty topics
    path = tmp_path / "automaton.txt"
    path.write_text("jane doe example\tPerson\n")
    ext = AhoExtractor()
    assert ext.load(path)
    (m,) = ext.extract("Jane Doe Example")
    assert m.schema_ == "Person"
    assert m.topics == set()


def test_topics_percolate(topics_store):
    mentions = by_text(topics_store.percolate(TEXT))
    assert mentions["Jane Doe Example"].topics == {"role.pep", "sanction"}
    assert mentions["Offshore Holdings Example"].topics == {"corp.offshore"}
    assert mentions["Plain Organisation Example"].topics == set()


def test_topics_rpc(fixtures_path, tmp_path):
    uri = str(tmp_path / "juditha")
    build = get_build_store(uri)
    io.load_proxies(fixtures_path / "topics.ftm.json", build)
    build.build()
    local = Store(uri)
    server, port = make_server(local, "localhost", 0, 2)
    server.start()
    remote = get_store(f"grpc://localhost:{port}")
    try:
        res = remote.search("Jane Doe Example")
        assert res is not None
        assert res.topics == {"role.pep", "sanction"}
        assert res.model_dump(exclude={"took"}) == local.search(
            "Jane Doe Example"
        ).model_dump(exclude={"took"})
        assert remote.search("Jane Doe Example", topics=("role.pep",)) is not None
        assert remote.search("Jane Doe Example", topics=("crime",)) is None
        assert remote.extract(TEXT) == local.extract(TEXT)
        assert remote.percolate(TEXT) == local.percolate(TEXT)
        assert by_text(remote.extract(TEXT))["Jane Doe Example"].topics == {
            "role.pep",
            "sanction",
        }
    finally:
        remote.close()
        server.stop(0).wait()
        get_store.cache_clear()
        get_build_store.cache_clear()
