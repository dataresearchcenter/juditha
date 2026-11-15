"""GeoNames loader: dump rows -> `Address` entities -> aggregator / index /
automaton -> lookup, extract, percolate, CLI and gRPC."""

import zipfile

import pytest
from typer.testing import CliRunner

from juditha import geonames, io
from juditha.cli import cli
from juditha.normalizer import name_key
from juditha.rpc.server import make_server
from juditha.store import Store, get_build_store, get_store, lookup

TEXT = "I just visited Sant Julia de loria last week, then Berlin."


@pytest.fixture()
def geonames_store(fixtures_path, store):
    io.load_geonames(fixtures_path / "geonames.txt", store)
    store.build()
    return store


def by_text(mentions):
    return {m.text: m for m in mentions}


def test_geonames_rows(fixtures_path):
    rows = {
        r["geonameid"]: r for r in geonames.iter_rows(fixtures_path / "geonames.txt")
    }
    # A rows need 4 chars, P rows 8, everything needs population >= 5000
    assert set(rows) == {
        "3039162",
        "3039163",
        "2925533",
        "2950157",
        "2950159",  # "Berlin" city row: 6 chars < 8, but 3.4M people
        "2911298",  # "Hamburg" city row: 7 chars < 8, but 1.9M people
        "6547539",
        "9000002",
        "6448311",  # "Auch" FR, ADM4: 4 chars, 22k people, passes the row floors
        "2921044",  # "Federal Republic of Germany", PCLI
    }
    assert "9000001" not in rows  # "Rolle": 5 chars < 8, 6000 people
    assert "9000003" not in rows  # population 120
    assert "2891547" not in rows  # "Kern": population 0


def test_geonames_variants(fixtures_path):
    rows = {
        r["geonameid"]: r for r in geonames.iter_rows(fixtures_path / "geonames.txt")
    }
    variants = geonames.select_variants(rows["9000002"], min_alternate_length=6)
    # canonical first; "Kern" is a substring token, "12345" numeric, "Kerny"
    # too short, "Kern Co" and the accented alternate survive
    assert variants == ["Kern County", "Kern Co", "Comté de Kern"]
    variants = geonames.select_variants(rows["2950157"], min_alternate_length=6)
    assert variants[0] == "Land Berlin"
    assert "Berlin" not in variants  # substring of "land berlin"
    assert "Berlino" in variants
    # countries are named formally and the bare name has no row of its own,
    # so the substring rule is waived for PCL* rows
    variants = geonames.select_variants(rows["2921044"], min_alternate_length=6)
    assert variants[0] == "Federal Republic of Germany"
    assert "Germany" in variants
    assert "Deutschland" in variants
    # the alternate floor is a setting
    variants = geonames.select_variants(rows["9000002"], min_alternate_length=4)
    assert "Kerny" in variants
    assert "Kern" not in variants


def test_geonames_dictionary_words(fixtures_path):
    rows = {
        r["geonameid"]: r for r in geonames.iter_rows(fixtures_path / "geonames.txt")
    }
    assert geonames.is_dictionary_word("auch")
    assert geonames.is_dictionary_word("berlin")
    assert not geonames.is_dictionary_word("katharina")
    # a small place whose name is a common word is dropped entirely ...
    assert geonames.select_variants(rows["6448311"], min_alternate_length=6) == []
    # ... a big one is kept, name and alternates alike
    variants = geonames.select_variants(rows["2950159"], min_alternate_length=6)
    assert variants[0] == "Berlin"
    assert "Berlino" in variants
    # hyphenated tokens are not checked
    assert geonames._is_plain_token("auch") and not geonames._is_plain_token(
        "saint-laurent"
    )
    assert "GEO:6448311" not in {
        e.id for e in geonames.iter_entities(fixtures_path / "geonames.txt")
    }


def test_geonames_entities(fixtures_path):
    entities = {e.id: e for e in geonames.iter_entities(fixtures_path / "geonames.txt")}
    e = entities["GEO:3039163"]
    assert e.schema.name == "Address"
    assert e.get("name") == ["Sant Julià de Lòria"]
    # asciiname dedupes against the name on the normalized form
    assert "Sant Julia de Loria" not in e.get("alias")
    assert "San Julia" in e.get("alias")
    assert e.get("country") == ["ad"]


def test_geonames_zip(fixtures_path, tmp_path):
    path = tmp_path / "XX.zip"
    with zipfile.ZipFile(path, "w") as zh:
        # real dumps carry a readme next to the data file
        zh.writestr("readme.txt", "This file is not the data.")
        zh.write(fixtures_path / "geonames.txt", "XX.txt")
    entity_ids = {e.id for e in geonames.iter_entities(path)}
    assert entity_ids == {
        e.id for e in geonames.iter_entities(fixtures_path / "geonames.txt")
    }
    assert "GEO:3039162" in entity_ids


def test_geonames_aggregator(geonames_store):
    doc = geonames_store.aggregator.get_doc(name_key("Sant Julià de Lòria"))
    assert doc is not None
    assert doc.schemata == {"Address"}
    assert doc.countries == {"ad"}
    assert "San Julia" in doc.aliases
    # single-token place survives the aggregator caption floor; the city row
    # and the ADM4 row cluster under the same key
    doc = geonames_store.aggregator.get_doc("berlin")
    assert doc is not None
    assert doc.names == {"Berlin"}
    assert "Stadt Berlin" in doc.aliases


def test_geonames_search(geonames_store):
    res = geonames_store.search("Sant Julia de Loria")
    assert res is not None
    assert res.common_schema == "Address"
    assert res.countries == {"ad"}
    res = geonames_store.search("Berlin")
    assert res is not None
    assert res.caption == "Berlin"
    assert geonames_store.search("Berlin", schemata=("Person",)) is None
    # alias matches: the query is the name_key of an alternate, not of the
    # name, so neither the key nor the surface forms are within fuzzy reach
    res = geonames_store.search("Germany")
    assert res is not None
    assert res.caption == "Federal Republic of Germany"
    assert res.score == 1
    res = geonames_store.search("Deutschland")
    assert res is not None
    assert res.caption == "Federal Republic of Germany"
    res = geonames_store.search("Berlino")
    assert res is not None
    assert "Berlino" in res.aliases
    assert res.score == 1


def test_geonames_lookup(geonames_store):
    get_store.cache_clear()
    res = lookup("Sant Julia de Loria", uri=geonames_store.uri)
    assert res is not None
    assert res.common_schema == "Address"
    get_store.cache_clear()
    get_build_store.cache_clear()


def test_geonames_extract(geonames_store):
    mentions = by_text(geonames_store.extract(TEXT))
    m = mentions["Sant Julia de loria"]
    assert m.schema_ == "Address"
    assert m.topics == set()
    # single-token place
    m = mentions["Berlin"]
    assert m.schema_ == "Address"
    assert m.model_dump(mode="json")["schema"] == "Address"
    mentions = by_text(geonames_store.extract("Flights to Frankfurt am Main"))
    assert mentions["Frankfurt am Main"].schema_ == "Address"
    # big city below the raw name floor
    mentions = by_text(geonames_store.extract("Hamburg is a port."))
    assert mentions["Hamburg"].schema_ == "Address"
    # a country's bare name, only an alternate of its formal name
    mentions = by_text(geonames_store.extract("Made in Germany."))
    assert mentions["Germany"].schema_ == "Address"
    # adjacent single-token places all match
    mentions = geonames_store.extract("Berlin, Hamburg, Berlin.")
    assert [m.text for m in mentions] == ["Berlin", "Hamburg", "Berlin"]
    # dropped rows never match, nor does a dictionary-word place
    assert geonames_store.extract("Rolle and Tinyville Hamlet and Kern, auch.") == []


def test_geonames_percolate(geonames_store):
    mentions = by_text(geonames_store.percolate(TEXT))
    m = mentions["Sant Julia de loria"]
    assert m.schema_ == "Address"
    # single-token names stay below the percolator floor
    assert "Berlin" not in mentions


def test_geonames_cli(fixtures_path, tmp_path, monkeypatch):
    get_store.cache_clear()
    get_build_store.cache_clear()
    lookup.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", str(tmp_path / "juditha"))
    runner = CliRunner()
    res = runner.invoke(
        cli, ["load-geonames", "-i", str(fixtures_path / "geonames.txt")]
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(cli, ["build"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(cli, ["lookup", "Berlin", "--schemata", "Address"])
    assert res.exit_code == 0
    assert "'Address'" in res.output
    doc = tmp_path / "doc.txt"
    doc.write_text(TEXT)
    res = runner.invoke(cli, ["extract", "-i", str(doc)])
    assert res.exit_code == 0
    assert '"schema":"Address"' in res.output
    assert "Sant Julia de loria" in res.output
    get_store.cache_clear()
    get_build_store.cache_clear()


def test_geonames_rpc(fixtures_path, tmp_path):
    uri = str(tmp_path / "juditha")
    build = get_build_store(uri)
    io.load_geonames(fixtures_path / "geonames.txt", build)
    build.build()
    local = Store(uri)
    server, port = make_server(local, "localhost", 0, 2)
    server.start()
    remote = get_store(f"grpc://localhost:{port}")
    try:
        res = remote.search("Berlin")
        assert res is not None
        assert res.common_schema == "Address"
        assert res.model_dump(exclude={"took"}) == local.search("Berlin").model_dump(
            exclude={"took"}
        )
        assert remote.extract(TEXT) == local.extract(TEXT)
        assert remote.percolate(TEXT) == local.percolate(TEXT)
        assert by_text(remote.extract(TEXT))["Berlin"].schema_ == "Address"
    finally:
        remote.close()
        server.stop(0).wait()
        get_store.cache_clear()
        get_build_store.cache_clear()
