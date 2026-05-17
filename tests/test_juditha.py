from pathlib import Path

import pytest
from ftmq.util import make_entity
from typer.testing import CliRunner

from juditha import io
from juditha.cli import cli
from juditha.store import get_store, lookup

runner = CliRunner()


@pytest.fixture()
def loaded_store(fixtures_path, tmp_path, monkeypatch):
    """Store with EU authorities loaded and built."""
    get_store.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", str(tmp_path / "juditha"))
    store = get_store()
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    store.build()
    return store


def test_lookup_exact(loaded_store):
    name = "European Parliament"
    res = loaded_store.search(name)
    assert res is not None
    assert res.score > 0.97
    assert res.query == name
    assert name in res.names
    assert "PublicBody" in res.schemata


def test_lookup_case_insensitive(loaded_store):
    name = "European Parliament"
    res = loaded_store.search(name.lower())
    assert res is not None
    assert res.query == name.lower()


def test_lookup_fuzzy(loaded_store):
    res = loaded_store.search("European Parlament")
    assert res is not None
    assert res.score < 1


def test_lookup_low_threshold(loaded_store):
    assert loaded_store.search("European Parlment") is None
    res = loaded_store.search("European Parlment", threshold=0.5)
    assert res is not None
    assert 0.5 < res.score < 1


def test_lookup_not_found(loaded_store):
    assert loaded_store.search("xyzzyplugh gibberish") is None


def test_extract(loaded_store):
    mentions = loaded_store.extract("The European Parliament met today.")
    texts = [m.text for m in mentions]
    assert "European Parliament" in texts


# --- Aggregator-access tests ---


def test_lookup_with_entity(fixtures_path, store):
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    jane = make_entity(
        {
            "id": "j",
            "schema": "Person",
            "properties": {
                "name": ["Jane Doe"],
                "nationality": ["mt"],
                "alias": ["jani"],
            },
        }
    )
    store.aggregator.put(jane)
    store.aggregator.flush()
    store.build()

    res = store.search("Jane Doe")
    assert res is not None
    assert "mt" in res.countries
    assert "jani" in res.aliases

    assert store.search("Jane Doe", schemata=("Person",)) is not None
    assert store.search("Jane Doe", schemata=("Company",)) is None

    # fuzzy jane
    res = store.search("Jane Dae", threshold=0)
    assert res is not None
    assert res.names == {"Jane Doe"}
    assert res.score < 1


def test_juditha_cli(monkeypatch, fixtures_path: Path, tmp_path):
    # Need to clear cache to pick up new env var
    get_store.cache_clear()
    lookup.cache_clear()
    # Set env var
    monkeypatch.setenv("JUDITHA_URI", tmp_path)

    # Run CLI commands
    # Note: Each invoke creates/reuses a cached Store with the same LevelDB connection
    runner.invoke(cli, ["load-names", "-i", str(fixtures_path / "names.txt")])
    runner.invoke(cli, ["build"])

    res = runner.invoke(cli, ["lookup", "Jane Doe"])
    assert res.exit_code == 0
    assert "Jane Doe" in res.output

    # name_key is order-independent, so "doe, jane" also matches "Jane Doe"
    res = runner.invoke(cli, ["lookup", "doe, jane"])
    assert res.exit_code == 0
    assert "Jane Doe" in res.output

    res = runner.invoke(
        cli,
        ["load-entities", "-i", str(fixtures_path / "eu_authorities.ftm.json")],
    )
    assert res.exit_code == 0
    res = runner.invoke(cli, ["build"])
    assert res.exit_code == 0


def test_store_env(monkeypatch, tmp_path):
    # Need to clear cache to pick up new env var
    get_store.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", tmp_path)
    store = get_store()
    assert store.uri == str(tmp_path)
