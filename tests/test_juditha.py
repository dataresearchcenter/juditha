from pathlib import Path

from ftmq.util import make_entity
from typer.testing import CliRunner

from juditha import io
from juditha.cli import cli
from juditha.store import get_store, lookup

runner = CliRunner()


def test_juditha_base(fixtures_path, store):
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    store.build()
    name = "European Parliament"
    res = lookup(name, uri=store.uri)
    assert res is not None
    assert res.score > 0.97
    assert res.caption == name
    assert res.names == {name}
    assert res.schemata == {"PublicBody"}
    assert res.common_schema == "PublicBody"

    res2 = lookup(name.lower(), uri=store.uri)
    assert res2 is not None
    assert res2.caption == name
    assert res2.query == name.lower()
    assert res2.score == res.score

    # fuzzy match
    res_fuzzy = lookup("European Parlament", uri=store.uri)
    assert res_fuzzy is not None
    assert res_fuzzy.score < 1

    # lower threshold
    assert lookup("European Parlment") is None
    res_fuzzy = lookup("European Parlment", threshold=0.5, uri=store.uri)
    assert res_fuzzy is not None
    assert 0.5 < res_fuzzy.score < 1

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
    jane = lookup("Jane Doe", uri=store.uri)
    assert jane is not None
    assert "mt" in jane.countries
    assert "jani" in jane.aliases

    assert lookup("Jane Doe", schemata=("Person",), uri=store.uri) is not None
    assert lookup("Jane Doe", schemata=("Company",), uri=store.uri) is None
    assert lookup("Jani Doe", uri=store.uri) is None

    # fuzzy jane
    res = lookup("Jane Dae", uri=store.uri, threshold=0)
    assert res is not None
    assert res.names == {"Jane Doe"}
    assert res.score < 1


def test_juditha_cli(monkeypatch, fixtures_path: Path, tmp_path):
    # Need to clear cache to pick up new env var
    get_store.cache_clear()
    # Set env var
    monkeypatch.setenv("JUDITHA_URI", tmp_path)

    # Run CLI commands
    # Note: Each invoke creates/reuses a cached Store with the same LevelDB connection
    runner.invoke(cli, ["load-names", "-i", str(fixtures_path / "names.txt")])
    runner.invoke(cli, ["build"])

    res = runner.invoke(cli, ["lookup", "Jane Doe"])
    assert res.exit_code == 0
    assert "Jane Doe" in res.output

    res = runner.invoke(cli, ["lookup", "doe, jane"])
    assert res.exit_code == 0
    assert "not found" in res.output
    res = runner.invoke(cli, ["lookup", "doe, jane", "--threshold", "0.1"])
    assert res.exit_code == 0

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
