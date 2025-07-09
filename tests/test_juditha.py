from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from juditha import io, lookup
from juditha.api import app
from juditha.cli import cli
from juditha.store import get_store

runner = CliRunner()


def test_io(fixtures_path, store):
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

    assert lookup("European", uri=store.uri) is None
    assert lookup("European", threshold=0.5, uri=store.uri) is not None


def test_api(monkeypatch, fixtures_path, store):
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    store.build()

    monkeypatch.setenv("JUDITHA_URI", store.uri)
    get_store.cache_clear()
    client = TestClient(app)

    res = client.head("/European Parliament")
    assert res.status_code == 200
    res = client.head("/European parlament")
    assert res.status_code == 404

    res = client.get("/European Parliament")
    assert res.json()["score"] > 0.97

    res = client.get("/European?threshold=0.5")
    assert res.json()["query"] == "European"
    assert "European" in res.json()["caption"]  # FIXME
    assert res.json()["schemata"] == ["PublicBody"]
    assert res.json()["score"] > 0.5

    res = client.head("/shdfjkoshfaj")
    assert res.status_code == 404
    res = client.get("/dshjka")
    assert res.status_code == 404


def test_cli(monkeypatch, fixtures_path: Path, tmp_path):
    monkeypatch.setenv("JUDITHA_URI", tmp_path)
    get_store.cache_clear()

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
    get_store.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", tmp_path)
    store = get_store()
    assert store.uri == str(tmp_path)
