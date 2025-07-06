from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from juditha import io, lookup
from juditha.api import app
from juditha.cli import cli

runner = CliRunner()


def test_io(fixtures_path, store):
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    name = "European Parliament"
    res = lookup(name)
    assert res.score > 0.97
    assert res.name == name
    assert res.names == [name]
    assert res.schema_ == "PublicBody"

    res2 = lookup(name.lower())
    assert res2.name == name
    assert res2.query == name.lower()
    assert res2.score == res.score

    assert lookup("European") is None
    assert lookup("European", threshold=0.5) is not None


def test_api(fixtures_path, store):
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)

    client = TestClient(app)

    res = client.head("/European Parliament")
    assert res.status_code == 200
    res = client.head("/European parlament")
    assert res.status_code == 404

    res = client.get("/European Parliament")
    assert res.json()["score"] > 0.97

    res = client.get("/European?threshold=0.5")
    assert res.json()["query"] == "European"
    assert "European" in res.json()["name"]  # FIXME
    assert res.json()["schema"] == "PublicBody"
    assert res.json()["score"] > 0.5

    res = client.head("/shdfjkoshfaj")
    assert res.status_code == 404
    res = client.get("/dshjka")
    assert res.status_code == 404


def test_cli(fixtures_path: Path):
    runner.invoke(cli, ["load-names", "-i", str(fixtures_path / "names.txt")])
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
