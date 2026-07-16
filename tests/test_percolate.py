from pathlib import Path

import pytest
from ftmq.util import make_entity
from typer.testing import CliRunner

from juditha import io
from juditha.cli import cli
from juditha.store import get_store

runner = CliRunner()


@pytest.fixture()
def percolatable(fixtures_path, store):
    """Store with EU authorities loaded and tantivy + extractor built."""
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    store.build()
    return store


def test_percolate_known_name(percolatable):
    text = "The European Parliament met today."
    mentions = percolatable.percolate(text)
    assert len(mentions) >= 1
    match = [m for m in mentions if "European Parliament" in m.text]
    assert len(match) == 1
    m = match[0]
    assert m.start == text.index("European Parliament")
    assert m.end == m.start + len("European Parliament")
    assert text[m.start : m.end] == "European Parliament"


def test_percolate_no_match(percolatable):
    assert percolatable.percolate("Nothing relevant here at all.") == []


def test_percolate_multiple(percolatable):
    text = "The European Parliament and the European Council held a session."
    mentions = percolatable.percolate(text)
    texts = {m.text for m in mentions}
    assert "European Parliament" in texts
    assert "European Council" in texts


def test_percolate_offsets(percolatable):
    prefix = "xxxx "
    text = prefix + "European Parliament is important."
    mentions = percolatable.percolate(text)
    match = [m for m in mentions if "European Parliament" in m.text]
    assert len(match) == 1
    m = match[0]
    assert m.start == len(prefix)
    assert text[m.start : m.end] == "European Parliament"


def test_percolate_dedup(percolatable):
    """Same entity mentioned twice should produce two mentions (per-span)."""
    text = "The European Parliament opened. The European Parliament closed."
    mentions = percolatable.percolate(text)
    ep = [m for m in mentions if "European Parliament" in m.text]
    assert len(ep) == 2
    assert ep[0].start != ep[1].start


def test_percolate_single_token_excluded(tmp_path):
    """Single-token names are always excluded from percolation (MIN_TOKEN_COUNT=2)."""
    for name in ("EU", "Britta", "Katharina"):
        entity = make_entity(
            {
                "id": f"single-{name}",
                "schema": "Person",
                "properties": {"name": [name]},
            }
        )
        s = get_store(str(tmp_path / f"s_{name}"))
        s.aggregator.load_entities([entity])
        s.build()
        assert s.percolate(f"We met {name} today.") == []


def test_percolate_multi_token_included(tmp_path):
    """Short multi-token names ("Al Qamar") are matched."""
    entity = make_entity(
        {
            "id": "multi",
            "schema": "Organization",
            "properties": {"name": ["Al Qamar"]},
        }
    )
    s = get_store(str(tmp_path / "multi"))
    s.aggregator.load_entities([entity])
    s.build()
    mentions = s.percolate("The group Al Qamar was mentioned.")
    assert len(mentions) == 1
    assert mentions[0].text == "Al Qamar"


def test_percolate_parity_with_extract(percolatable):
    """For canonical-form mentions on the EU authorities fixture, percolate
    and extract should find the same (text, start) pairs."""
    text = (
        "The European Parliament met today, the European Council convened "
        "later, and the European Central Bank issued a statement."
    )
    p = {(m.text, m.start) for m in percolatable.percolate(text)}
    e = {(m.text, m.start) for m in percolatable.extract(text)}
    assert p == e


def test_percolate_slop(tmp_path):
    """With slop=1 a name with an intervening token still matches."""
    entity = make_entity(
        {
            "id": "j",
            "schema": "Person",
            "properties": {"name": ["Jane Doe"]},
        }
    )
    s = get_store(str(tmp_path / "slop"))
    s.aggregator.load_entities([entity])
    s.build()
    text = "Jane M. Doe was here."
    # No match at slop=0 (intervening token "m" / "M.")
    assert s.percolate(text, slop=0) == []
    # Match at slop=1 — span covers Jane..Doe inclusive
    mentions = s.percolate(text, slop=1)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.start == text.index("Jane")
    assert m.end == text.index("Doe") + len("Doe")
    assert text[m.start : m.end] == "Jane M. Doe"


def test_cli_percolate(monkeypatch, fixtures_path: Path, tmp_path):
    """`juditha percolate` reads stdin/file, writes mentions to stdout."""
    get_store.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", str(tmp_path / "juditha"))
    store = get_store()
    io.load_proxies(fixtures_path / "eu_authorities.ftm.json", store)
    store.build()

    text_path = tmp_path / "input.txt"
    text_path.write_text("The European Parliament met today.")

    res = runner.invoke(cli, ["percolate", "-i", str(text_path)])
    assert res.exit_code == 0
    assert "European Parliament" in res.output
    assert '"schema":"PublicBody"' in res.output or "PublicBody" in res.output


def test_percolate_short_token_name_matches_at_default_msm(tmp_path):
    """At the default MSM=2, "Jane Doe" matches: both tokens are
    indexed in `tokens` (no per-token length filter) and the input
    contributes both to the blocking set."""
    entity = make_entity(
        {
            "id": "j",
            "schema": "Person",
            "properties": {"name": ["Jane Doe"]},
        }
    )
    s = get_store(str(tmp_path / "j"))
    s.aggregator.load_entities([entity])
    s.build()
    mentions = s.percolate("Jane Doe was here.")
    assert any(m.text == "Jane Doe" for m in mentions)


def test_percolate_surface_strips_punctuation_between_tokens(tmp_path):
    """When the tokenizer strips punctuation between two matched
    tokens (quotes, periods, hyphens), the captured `text` reflects
    the matched-token reconstruction, not the raw byte slice. `start`
    / `end` still bracket the inclusive original span so callers can
    recover the raw slice if needed."""
    entity = make_entity(
        {
            "id": "ms",
            "schema": "Organization",
            "properties": {"name": ["men so"]},
        }
    )
    s = get_store(str(tmp_path / "ms"))
    s.aggregator.load_entities([entity])
    s.build()
    text = 'foo bar men". So baz'
    mentions = s.percolate(text)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.text == "men So"
    raw = text[m.start : m.end]
    assert raw == 'men". So'


def test_percolate_surface_slop_drops_intervening_punctuation(tmp_path):
    """At slop=1 the surface contains intervening tokens (intended)
    but NOT the punctuation between them."""
    entity = make_entity(
        {
            "id": "j",
            "schema": "Person",
            "properties": {"name": ["Jane Doe"]},
        }
    )
    s = get_store(str(tmp_path / "jslop"))
    s.aggregator.load_entities([entity])
    s.build()
    text = "Jane M. Doe was here."
    mentions = s.percolate(text, slop=1)
    assert len(mentions) == 1
    m = mentions[0]
    assert m.text == "Jane M Doe"
    assert text[m.start : m.end] == "Jane M. Doe"


def test_percolate_one_token_input_short_circuits(percolatable):
    """Inputs with fewer tokens than `percolate_min_should_match`
    short-circuit to []. Single-token inputs can't satisfy MSM=2."""
    assert percolatable.percolate("Parliament.") == []


def test_cli_percolate_slop(monkeypatch, tmp_path):
    """`juditha percolate --slop 1` finds "Jane M. Doe" for stored "Jane Doe"."""
    get_store.cache_clear()
    monkeypatch.setenv("JUDITHA_URI", str(tmp_path / "juditha"))
    store = get_store()
    entity = make_entity(
        {
            "id": "j",
            "schema": "Person",
            "properties": {"name": ["Jane Doe"]},
        }
    )
    store.aggregator.load_entities([entity])
    store.build()

    text_path = tmp_path / "input.txt"
    text_path.write_text("Jane M. Doe was here.")

    res = runner.invoke(cli, ["percolate", "-i", str(text_path)])
    assert res.exit_code == 0
    assert "Jane" not in res.output  # no match at slop=0

    res = runner.invoke(cli, ["percolate", "-i", str(text_path), "--slop", "1"])
    assert res.exit_code == 0
    # Surface is the matched-token reconstruction (period stripped),
    # not the raw slice; raw slice is recoverable via start/end.
    assert "Jane M Doe" in res.output
