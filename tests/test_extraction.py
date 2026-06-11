import pytest

from juditha.aggregator import Aggregator
from juditha.extraction import AhoExtractor


@pytest.fixture()
def extractor(tmp_path, eu_authorities):
    aggregator = Aggregator(tmp_path / "agg.db")
    aggregator.load_entities(eu_authorities)
    ext = AhoExtractor()
    ext.build(aggregator)
    return ext


def test_extractor_build(extractor):
    assert extractor._ac is not None
    assert len(extractor._patterns) > 0
    assert len(extractor._pattern_schema) == len(extractor._patterns)


def test_extract_known_name(extractor):
    text = "The European Parliament met today."
    mentions = extractor.extract(text)
    assert len(mentions) >= 1
    match = [m for m in mentions if "European Parliament" in m.text]
    assert len(match) == 1
    m = match[0]
    assert m.schema_ == "PublicBody"
    assert m.start == text.index("European Parliament")
    assert m.end == m.start + len("European Parliament")
    assert text[m.start : m.end] == "European Parliament"


def test_extract_no_match(extractor):
    text = "Nothing relevant here at all."
    mentions = extractor.extract(text)
    assert mentions == []


def test_extract_multiple(extractor):
    text = "The European Parliament and the European Council held a session."
    mentions = extractor.extract(text)
    texts = {m.text for m in mentions}
    assert "European Parliament" in texts
    assert "European Council" in texts
    assert len(mentions) >= 2


def test_extract_offsets(extractor):
    prefix = "xxxx "
    text = prefix + "European Parliament is important."
    mentions = extractor.extract(text)
    match = [m for m in mentions if "European Parliament" in m.text]
    assert len(match) == 1
    m = match[0]
    assert m.start == len(prefix)
    assert text[m.start : m.end] == "European Parliament"


def test_extract_dedup(extractor):
    # Same entity mentioned once should produce one mention
    text = "The European Parliament opened. The European Parliament closed."
    mentions = extractor.extract(text)
    ep = [m for m in mentions if "European Parliament" in m.text]
    # Should have exactly 2 (one per occurrence), not more from overlapping
    assert len(ep) == 2
    assert ep[0].start != ep[1].start


def test_extract_single_token_excluded(tmp_path):
    """Single-token names are always excluded from the automaton."""
    from ftmq.util import make_entity

    for name in ("EU", "Britta", "Katharina"):
        entity = make_entity(
            {
                "id": f"single-{name}",
                "schema": "Person",
                "properties": {"name": [name]},
            }
        )
        aggregator = Aggregator(tmp_path / f"single_{name}.db")
        aggregator.load_entities([entity])
        ext = AhoExtractor()
        ext.build(aggregator)
        assert ext.extract(f"We met {name} today.") == []


def test_extract_multi_token_included(tmp_path):
    """Multi-token names are included even if total length < 10."""
    from ftmq.util import make_entity

    # "Al Qamar" is 2 tokens and len < 10 -> included because >= 2 tokens
    entity = make_entity(
        {
            "id": "multi",
            "schema": "Organization",
            "properties": {"name": ["Al Qamar"]},
        }
    )
    aggregator = Aggregator(tmp_path / "multi.db")
    aggregator.load_entities([entity])
    ext = AhoExtractor()
    ext.build(aggregator)
    mentions = ext.extract("The group Al Qamar was mentioned.")
    assert len(mentions) == 1
    assert mentions[0].text == "Al Qamar"


def test_extractor_save_load(extractor, tmp_path):
    save_path = tmp_path / "aho_save"
    extractor.save(save_path)

    loaded = AhoExtractor()
    assert loaded.load(save_path) is True
    assert len(loaded._patterns) == len(extractor._patterns)

    text = "The European Parliament met today."
    orig_mentions = extractor.extract(text)
    loaded_mentions = loaded.extract(text)
    assert len(orig_mentions) == len(loaded_mentions)
    for a, b in zip(orig_mentions, loaded_mentions):
        assert a.text == b.text
        assert a.start == b.start
        assert a.end == b.end
        assert a.schema_ == b.schema_
