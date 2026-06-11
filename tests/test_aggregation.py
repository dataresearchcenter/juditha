from juditha.aggregator import Aggregator
from juditha.model import Doc, get_common_schema
from juditha.normalizer import name_key


def test_aggregation_entities(tmp_path, eu_authorities):
    path = tmp_path / "names.db"
    aggregator = Aggregator(path)
    aggregator.load_entities(eu_authorities)
    doc = next(aggregator.iterate())
    assert isinstance(doc, Doc)
    assert len(doc.names) > 0
    assert len(doc.schemata) > 0
    all_docs = list(aggregator.iterate())
    assert len(all_docs) > 100  # ~152 EU authority entities
    assert aggregator.count == len(all_docs)

    aggregator = Aggregator(tmp_path / "names2.db")
    aggregator.load_entities(eu_authorities[:1])
    assert len([e for e in aggregator.iterate()]) == 1


def test_aggregation_get_doc(tmp_path, eu_authorities):
    path = tmp_path / "names.db"
    aggregator = Aggregator(path)
    aggregator.load_entities(eu_authorities)
    key = name_key("European Parliament")
    doc = aggregator.get_doc(key)
    assert doc is not None
    assert "European Parliament" in doc.names


def test_name_key_order_independent():
    assert name_key("Jane Doe") == name_key("Doe, Jane")
    assert name_key("Jane Doe") == name_key("doe jane")


def test_name_key_accent_independent():
    assert name_key("Jané Doe") == name_key("Jane Doe")


def test_aggregation_util():
    assert get_common_schema("Person") == "Person"
    assert get_common_schema("Person", "Organization") == "LegalEntity"
    assert get_common_schema("Company", "Organization") == "Company"
    assert get_common_schema("Company", "Person") == "LegalEntity"
