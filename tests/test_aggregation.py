from juditha.aggregator import Aggregator
from juditha.model import get_common_schema
from juditha.store import Doc


def test_aggregation_entities(tmp_path, eu_authorities):
    path = tmp_path / "names.db"
    aggregator = Aggregator(path)
    aggregator.load_entities(eu_authorities)
    doc = next(aggregator.iterate())
    assert isinstance(doc, Doc)
    assert doc.names == {"Agencia Ejecutiva de Innovación y Redes"}
    assert doc.schemata == {"PublicBody"}
    assert aggregator.count == 152
    assert len([e for e in aggregator.iterate()]) == 152

    aggregator = Aggregator(tmp_path / "names2.db")
    aggregator.load_entities(eu_authorities[:1])
    assert len([e for e in aggregator.iterate()]) == 1


def test_aggregation_util():
    assert get_common_schema("Person") == "Person"
    assert get_common_schema("Person", "Organization") == "LegalEntity"
    assert get_common_schema("Company", "Organization") == "Company"
    assert get_common_schema("Company", "Person") == "LegalEntity"
