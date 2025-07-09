from juditha.aggregator import Aggregator
from juditha.model import get_common_schema
from juditha.store import Doc


def test_aggregation_entities(tmp_path, eu_authorities):
    path = tmp_path / "names.db"
    aggregator = Aggregator(path)
    aggregator.load_entities(eu_authorities)
    doc = next(aggregator.iterate())
    assert isinstance(doc, Doc)
    assert doc.caption == "Agencia Ejecutiva de Innovación y Redes"
    assert doc.names == {"Agencia Ejecutiva de Innovación y Redes"}
    assert doc.schemata == {"PublicBody"}
    assert aggregator.count == 153


def test_aggregation_util():
    assert get_common_schema("Person") == "Person"
    assert get_common_schema("Person", "Organization") == "LegalEntity"
    assert get_common_schema("Company", "Organization") == "Company"
    assert get_common_schema("Company", "Person") == "LegalEntity"
