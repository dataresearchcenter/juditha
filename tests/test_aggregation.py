from juditha.aggregator import Aggregator
from juditha.store import Doc


def test_aggregation_entities(tmp_path, eu_authorities):
    path = tmp_path / "names.db"
    aggregator = Aggregator(path)
    aggregator.load_entities(eu_authorities)
    doc = next(aggregator.iterate())
    assert isinstance(doc, Doc)
    assert doc.caption == "Agencia Ejecutiva de Innovación y Redes"
    assert doc.names == {"Agencia Ejecutiva de Innovación y Redes"}
    assert aggregator.count == 153
