from juditha.aggregator import Aggregator
from juditha.validate import Validator


def test_validator(tmp_path, eu_authorities):
    path = tmp_path / "names.db"
    aggregator = Aggregator(path)
    aggregator.load_entities(eu_authorities)

    validator = Validator(tmp_path / "validator", aggregator)
    name = "European Thingy"
    assert validator.validate_name(name, "ORG")
    assert not validator.validate_name(name, "PER")
    assert not validator.validate_name(name, "LOC")
