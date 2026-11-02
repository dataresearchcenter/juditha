"""Load-time data patches (`JUDITHA_PATCHES`, datapatch lookups)."""

import pytest
from ftmq.util import make_entity

from juditha import io
from juditha.patch import get_patches, patch_entity
from juditha.store import get_build_store, get_store


def company(id_, name, **props):
    return make_entity(
        {"id": id_, "schema": "Company", "properties": {"name": [name], **props}}
    )


def test_patch_entity(fixtures_path):
    lookups = get_patches(str(fixtures_path / "patches.yml"))
    assert set(lookups) == {"name", "topics"}
    # by name, case-insensitive
    e = patch_entity(company("de", "deutschland", jurisdiction=["de"]), lookups)
    assert e.schema.name == "PublicBody"
    assert e.get("jurisdiction") == ["de"]
    # by topic
    e = patch_entity(company("x", "ACME Public", topics=["corp.public"]), lookups)
    assert e.schema.name == "PublicBody"
    # no match: same object
    e = company("fr", "Frankreich")
    assert patch_entity(e, lookups) is e
    # a property the new schema doesn't have is dropped, the rest survives
    e = patch_entity(company("de2", "Deutschland", capital=["1"]), lookups)
    assert e.schema.name == "PublicBody"
    assert not e.get("capital", quiet=True)
    assert e.get("name") == ["Deutschland"]


def test_patch_unknown_schema(tmp_path):
    path = tmp_path / "patches.yml"
    path.write_text("name:\n  options:\n    - match: Foo\n      schema: Nope\n")
    lookups = get_patches(str(path))
    with pytest.raises(ValueError):
        patch_entity(company("f", "Foo"), lookups)


def test_patch_no_patches():
    assert get_patches(None) == {}


def test_patch_on_load(fixtures_path, tmp_path, monkeypatch):
    get_store.cache_clear()
    get_build_store.cache_clear()
    monkeypatch.setenv("JUDITHA_PATCHES", str(fixtures_path / "patches.yml"))
    store = get_build_store(str(tmp_path / "juditha"))
    # every loader goes through Aggregator.put: plain names ...
    names = tmp_path / "names.txt"
    names.write_text("Deutschland\nFrankreich\n")
    io.load_names(names, store)
    # ... and FTM entities
    store.aggregator.load_entities(
        [
            company("acme", "ACME Public Example", topics=["corp.public"]),
            company("priv", "Private Holdings Example"),
        ]
    )
    store.build()
    assert store.aggregator.get_doc("deutschland").schemata == {"PublicBody"}
    assert store.aggregator.get_doc("frankreich").schemata == {"LegalEntity"}
    res = store.search("ACME Public Example")
    assert res is not None
    assert res.common_schema == "PublicBody"
    assert res.topics == {"corp.public"}
    assert store.search("Private Holdings Example").common_schema == "Company"
    get_store.cache_clear()
    get_build_store.cache_clear()
