"""Manual data fixes applied while loading, via datapatch lookups.

`JUDITHA_PATCHES` points at a YAML file of named
[datapatch](https://github.com/opensanctions/datapatch) lookups. Each lookup
is named after the entity attribute it matches on: an FTM property (`name`,
`topics`, `country`, ...) or `schema`. A matching option patches the entity
before it reaches the aggregator; so far the only patch is `schema`:

    name:
      normalize: true
      lowercase: true
      options:
        - match: Deutschland
          schema: PublicBody
    topics:
      options:
        - match: corp.public
          schema: PublicBody

Patches run at load time only (`Aggregator.put`), never at build.
"""

from functools import cache

import yaml
from anystore.io import smart_read
from anystore.logging import get_logger
from datapatch import Lookup, get_lookups
from followthemoney import EntityProxy, model

log = get_logger(__name__)

Lookups = dict[str, Lookup]


@cache
def get_patches(uri: str | None) -> Lookups:
    """The lookups configured at `uri`, empty when there is none."""
    if not uri:
        return {}
    data = yaml.safe_load(smart_read(uri, mode="r")) or {}
    lookups = get_lookups(data)
    log.info("Loaded data patches.", uri=uri, lookups=sorted(lookups))
    return lookups


def _values(entity: EntityProxy, name: str) -> list[str]:
    if name == "schema":
        return [entity.schema.name]
    return entity.get(name, quiet=True)


def patch_entity(entity: EntityProxy, lookups: Lookups) -> EntityProxy:
    """Apply the first matching option of every lookup. Returns the entity
    itself when nothing matches, otherwise a new proxy with the patched
    schema (properties the new schema doesn't have are dropped)."""
    schema: str | None = None
    for name, lookup in lookups.items():
        for value in _values(entity, name):
            result = lookup.match(value)
            if result is not None and result.schema:
                schema = str(result.schema)
                break
        if schema:
            break
    if not schema or schema == entity.schema.name:
        return entity
    if model.get(schema) is None:
        raise ValueError(f"Unknown schema in data patch: {schema}")
    data = entity.to_dict()
    data["schema"] = schema
    return model.get_proxy(data)
