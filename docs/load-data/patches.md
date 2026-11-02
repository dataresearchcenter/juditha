# Load data / Patches

Source data is never quite right: a registry lists "Deutschland" as a `Company`, a dataset marks state-owned enterprises with the topic `corp.public` but keeps them as `Company`. Rather than fixing the source, juditha applies manual data patches while loading. The patch file is a set of [datapatch](https://github.com/opensanctions/datapatch) lookups, the same mechanism the [OpenSanctions](https://www.opensanctions.org/) pipeline uses for cleaning.

Patches run at load time only, inside `Aggregator.put`, so every loader applies them: `load-entities`, `load-dataset`, `load-catalog`, `load-names` and `load-geonames`, as well as entities pushed from Python. `juditha build` never patches; to apply a changed patch file, reload the data.

## Configuration

Point `JUDITHA_PATCHES` at a YAML file (any [anystore](https://docs.investigraph.dev/lib/anystore/) uri):

```bash
export JUDITHA_PATCHES=./patches.yml
juditha load-entities -i entities.ftm.json
juditha build
```

See [`JUDITHA_PATCHES`](../reference/settings.md#juditha_patches).

## Patch file

Each top-level key is a lookup named after the entity attribute it matches on: an FTM property (`name`, `topics`, `country`, ...) or `schema`. Every value of that attribute is run through the lookup; the first matching option patches the entity.

```yaml
name:
  normalize: true
  lowercase: true
  options:
    - match: Deutschland
      schema: PublicBody
    - match:
        - Bundesrepublik Deutschland
        - Federal Republic of Germany
      schema: PublicBody
topics:
  options:
    - match: corp.public
      schema: PublicBody
schema:
  options:
    - match: Vessel
      schema: Thing
```

Options match with `match` (exact), `contains` or `regex`, each accepting a single value or a list. `normalize: true` folds accents and punctuation and `lowercase: true` folds case, on the lookup or per option; without them matching is exact. When several options of one lookup match the same value, datapatch raises unless `weight` disambiguates. See the [datapatch README](https://github.com/opensanctions/datapatch) for the full option syntax.

The only patch so far is `schema`: the matched entity is re-created with the given FollowTheMoney schema. Properties the new schema does not know are dropped (a `Company` turned `PublicBody` loses `capital`, keeps `name`, `jurisdiction` and `topics`). An unknown schema name raises at load time.

## From Python

```python
from juditha.patch import get_patches, patch_entity

lookups = get_patches("patches.yml")
entity = patch_entity(entity, lookups)
```

`get_build_store().aggregator.put(entity)` applies the configured patches itself, so this is only needed to patch entities going elsewhere.
