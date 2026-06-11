# Load data / Entities

The richer input shape is [FollowTheMoney](https://followthemoney.tech) entities (also called *proxies*): JSON objects carrying a schema (`Person`, `Company`, `Organization`, `PublicBody`, `LegalEntity`, …), a primary `name`, optional `previousName` and `alias` lists, `countries`, and arbitrary other typed properties.

Use this format when you have a registry, sanctions list, or dataset that already carries metadata you want to filter on at query time (e.g. `schemata=("Person",)`).

## Entity shape

One FTM entity per line, newline-delimited JSON:

```json
{"id": "ep-1", "schema": "PublicBody", "properties": {"name": ["European Parliament"], "country": ["eu"]}}
{"id": "ec-1", "schema": "PublicBody", "properties": {"name": ["European Council"], "alias": ["EU Council"]}}
{"id": "jane-1", "schema": "Person", "properties": {"name": ["Jane Doe"], "nationality": ["mt"], "alias": ["Jani"]}}
```

`juditha` uses `name` and `previousName` as the primary surface forms, `alias` as the secondary set, `countries` and `schemata` as facets, and the entity's caption (or `pick_name`-derived best name) as the cluster anchor.

## Single file or stream

```bash
juditha load-entities -i entities.ftm.json
cat entities.ftm.json | juditha load-entities
juditha load-entities -i https://data.ftm.store/eu_authorities/entities.ftm.json
```

After loading, you must build:

```bash
juditha build
```

## A complete dataset

Per the [nomenklatura](https://github.com/opensanctions/nomenklatura) convention, a *dataset* is a JSON config that points at one or more entity resources. `juditha load-dataset` reads the config, fetches `entities.ftm.json` (and optionally `names.txt`) from its resources, and streams them into the aggregator.

```bash
juditha load-dataset -i https://data.ftm.store/eu_authorities/index.json
juditha build
```

## A complete catalog

A *catalog* is a JSON config that points at multiple datasets. `juditha load-catalog` walks them all:

```bash
juditha load-catalog -i https://data.ftm.store/investigraph/catalog.json
juditha build
```

This can pull in gigabytes of data; load times scale linearly with the entity count.

## From Python

```python
from juditha import get_store
from juditha import io

store = get_store()

# Pick the shape you have:
io.load_proxies("entities.ftm.json", store)
io.load_dataset("https://data.ftm.store/eu_authorities/index.json", store)
io.load_catalog("https://data.ftm.store/investigraph/catalog.json", store)

store.build()
```

Each `load_*` helper accepts an optional `sync=True` flag that runs `store.build()` immediately after loading – convenient for small datasets, wasteful when you're chaining multiple loads (build once at the end instead).

## Pushing entities one at a time

For programmatic ingest you can skip the loaders and push proxies directly:

```python
from ftmq.util import make_entity
from juditha import get_store

store = get_store()
entity = make_entity({
    "id": "j",
    "schema": "Person",
    "properties": {
        "name": ["Jane Doe"],
        "nationality": ["mt"],
        "alias": ["Jani"],
    },
})
store.aggregator.put(entity)
store.aggregator.flush()
store.build()
```

`put` buffers up to 10 000 entries and auto-flushes; `flush()` is only needed at the end of a batch (or before `build`).

## Inspecting the aggregator

The aggregator clusters by `name_key` (canonical, order-independent form). To see what's in there:

```bash
juditha iterate -o docs.jsonl
```

Each line is one cluster as JSON, with the union of names, aliases, countries, and schemata that flowed into it from the FTM entities you loaded.
