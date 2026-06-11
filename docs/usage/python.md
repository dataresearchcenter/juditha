# Usage / Python

`juditha` is in-process. Import it, get a `Store`, call its methods. There is no HTTP API and no client / server split.

## The minimal API

The package re-exports two helpers:

```python
from juditha import lookup, get_store
```

`lookup` is a memoised, top-level convenience for the most common case (one query, one best match). `get_store` returns the (cached) `Store` object for fine-grained access (search with filters, extraction, percolation, iterating the aggregator).

## `lookup()`

```python
from juditha import lookup

res = lookup(
    "Jane Doe",
    threshold=0.95,           # optional, defaults to settings.fuzzy_threshold
    uri=None,                 # optional, defaults to settings.uri
    schemata=("Person",),     # optional FollowTheMoney schema narrowing, must be a tuple
)
```

`lookup` is wrapped in `lru_cache(100_000)`, so repeated queries (same args) are O(1). Return value is `Result | None`.

`Result` extends `Doc` with `query`, `score`, `took` (ms), `caption` (best display name via rigour's `pick_name`), and `common_schema` ([FollowTheMoney](https://followthemoney.tech) schema reduction):

```python
res.key            # "doe jane" – the order-independent canonical key
res.names          # {"Jane Doe"}
res.aliases        # set of alternate surface forms
res.countries      # ISO country codes derived from the FTM entity
res.schemata       # FTM schemata that contributed to this cluster
res.score          # similarity in [0, 1]
res.caption        # human-readable display name
res.common_schema  # e.g. "Person", "Organization", "LegalEntity"
```

## `get_store()` and the `Store` class

```python
from juditha import get_store

store = get_store()                  # uses settings.uri (env var or default)
store = get_store("/var/lib/juditha") # explicit path
```

`get_store` resolves the URI at call time and caches one `Store` per resolved URI. plyvel allows only one open handle per LevelDB path, so this cache is effectively a per-URI singleton.

The methods you will use most:

```python
# Best-match search, same engine as juditha.lookup
result = store.search(query, threshold=None, limit=None, schemata=None)

# Aho-Corasick extraction over fulltext
mentions = store.extract("Some text mentioning Jane Doe.")

# Percolation: reverse search of the names index against the text
mentions = store.percolate("Some text mentioning Jane Doe.", slop=0)
```

`extract` and `percolate` both return `list[Mention]`. See [Extract](../extras/extract.md) and [Percolate](../extras/percolate.md) for the differences.

### Writing into the store

```python
from juditha import get_store
from juditha import io

store = get_store()

# Either: stream FTM entities into the aggregator
io.load_proxies("entities.ftm.json", store)

# ...or push individual entities
store.aggregator.put(some_entity_proxy)
store.aggregator.flush()

# Then rebuild the searchable index + extractor
store.build()
```

`store.build()` deletes and recreates the tantivy index, then iterates the aggregator once feeding both tantivy and the Aho-Corasick extractor.

### Shutting down

In a long-running worker you do not need to do anything explicit; the cached `Store` lives for the process lifetime.

In one-shot scripts or tests that switch URIs, call `store.close()` to flush pending writes, drain tantivy merges, and close the LevelDB handle:

```python
store = get_store("/tmp/jtest")
# ... do work ...
store.close()
```

## Models

`juditha.model` exposes the data classes you get back from the API. All inherit from `pydantic.BaseModel`.

```python
from juditha.model import Doc, Result, Mention
```

- `Doc(key, names, aliases, countries, schemata, score)` – an aggregated cluster.
- `Result(Doc, query, took, common_schema, caption)` – a search hit.
- `Mention(text, start, end, schema_)` – a span extracted from a fulltext. The Python attribute is `schema_` (the JSON field is `schema`; see the [Pydantic alias note](#pydantic-alias-on-mention)).

### Pydantic alias on `Mention`

`Mention.schema_` carries the FTM-style schema label of the matched name. The Python attribute is `schema_` because `BaseModel.schema` is reserved. The JSON surface uses `"schema"` via a Pydantic alias, so `mention.model_dump_json()` produces `{"text": "...", "start": ..., "end": ..., "schema": "..."}`. Both `Mention(schema="Person")` and `Mention(schema_="Person")` work on the constructor side.
