# Usage / Python

`juditha` is in-process by default. Import it, get a `Store`, call its methods. It can also be [served over gRPC](api.md), in which case the same calls go over the wire without any call-site change.

## The minimal API

The package re-exports three helpers:

```python
from juditha import lookup, get_store, get_build_store
```

`lookup` is a memoised, top-level convenience for the most common case (one query, one best match). `get_store` returns the (cached) read-only store for querying (search with filters, extraction, percolation). `get_build_store` returns the (cached) write-capable store for loading data and building the index.

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

## Reading: `get_store()` and the `Store` class

```python
from juditha import get_store

store = get_store()                   # uses settings.uri (env var or default)
store = get_store("/var/lib/juditha") # explicit path
store = get_store("grpc://juditha:50051")  # remote, see the gRPC api page
```

`get_store` resolves the URI at call time and caches one store per resolved URI. A `grpc://` URI yields an `ApiStore` that proxies to a [remote server](api.md), anything else a local `Store` reading the tantivy index and the Aho-Corasick automaton.

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

Those three methods are the whole read surface, declared on the `BaseStore` abstract base class that both `Store` and `ApiStore` implement. Anything typed against `BaseStore` works local or remote.

A read-only `Store` deliberately does not open the LevelDB aggregator. plyvel takes an exclusive lock per LevelDB path, so opening it would cap the host at one juditha reader process; leaving it shut is what lets many workers (and a `juditha serve` process) mmap the same index side by side.

## Writing: `get_build_store()` and the `BuildStore` class

```python
from juditha import get_build_store
from juditha import io

store = get_build_store()

# Either: stream FTM entities into the aggregator
io.load_proxies("entities.ftm.json", store)

# ...or push individual entities
store.aggregator.put(some_entity_proxy)
store.aggregator.flush()

# Then rebuild the searchable index + extractor
store.build()
```

`BuildStore` extends `Store` with the LevelDB aggregator and the tantivy writer, so it can search as well as write. It is local-only: `get_build_store("grpc://...")` raises a `ValueError`. Because it holds the LevelDB handle, `get_build_store` caches one instance per URI, effectively a per-URI singleton.

`store.build()` deletes and recreates the tantivy index, then iterates the aggregator once feeding both tantivy and the Aho-Corasick extractor. It also clears the read-store cache, so a `Store` created earlier in the same process does not keep serving the deleted index.

### Shutting down

In a long-running worker you do not need to do anything explicit; the cached store lives for the process lifetime.

In one-shot scripts or tests that switch URIs, call `store.close()`. On a `BuildStore` that flushes pending writes, drains tantivy merges and closes the LevelDB handle; on an `ApiStore` it closes the gRPC channel; on a read-only `Store` it does nothing.

```python
store = get_build_store("/tmp/jtest")
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
