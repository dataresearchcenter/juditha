# Settings

Settings management is handled via [pydantic-settings](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/), so either env vars with `JUDITHA_` prefix or a local `.env` file works.

## Storage and query

### `JUDITHA_URI`

| | |
| --- | --- |
| Field | `uri` |
| Type | `str` |
| Default | `juditha.db` |
| Read at | every `get_store()` call (resolved at call time) |
| Rebuild needed | n/a (it selects which store to open) |

Path to the store directory. Holds the LevelDB aggregator (`names.db/`), the tantivy index (`tantivy.db/`), and the Aho-Corasick patterns file (`automaton.txt`). Supports any anystore-resolvable path: local filesystem, S3, ...

### `JUDITHA_FUZZY_THRESHOLD`

| | |
| --- | --- |
| Field | `fuzzy_threshold` |
| Type | `float` |
| Default | `0.97` |
| Read at | every `Store.search` / `lookup()` call |
| Rebuild needed | no |

Jaro similarity threshold for `lookup` / `Store.search`. Hits below this score are dropped. Lower for noisier inputs (OCR, NER spans with edit-distance corruption); raise to demand near-exact matches.

### `JUDITHA_LIMIT`

| | |
| --- | --- |
| Field | `limit` |
| Type | `int` |
| Default | `10` |
| Read at | every `Store.search` / `lookup()` call |
| Rebuild needed | no |

Maximum candidate hits tantivy returns per search before the Jaro / rapidfuzz rerank stage. Raise if the rerank often drops the only correct candidate because it ranked outside the top-10 by BM25.

### `JUDITHA_MIN_LENGTH`

| | |
| --- | --- |
| Field | `min_length` |
| Type | `int` |
| Default | `4` |
| Read at | every `Store.search` / `lookup()` call |
| Rebuild needed | no |

Minimum length of the normalized query. Queries that normalize to fewer than this many characters return `None` without touching tantivy.

## Index-time tuning

### `JUDITHA_MIN_TOKEN_LENGTH`

| | |
| --- | --- |
| Field | `min_token_length` |
| Type | `int` |
| Default | `4` |
| Read at | **build time** (`Store.build`, `AhoExtractor.add_doc`) and query time (percolator blocking) |
| Rebuild needed | **yes** (`juditha build`) |

Per-token length floor shared by:

- The percolator's blocking set (`{f for f in text_forms if len(f) >= min_token_length}`)
- The `tokens` field on the names index (populated at index time with the same floor)
- The Aho-Corasick extractor's pattern-length filter, derived as `min_token_length × MIN_TOKEN_COUNT` (default `4 × 2 = 8` total normalized chars)

Lower this to admit shorter, stopword-ish tokens. Short-token names like `Abu Bakr` or `XI Jinping` then carry richer blocking signatures and survive the BM25 `percolate_block_limit` cap on large corpora. Cost: index size grows and the blocking-set fan-out increases at query time.

Index-time and query-time use of this setting are coupled: the percolator's blocking set at query time MUST match the floor used when populating the `tokens` field at index time, otherwise blocking misses real candidates. Always run `juditha build` after changing this.

## Query-time tuning

### `JUDITHA_PERCOLATE_BLOCK_LIMIT`

| | |
| --- | --- |
| Field | `percolate_block_limit` |
| Type | `int` |
| Default | `10_000` |
| Read at | every `Store.percolate` / `percolate()` call |
| Rebuild needed | no |

Maximum number of candidate `Doc`s the percolator's blocking stage returns. Candidates are BM25-ranked by how many (and how rare) of the input text's tokens they match. Tantivy 0.25 does not expose a `minimum_should_match` knob, so the only way to ensure low-frequency candidates aren't crowded out at multi-million-cluster scale is to raise this cap.

| Setting | When | Cost |
| --- | --- | --- |
| `10_000` (default) | Corpora up to ~1 M clusters, latency-sensitive workloads | ~1 s on 90 K-token input |
| `100_000` | Multi-million-cluster corpora where parity with Aho matters | ~4 to 5 s on 90 K-token input |
| `1_000_000` | Investigative / batch jobs where recall trumps latency | linear in candidates surviving blocking |

See [Benchmark](../benchmark.md) for the cost / recall curve at three corpus sizes.

## Debug

### `JUDITHA_DEBUG`

| | |
| --- | --- |
| Field | `debug` |
| Type | `bool` |
| Default | `false` |
| Read at | CLI startup |
| Rebuild needed | no |

When `true`, typer renders rich tracebacks on CLI errors. Useful for development.

## Module-level constants (not settings)

Some thresholds live as module-level constants in `juditha.percolator` and `juditha.extraction` rather than on `Settings`, either because they're conceptually coupled to a class invariant or because changing them is invasive enough to warrant a code change:

| Constant | Module | Value | Purpose |
| --- | --- | ---: | --- |
| `MIN_TOKEN_COUNT` | `juditha.percolator`, `juditha.extraction` | `2` | Single-token names are always skipped (too noisy as both Aho patterns and phrase-query candidates). Same value, declared in two places to keep modules independent. |
| `NUM_CPU` | `juditha.store` | `multiprocessing.cpu_count()` | Sizes the tantivy writer's heap × num_threads at build time. |
