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
| Read at | **build time** (`AhoExtractor.add_doc`) |
| Rebuild needed | **yes** (`juditha build`) |

Per-token length floor for the Aho-Corasick extractor's pattern-length filter, derived as `min_token_length × MIN_TOKEN_COUNT` (default `4 × 2 = 8` total normalized chars). Lower this to admit shorter patterns into the automaton (e.g. `"Le Pen"`, `"Al Qamar"`). Cost: more spurious mentions from very short patterns.

The percolator does NOT consult this setting: it uses the hardcoded `MIN_TOKEN_CHARS = 2` floor in `juditha.percolator` symmetrically at index time (the `tokens` field) and query time (the `blocking_set`). Noise above 2 chars is filtered at query time by [`percolate_min_should_match`](#juditha_percolate_min_should_match) instead.

## Query-time tuning

### `JUDITHA_PERCOLATE_BLOCK_LIMIT`

| | |
| --- | --- |
| Field | `percolate_block_limit` |
| Type | `int` |
| Default | `10_000` |
| Read at | every `Store.percolate` / `percolate()` call |
| Rebuild needed | no |

Maximum number of candidate `Doc`s the percolator's blocking stage returns. Candidates are BM25-ranked by how many (and how rare) of the input text's tokens they match. See also [`JUDITHA_PERCOLATE_MIN_SHOULD_MATCH`](#juditha_percolate_min_should_match) – raising that knob on a "clean" corpus removes most of the pressure on this cap by pruning weak-overlap candidates at the posting-list stage.

| Setting | When | Cost |
| --- | --- | --- |
| `10_000` (default) | Corpora up to ~1 M clusters, latency-sensitive workloads | ~1 s on 90 K-token input |
| `100_000` | Multi-million-cluster corpora where parity with Aho matters | ~4 to 5 s on 90 K-token input |
| `1_000_000` | Investigative / batch jobs where recall trumps latency | linear in candidates surviving blocking |

See [Benchmark](../benchmark.md) for the cost / recall curve at three corpus sizes.

### `JUDITHA_PERCOLATE_MIN_SHOULD_MATCH`

| | |
| --- | --- |
| Field | `percolate_min_should_match` |
| Type | `int` |
| Default | `2` |
| Read at | every `Store.percolate` / `percolate()` call |
| Rebuild needed | no |

`minimum_number_should_match` passed to the percolator blocking `boolean_query` (tantivy 0.26+). Counts the number of distinct input-text tokens a candidate `Doc` must share with the query before BM25 even ranks it.

Default `2` is recall-safe for names whose tokens all clear `MIN_TOKEN_CHARS = 2` (a hardcoded floor in `juditha.percolator` that strips single-char long-tail noise like initials and lone digits, symmetric at index and query time). The percolator only ever phrase-queries names with `>= MIN_TOKEN_COUNT == 2` tokens, so every percolatable doc that clears the char floor contributes >= 2 tokens to the index. Names like `"A Lee"` (where only `"lee"` clears 2 chars) silently miss – accepted as the noise-vs-recall trade-off of the char floor.

| Setting | When | Cost |
| --- | --- | --- |
| `2` (default) | Any corpus | Big win on multi-million-cluster corpora: most noise candidates are dropped at the posting-list stage, `percolate_block_limit` rarely binds |
| `3+` | Long-input batch jobs against multi-token-name corpora where you've measured the cut | Drops names with exactly 2 tokens unless the input shares 3+ tokens with the doc; only use after measuring |

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
