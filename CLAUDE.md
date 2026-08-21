# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

1. Don’t assume. Don’t hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.

## Code Style

- Do not remove existing inline code comments when editing files.

## Documentation Style

These rules apply to every Markdown file in this repo (README, `CLAUDE.md`, `docs/`, plan files, in-tree notes):

- Do not use the long dash (em-dash, `—`). When a dash is needed, use the en-dash (`–`) or rephrase. Hyphens (`-`) in compound words are fine.
- Do not hard-wrap lines. Each paragraph is exactly one line. Lists, code fences, tables, and headings are not paragraphs and follow their normal Markdown rules.

The user-facing docs site lives under `docs/`, is built with [zensical](https://github.com/zensical/zensical) (`mkdocs.yml` is its config), and is deployed via `make documentation`. The settings reference at `docs/reference/settings.md` is the canonical source for environment-variable behaviour.

## Project Overview

Juditha is a fast canonical name lookup service. It validates and canonizes Named Entity Recognition (NER) results against known name lists (company registries, persons of interest, etc.).

## Environment

Always use the virtualenv at `.venv`: run Python via `.venv/bin/python` and tools via `.venv/bin/<tool>`.

## Development Commands

```bash
make install      # poetry install --with dev
make test         # pytest with coverage
make typecheck    # mypy --strict juditha
make lint         # flake8
make pre-commit   # all hooks
make documentation  # zensical build + S3 sync (deploys docs.investigraph.dev)

# Single test
.venv/bin/pytest tests/test_juditha.py::test_function_name -v

# Local docs preview (no S3 push)
.venv/bin/zensical build
```

## Architecture

### Core Components

- **Store** (`store.py`): Main interface. Owns a tantivy `Index` (mmap'd on disk so the OS page cache is shared across procrastinate workers; memory footprint is 1×, not N×) and the LevelDB `Aggregator`. Public methods: `search(q, threshold, limit, schemata)`, `extract(text)` (Aho-Corasick), `percolate(text, slop)` (tantivy reverse-search), `build()`, `put()`, `flush()`, `close()`. Instances come from `get_store(uri=None)`, a thin wrapper over the `@cache`d `_store_for_uri(uri)` registry (URI-keyed, one instance per resolved path so plyvel's single-handle-per-path rule is respected). The wrapper resolves `None` against `Settings().uri` at call time, so a changed `JUDITHA_URI` is picked up without clearing the cache; `get_store.cache_clear` is aliased to the registry's for tests.

- **Aggregator** (`aggregator.py`): LevelDB-based write-optimised storage that clusters entities by `normalizer.name_key()`. Uses null-byte delimited key structure with a single `_d_` prefix for efficient prefix iteration.

- **AhoExtractor** (`extraction.py`): Aho-Corasick automaton for exact multi-token name matching on fulltext. Patterns are wrapped with leading and trailing whitespace so the automaton itself enforces token-boundary alignment. Pattern-length floor is derived from `Settings().min_token_length × MIN_TOKEN_COUNT` so it stays coupled to the percolator's blocking threshold.

- **Percolator** (`percolator.py`): Reverse-search of one document against many stored names (ES-percolate pattern). Pulls candidate `Doc`s from the names index via a BM25-scored `boolean_query` of `Should term_query` clauses over the `tokens` field, then phrase-queries each candidate against a per-call in-memory tantivy text index. `Store.percolate` is a thin wrapper around the module-level `percolate(schema, index, text, slop=0)` function.

- **Normalizer** (`normalizer.py`): `name_key()` produces an order-independent canonical string by sorting comparable tokens (so "Jane Doe" and "Doe, Jane" share a key). `icu_normalize()` runs rigour's NFKC casefold + `maybe_ascii` Latin transliteration (covers Cyrillic, Greek, etc., not just diacritics). `tokenize()` returns offset-preserving `NormalizedToken` objects; `tokenize_forms()` returns just the normalized forms as a hashable `tuple[str, ...]` (no offsets), cached via `lru_cache`.

Juditha is in-process only. There is no HTTP API; call `juditha.lookup()` (or `juditha.get_store().search()` / `.extract()` / `.percolate()`) directly from your worker / pipeline.

### Data Flow

1. **Load**: Import FTM entities or plain name lists → `Aggregator` (LevelDB).
2. **Build**: `Aggregator` → tantivy index + `AhoExtractor` (automaton). Single iteration through the aggregator feeds both consumers in one pass.
3. **Lookup**: Query → tantivy fuzzy `BooleanQuery` → top-K candidates → jaro rerank → rapidfuzz fallback → `Result`.
4. **Extract**: Text → Aho-Corasick automaton → `list[Mention]`.
5. **Percolate**: Text → tokenize → BM25 blocking on `tokens` field → in-memory text index → phrase query per candidate → `list[Mention]`.

### Tantivy schema (`store.make_schema`)

Every name-like field uses `tokenizer_name="raw"` so multi-word names index as a single term; this is what makes `FuzzyTermQuery` match across the whole normalized name without splitting on whitespace.

- `key` (stored): normalized canonical form. Fuzzy target, BM25 anchor.
- `names`, `aliases` (stored, multi-value): surface forms.
- `schemata`, `countries` (stored, multi-value): FTM metadata, used as narrowing filters.
- `qid`, `symbols` (stored, multi-value): rigour-derived narrowing filters.
- `phonetic` (index-only, multi-value): metaphone codes per `Name.part`, for the lookup-side fuzzy phonetic clause.
- `tokens` (index-only, multi-value): per-token inverted index for the percolator's blocking stage. Populated at index time with every token ≥ `MIN_TOKEN_CHARS` (hardcoded `2` in `juditha.percolator`, imported by `store.put`) across `names ∪ aliases`, symmetric with the percolator's query-time `blocking_set` filter. This field does NOT use `Settings().min_token_length`.

### Key Models (`model.py`)

- `Doc`: Aggregated document with `key`, `names`, `aliases`, `countries`, `schemata`, `score`.
- `Result`: Search result extending `Doc` with `query`, `took`, `caption` (via rigour's `pick_name`), `common_schema`.
- `Mention`: Extracted entity mention with `text`, `start`, `end`, `schema_`. The Python attribute is `schema_` (because `BaseModel.schema` is reserved); JSON serialization uses `"schema"` via a Pydantic alias.

### Configuration

Every knob lives on `juditha.settings.Settings` and is overridable via `JUDITHA_*` environment variables (or a local `.env` file); `debug` is the one exception, see below. Canonical reference: `docs/reference/settings.md`.

- `JUDITHA_URI`: Storage path (default: `juditha.db`).
- `JUDITHA_FUZZY_THRESHOLD`: Match threshold (default: `0.97`).
- `JUDITHA_LIMIT`: Max tantivy candidates per `search` (default: `10`).
- `JUDITHA_MIN_LENGTH`: Minimum query length for `lookup` (default: `4`).
- `JUDITHA_MIN_TOKEN_LENGTH`: Per-token length floor for the Aho-Corasick extractor pattern-length floor (`min_token_length × MIN_TOKEN_COUNT`). Default `4`. The percolator does NOT consult this – it uses the hardcoded `MIN_TOKEN_CHARS = 2` floor in `juditha.percolator` symmetrically at index and query time. Changing this requires `juditha build`.
- `JUDITHA_PERCOLATE_BLOCK_LIMIT`: BM25-ranked candidate cap for the percolator's blocking stage (default: `10_000`). Raise on multi-million-cluster corpora where the cap starts causing recall asymmetry vs `extract`.
- `JUDITHA_PERCOLATE_MIN_SHOULD_MATCH`: `minimum_number_should_match` for the percolator's blocking `boolean_query` (default: `2`). Read at query time, no rebuild needed.
- `DEBUG`: Toggle verbose typer tracebacks (default: `false`). Note the missing prefix: the field carries `alias="debug"`, and a pydantic-settings alias bypasses `env_prefix`, so `JUDITHA_DEBUG` has no effect. `tests/` relies on this via `[tool.pytest_env] DEBUG = 1`.

Tests use the `tmp_path` fixture for a real on-disk store per test; there is no in-memory store backend.

## CLI Commands

```bash
# Load data
juditha load-entities -i entities.ftm.json
juditha load-names -i names.txt
juditha load-dataset -i https://data.ftm.store/dataset/index.json
juditha load-catalog -i https://data.ftm.store/catalog/index.json

# Build the tantivy index + Aho-Corasick automaton
juditha build

# Inspect aggregator contents (one Doc as JSON per line)
juditha iterate -o docs.jsonl

# Lookup
juditha lookup "Jane Doe" --threshold 0.5

# Extract entity mentions (Aho-Corasick)
juditha extract -i document.txt -o mentions.json

# Percolate (tantivy reverse-search)
juditha percolate -i document.txt -o mentions.json --slop 1
```

## Dependencies

Runtime:
- `tantivy` (≥ 0.26): full-text search index (mmap, FuzzyTermQuery, phrase_query)
- `followthemoney` (≥ 4.10.2) / `ftmq[level]`: FTM entity model + LevelDB-backed querying. `ftmq` is currently pinned to the git branch `dataresearchcenter/ftmq@refactor/ql`, not a PyPI release – `juditha.io` builds its filter with the new `ftmq.Query` / `ftmq.M` API.
- `rigour` (≥ 2.3.1): name parsing, normalization, phonetics, symbols
- `anystore`: URI-based I/O abstraction
- `rapidfuzz` / `jellyfish`: string similarity scoring (rerank stage)
- `ahocorasick-rs`: Aho-Corasick automaton for extraction

Dev:
- `pytest`, `pytest-coverage`, `pytest-env`
- `mypy`, `flake8`, `black`, `isort`, `pre-commit`
- `zensical`: docs site generator

## rigour 2.x notes (worth knowing when extending)

- `SymbolCategory` is a Rust-backed enum: use `.value` (returns e.g. `"ORG_CLASS"`), not `.name`.
- `analyze_names(NameTypeTag.PER|ORG, [name])` is the unified replacement for the deleted `tag_org_name` / `tag_person_name` helpers. Returns a set of `Name` objects whose `.symbols` is populated; `NAME`-category symbols carry Q-IDs.
- `load_person_names_mapping` / `load_person_names` are deleted; the Tagger reads the corpus internally.
- `rigour.text.translit.maybe_ascii(text)` + `rigour.text.normalize.normalize(text, NFKC|CASEFOLD)` is the public replacement for the ICU NFKC_CF + Latin-ASCII chain. Used by `juditha.normalizer.icu_normalize`. Transliterates Cyrillic / Greek to Latin (which pure ICU left untouched).

## Percolator notes (worth knowing when extending)

- The in-memory text index built per `percolate()` call uses `tokenizer_name="whitespace"`, **not** `default`. Our `juditha.normalizer.tokenize()` uses the regex `[\w'-]+` and keeps apostrophes / hyphens inside tokens (`"sa'adat"`, `"jean-pierre"`, `"al-sisi"`). Tantivy's `default` tokenizer would re-split those on punctuation, so the phrase query would miss. `whitespace` keeps our pre-tokenized stream intact.
- The blocking query is a `boolean_query` of per-token `Should term_query` clauses, **not** `term_set_query`. The latter is constant-score, so a `Settings.percolate_block_limit`-sized top-K cut would degenerate to segment-order on large corpora; with `Should` clauses BM25 ranks docs that match more / rarer tokens higher.
- Tantivy 0.26 exposes `minimum_number_should_match` on `boolean_query` (0.25 did not). The percolator passes `Settings.percolate_min_should_match` (default `2`, env `JUDITHA_PERCOLATE_MIN_SHOULD_MATCH`). Recall-safe at default for names whose tokens all clear the hardcoded `MIN_TOKEN_CHARS = 2` floor in `juditha.percolator`. The floor strips single-char long-tail noise (initials, lone digits) symmetrically at index and query time; names with single-char tokens (e.g. `"A Lee"`) silently miss – accepted as noise-vs-recall trade. Raise MSM to 3+ on long inputs against multi-token-name corpora where you've measured the cut.
- Per-call cost in `percolate` scales linearly with the number of candidates surviving blocking. The in-memory text index is rebuilt every call but pinned to `num_threads=1` so the writer's 15 MB minimum heap isn't multiplied by CPU count.
