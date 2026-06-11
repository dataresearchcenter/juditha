# Quickstart

Five-minute walkthrough: install, populate a tiny corpus, build the index, run a lookup. CLI version first, then the same flow from Python.

## Install

`juditha` is on PyPI. It pulls in [tantivy](https://github.com/quickwit-oss/tantivy), [rigour](https://github.com/opensanctions/rigour), [FollowTheMoney](https://followthemoney.tech), and [anystore](https://docs.investigraph.dev/lib/anystore/).

```bash
pip install juditha
```

You can verify the install with `juditha --version`.

## CLI flow

`juditha` keeps its state in a directory. By default this is `juditha.db` in the current working directory; override it with the `JUDITHA_URI` environment variable.

### 1. Load names

Each line in the input becomes a known name. You can pipe from stdin or point at a file / URL.

```bash
printf "Jane Doe\nAlice Smith\nACME Holdings" | juditha load-names
```

### 2. Build the index

Loading is cheap (it just writes to the LevelDB aggregator). The tantivy index and the Aho-Corasick extractor are built in one pass over the aggregator:

```bash
juditha build
```

You need to rerun `juditha build` whenever you load new data.

### 3. Lookup

```bash
juditha lookup "Jane Doe"
juditha lookup "jane doe"         # case-insensitive, matches
juditha lookup "doe, jane"        # order-independent, matches
juditha lookup "Jane Dae"         # one-character typo, requires lower threshold
juditha lookup "Jane Dae" --threshold 0.5
```

The default fuzzy threshold is `0.97`; lower it for noisier inputs.

## Python flow

The same corpus is reachable from Python via the `juditha` package:

```python
from juditha import lookup

# Hits at the default threshold (0.97)
res = lookup("Jane Doe")
assert res is not None
assert "Jane Doe" in res.names

# Same canonical key, different surface form
assert lookup("doe, jane") is not None

# Below threshold returns None
assert lookup("Jane Dae") is None
assert lookup("Jane Dae", threshold=0.5) is not None
```

The result is a `juditha.model.Result` (a `Doc` extended with `query`, `score`, `caption`, `common_schema`). See [Usage / Python](usage/python.md) for the full surface.

## Where to point JUDITHA_URI

For a long-running worker, set `JUDITHA_URI` once before starting the process:

```bash
export JUDITHA_URI=/var/lib/juditha
juditha build
juditha lookup "Jane Doe"
```

Inside Python the same env var is read by `juditha.settings.Settings`; you can also pass `uri=` explicitly to `lookup()` or `get_store()`.

## Next

- Load real data: [Names](load-data/names.md), [Entities](load-data/entities.md).
- Drive it from a shell pipeline: [Usage / CLI](usage/cli.md), [CLI reference](cli-reference.md).
- Embed in a worker: [Usage / Python](usage/python.md).
- Pull mentions out of a fulltext: [Extract](extras/extract.md), [Percolate](extras/percolate.md).
