# Usage / CLI

`juditha` ships a [typer](https://typer.tiangolo.com)-based CLI. Every subcommand is also reachable from Python via the corresponding `Store` method, so the shell flow is mostly for convenience and one-shot scripts.

This page covers the workflow, environment variables, and `stdin` / `stdout` conventions. For the exhaustive list of options per subcommand, see [CLI reference](../cli-reference.md).

## The typical workflow

```text
load-* (any number of times)  ->  build  ->  lookup / extract / percolate
```

`load-*` writes into the LevelDB aggregator. Nothing is searchable until you run `build`, which rebuilds the tantivy index and the Aho-Corasick extractor in one pass over the aggregator.

```bash
# Ingest
juditha load-names    -i names.txt
juditha load-entities -i entities.ftm.json
juditha load-dataset  -i https://data.ftm.store/eu_authorities/index.json

# Materialise
juditha build

# Query
juditha lookup "European Parliament"
juditha extract   -i document.txt
juditha percolate -i document.txt
```

## Storage location

The store lives in a directory. Default: `juditha.db` in the current working directory. Override with the `JUDITHA_URI` environment variable.

```bash
export JUDITHA_URI=/var/lib/juditha
```

Inside the directory you will find:

- `names.db/` – LevelDB aggregator (one cluster per canonical name).
- `tantivy.db/` – the tantivy index (mmap-friendly; OS page cache shared across workers on the same host).
- `automaton.txt` – the Aho-Corasick patterns sidecar used by `juditha extract`.

You can wipe everything with `rm -rf "$JUDITHA_URI"` and start over.

## Input and output URIs

Every command accepts `-i URI` for input and (where it produces output) `-o URI` for output. URIs are handled by [anystore](https://docs.investigraph.dev/lib/anystore/), so anything anystore knows about works:

- `-` (default) for stdin / stdout.
- A local path: `-i ./names.txt`.
- An HTTP URL: `-i https://data.ftm.store/eu_authorities/entities.ftm.json`.
- An S3 URL: `-i s3://bucket/names.txt` (with the usual AWS credential envs).

```bash
# Pipe two sources
cat names1.txt names2.txt | juditha load-names

# Pull from S3
juditha load-names -i s3://my-bucket/known-orgs.txt
```

## Environment variables

All settings are prefixed with `JUDITHA_`. See [settings](../reference/settings.md)

## Top-level flags

`juditha --version` prints the installed version. `juditha --settings` prints the resolved settings (useful for confirming what `JUDITHA_URI` is pointing at). Subcommand-level options live with the subcommand; see [CLI reference](../cli-reference.md).
