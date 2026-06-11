# CLI reference

Complete reference for every `juditha` subcommand. For high-level workflow, see [Usage / CLI](usage/cli.md).

Every command honours the global `JUDITHA_*` environment variables documented in [Usage / CLI / Environment variables](usage/cli.md#environment-variables).

## Top-level

```text
juditha [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
| --- | --- |
| `--version` | Print the installed version and exit. |
| `--settings` | Print the resolved settings (URI, thresholds, …) and exit. |
| `--install-completion` | Install shell completion for the current shell. |
| `--show-completion` | Print shell completion script to stdout. |
| `--help` | Show top-level help and exit. |

## load-entities

Ingest [FollowTheMoney](https://followthemoney.tech) entities (newline-delimited JSON) into the LevelDB aggregator.

```text
juditha load-entities [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-i` | `-` | Input URI (file, HTTP, S3, stdin). |

Does not build the index. Run `juditha build` afterwards.

## load-names

Ingest a flat list of names (one per line). Each line is wrapped into a `LegalEntity` proxy.

```text
juditha load-names [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-i` | `-` | Input URI (file, HTTP, S3, stdin). |

Does not build the index. Run `juditha build` afterwards.

## load-dataset

Ingest a [nomenklatura](https://github.com/opensanctions/nomenklatura)-style dataset (a JSON config that points at one or more entity resources).

```text
juditha load-dataset [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-i` | `-` | Dataset URI. |

Does not build the index. Run `juditha build` afterwards.

## load-catalog

Ingest a [nomenklatura](https://github.com/opensanctions/nomenklatura)-style catalog (a JSON config that points at multiple datasets). Walks every dataset and loads it.

```text
juditha load-catalog [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-i` | `-` | Catalog URI. |

Does not build the index. Run `juditha build` afterwards.

## build

Rebuild the tantivy index and the Aho-Corasick extractor from the aggregator. Deletes the existing index directory first; safe to rerun.

```text
juditha build [OPTIONS]
```

No options beyond `--help`.

## lookup

Single-best-match search of the names index.

```text
juditha lookup [OPTIONS] VALUE
```

| Argument / Option | Default | Description |
| --- | --- | --- |
| `VALUE` | (required) | The query string. |
| `--threshold` | `0.97` | Minimum Jaro similarity for a match. Lower it for noisier inputs. |

Prints the `Result` (rich-formatted) on a hit, `not found` otherwise.

## iterate

Stream every aggregated cluster in the LevelDB aggregator as newline-delimited JSON. Useful for inspecting / dumping the loaded corpus.

```text
juditha iterate [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-o` | `-` | Output URI (file, HTTP, S3, stdout). |

## extract

Run the Aho-Corasick automaton over a fulltext and emit one JSON `Mention` per line. See [Extract](extras/extract.md).

```text
juditha extract [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-i` | `-` | Input URI (the document to scan). |
| `-o` | `-` | Output URI (the mention stream). |

## percolate

Reverse-search the names index against a fulltext via tantivy phrase queries. Output shape matches `extract`. See [Percolate](extras/percolate.md).

```text
juditha percolate [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-i` | `-` | Input URI (the document to scan). |
| `-o` | `-` | Output URI (the mention stream). |
| `--slop` | `0` | Allowed intervening tokens between name parts. `0` = exact phrase. |
