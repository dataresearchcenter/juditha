# Load data / Names

The simplest input shape is a flat list of names: one per line, UTF-8, no header. Use this when you don't have structured [FollowTheMoney](https://followthemoney.tech) entities and just want a quick "is this string a known name" check.

## Format

```text
Jane Doe
Alice Smith
ACME Holdings GmbH
European Parliament
European Central Bank
```

There is no concept of aliases, schema, country, or any other metadata in this format. Every line is treated as a `LegalEntity` (FTM's most general schema for a named legal person). If you need richer structure, switch to [entities](entities.md).

Empty lines and trailing whitespace are tolerated. Lines that normalize to fewer than `JUDITHA_MIN_LENGTH` characters are dropped at query time.

## From a file

```bash
juditha load-names -i names.txt
```

The path is interpreted by [anystore](https://docs.investigraph.dev/lib/anystore/), so local paths, HTTP, and S3 URLs all work:

```bash
juditha load-names -i https://example.org/known-orgs.txt
juditha load-names -i s3://my-bucket/persons-of-interest.txt
```

## From stdin

`-i -` (the default) reads from stdin:

```bash
printf "Jane Doe\nAlice Smith\n" | juditha load-names
cat names1.txt names2.txt | juditha load-names
```

Streaming from another tool also works:

```bash
curl -s https://example.org/list.txt | juditha load-names
```

## From Python

```python
from juditha import get_store
from juditha import io

store = get_store()
io.load_names("names.txt", store)
# or pipe-style from stdin:
io.load_names("-", store)

store.build()
```

`io.load_names` accepts an optional `schema=` argument (default `"LegalEntity"`) if you want every loaded name treated as a different FTM schema:

```python
io.load_names("companies.txt", store, schema="Company")
io.load_names("people.txt",    store, schema="Person")
```

## What happens under the hood

Each name is wrapped into a minimal FTM entity (id `name-<n>`, schema as configured, `name` property set to the input string) and pushed into the LevelDB aggregator via `Store.aggregator.put`. The aggregator clusters entries by `name_key` (order-independent, accent-stripped, prefix-cleaned canonical form), so multiple invocations of `load-names` accumulate – they don't overwrite.

To make the loaded data searchable you must run `juditha build` (or `store.build()` from Python). Until then `juditha lookup` will not see the new names.

## Reset

To start over, delete the store directory and reload:

```bash
rm -rf "$JUDITHA_URI"
juditha load-names -i names.txt
juditha build
```
