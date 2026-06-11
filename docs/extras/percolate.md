# Percolate

`juditha percolate` is the inverse of a normal search: instead of running one query against many stored documents, it runs *one document* (the input text) against *many stored queries* (every known name). It mirrors the [Elasticsearch percolate-query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-percolate-query) pattern.

Use this when:

- The known-name corpus is too big to materialise as an Aho-Corasick automaton at build time.
- You want phrase-level tolerance (intervening tokens, via `slop`).
- You want to keep the door open for future fuzzy-phrase or phonetic extraction (the percolator runs against tantivy, which has all those query types).

The output shape is identical to [`extract`](extract.md): a `list[Mention]` with original-text offsets.

## CLI

```bash
echo "The Jane Doe meeting was at 3pm." > /tmp/doc.txt
juditha percolate -i /tmp/doc.txt
# {"text":"Jane Doe","start":4,"end":12,"schema":"LegalEntity"}
```

`-i` and `-o` follow the same [anystore](https://docs.investigraph.dev/lib/anystore/) conventions as every other juditha subcommand.

### Slop

`--slop N` allows up to `N` intervening tokens between consecutive name parts. The default is `0` (exact phrase). With `slop=1`, a stored "Jane Doe" matches "Jane M. Doe" in the input:

```bash
echo "Jane M. Doe was here." > /tmp/doc.txt

juditha percolate -i /tmp/doc.txt              # no hit
juditha percolate -i /tmp/doc.txt --slop 1
# {"text":"Jane M. Doe","start":0,"end":11,"schema":"LegalEntity"}
```

The returned `start` / `end` cover the *entire matched span* in the original text (including the intervening token), so `text[m.start:m.end]` is what was actually matched.

## Python

```python
from juditha import get_store

store = get_store()
mentions = store.percolate("The Jane Doe meeting was at 3pm.")
# [Mention(text='Jane Doe', start=4, end=12, schema_='LegalEntity')]

mentions = store.percolate("Jane M. Doe was here.", slop=1)
# [Mention(text='Jane M. Doe', start=0, end=11, schema_='LegalEntity')]
```

## How it works

1. **Tokenize the input text** with `juditha.normalizer.tokenize`. Each `NormalizedToken` carries its ICU-normalized form *and* its original-text `start` / `end` offsets.
2. **Block against the names index**. The tantivy schema has a `tokens` field (multi-value, raw): every normalized token (≥ 4 chars) across every `name` and `alias`. A single `term_set_query` over the input text's tokens returns the candidate cluster `Doc`s in one posting-list union pass. Tantivy's BM25 ranks docs that share more / rarer tokens with the input text higher.
3. **Build an in-memory tantivy index of the input text**. One document, indexed with the positional `default` tokenizer so phrase queries work. Pinned to a single writer thread so the heap stays at tantivy's minimum (15 MB) instead of multiplying by CPU count.
4. **Phrase-query each candidate name against the in-memory text index**. `phrase_query(..., slop=slop)` decides whether the name actually appears in the text as a contiguous (or up-to-`slop`-gapped) sequence.
5. **Recover original offsets** via a Python subsequence scan over the original token list. The phrase-query confirms the match; the scan locates it by token index, which maps trivially back to byte offsets in the original text.

The whole second-stage in-memory index lives only for the duration of the percolate call and is GC'd immediately after.

## Tuning

`percolate_block_limit` (env `JUDITHA_PERCOLATE_BLOCK_LIMIT`, default `100_000`) caps the BM25-ranked candidate pool returned by the first stage. With BM25 ranking the top entries are almost always the right ones, but for extremely long inputs you may want to raise this; the cost is linear in the number of candidates that survive blocking. The cap is read from `Settings()` on every `percolate()` call, so env-var changes take effect without restart.

`settings.min_token_length` (env `JUDITHA_MIN_TOKEN_LENGTH`, default `4`) filters stopwords out of the blocking set and out of the `tokens` field at index time. The same setting feeds the Aho-Corasick extractor's pattern-length floor (`min_token_length × MIN_TOKEN_COUNT`), so the two methods share one knob. `MIN_TOKEN_COUNT = 2` lives in `juditha.percolator` and skips single-token names regardless.

## Percolate vs extract

| | `extract` (Aho-Corasick) | `percolate` (tantivy) |
| --- | --- | --- |
| Build memory | scales with corpus size (full automaton in RAM) | bounded (just the tantivy index) |
| Build time | one pass | already part of `juditha build` |
| Per-call time | very fast (O(n) over text) | one tantivy query plus per-candidate phrase check |
| Variant tolerance | exact normalized match only | `slop`; future fuzzy / phonetic via tantivy query types |
| Output | `list[Mention]` | `list[Mention]` |

For small-to-medium corpora the two produce identical results at `slop=0`; the `tests/test_percolate.py::test_percolate_parity_with_extract` test asserts that on the EU authorities fixture.
