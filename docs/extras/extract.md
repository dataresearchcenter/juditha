# Extract

`juditha extract` walks a fulltext and returns every stored name that appears in it. The mechanism is an [Aho-Corasick automaton](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm) built at index time over every multi-token normalized name in the corpus.

Use this when:

- You need exact (post-normalization) matches with maximum speed.
- Your corpus fits comfortably in memory at build time.
- You don't need fuzzy / phonetic / variant tolerance during extraction (those still apply to `juditha lookup`).

For variant-tolerant extraction (or for corpora large enough that the automaton would blow up at build time) see [Percolate](percolate.md).

## CLI

```bash
echo "The European Parliament met today." > /tmp/doc.txt
juditha extract -i /tmp/doc.txt
# {"text":"European Parliament","start":4,"end":23,"schema":"PublicBody"}
```

`-i` reads input from a file / URL / stdin. `-o` writes the mention list to a file / URL / stdout, one JSON object per line.

```bash
# Pipe straight from a document store
curl -s https://example.org/report.txt | juditha extract -o mentions.json
```

## Python

```python
from juditha import get_store

store = get_store()
text = "The European Parliament met today, and the European Council convened later."
mentions = store.extract(text)
for m in mentions:
    print(m.text, m.start, m.end, m.schema_)
# European Parliament 4 23 PublicBody
# European Council    44 60 PublicBody
```

`store.extract` returns `list[Mention]`. `Mention.text` is the original surface form (the slice `text[m.start:m.end]`), `Mention.schema_` is the [FollowTheMoney](https://followthemoney.tech) schema label (use `Mention.schema_` in Python; the JSON surface uses `"schema"` via Pydantic alias).

## How it works

1. At `build` time every name in every cluster is ICU-normalized (NFKC casefold + Latin transliteration via rigour) and tokenized. Names with fewer than 2 tokens or with a total normalized length under 8 characters are dropped (filters out "EU", "ag", and similar noise).
2. Each surviving pattern is wrapped with leading and trailing spaces (`" european parliament "`) so the automaton itself enforces token-boundary alignment.
3. At extraction time the input text is tokenized with the same ICU normalization, joined back together with spaces, and run through the automaton in a single O(n) pass.
4. Match positions in the normalized text are mapped back to original-text byte offsets via a precomputed `char_to_token` array, so `Mention.start` and `Mention.end` index into the *original* text.

## Persistence

The automaton is persisted alongside the tantivy index as `automaton.txt` (tab-delimited `pattern<TAB>schema` per line). The first `extract` call after a process start loads it from disk; subsequent calls reuse the in-memory automaton.

`juditha build` regenerates this file in the same pass that rebuilds the tantivy index.

## Limits

- Exact post-normalization match only. "Müller" and "muller" match (ICU folding), but "Jane Doe" and "Jane M. Doe" do not (the intervening token breaks the match).
- Single-token names ("EU", "Britta") are dropped to keep noise low. If you need to surface those, [percolate](percolate.md) doesn't change that floor either, by design.
- Build-time memory scales with corpus size. For multi-million-name corpora `juditha percolate` is the more sustainable choice.
