"""Percolator: reverse-search of one document against many stored names.

Mirrors the Elasticsearch percolate-query pattern. The names tantivy
index carries a per-token inverted field (`tokens`) populated at index
time by `Store.put`. At query time we:

1. Tokenize the input text via `juditha.normalizer.tokenize` (which
   also ICU-normalizes, so the blocking set already shares the same
   normalization as the indexed tokens — no extra normalization pass
   is needed here).
2. Issue a scored `boolean_query` of per-token `term_query` Should
   clauses against the `tokens` field on the names index. BM25 ranks
   docs sharing rarer / multiple input tokens higher, so the
   `Settings.percolate_block_limit` cap returns the most relevant
   candidates. (`term_set_query` looks similar but is constant-score,
   so the cap would degenerate to segment-order on large corpora.)
3. Build a one-document in-memory tantivy index of the input text
   (using the positional `default` tokenizer so phrase_query works).
4. For each candidate name (from `names | aliases`), phrase-query the
   in-memory text index. If it matches, recover original-text offsets
   via a Python token-subsequence scan over the `NormalizedToken`
   list — that gives us byte-exact `start` / `end` without parsing
   tantivy's snippet/highlight API.

`slop` (default 0) is passed through to `Query.phrase_query` AND to the
subsequence scan so original-offset recovery stays aligned with what
tantivy actually matched.
"""

from __future__ import annotations

from functools import cache
from typing import Iterable

import tantivy

from juditha.model import Mention, get_common_schema
from juditha.normalizer import tokenize, tokenize_forms
from juditha.settings import Settings

# Percolator tuning. `min_token_length` is in `Settings` (env
# `JUDITHA_MIN_TOKEN_LENGTH`, default 4) because both the percolator
# blocking filter and the AhoExtractor pattern-length floor consume it;
# changing it requires `juditha build` since the `tokens` field is
# populated at index time. The blocking-set cap is also in `Settings`
# (`percolate_block_limit`, env `JUDITHA_PERCOLATE_BLOCK_LIMIT`).
MIN_TOKEN_COUNT = 2  # single-token names too noisy to percolate


@cache
def _make_text_schema() -> tantivy.Schema:
    """Schema for the per-call in-memory text index.

    Uses the `whitespace` tokenizer, NOT `default`: we pre-tokenize the
    input with `juditha.normalizer.tokenize` (regex `[\\w'-]+`), which
    preserves apostrophes and hyphens inside tokens (`"sa'adat"`,
    `"jean-pierre"`, `"al-sisi"`). Tantivy's `default` tokenizer would
    re-split those on punctuation, so a phrase_query like
    `["ahmad", "sa'adat"]` would never hit; `whitespace` keeps our
    tokens intact and only splits on space (which is where we already
    put the boundaries).

    Both tokenizers index positions, so `phrase_query` works.
    """
    b = tantivy.SchemaBuilder()
    b.add_text_field("text", tokenizer_name="whitespace", stored=False)
    return b.build()


def _build_text_index(text_forms: list[str]) -> tantivy.Index:
    """Build a one-document in-memory tantivy index of the pre-normalized
    text tokens, joined by spaces.

    Pre-normalization matters: the names index stores ICU-normalized
    tokens; we feed the same normalized forms into the text index so
    phrase_query token-matching aligns on both sides.

    Heap is the tantivy minimum (15 MB) but pinned to a single thread —
    tantivy's writer defaults to one thread per CPU which would multiply
    the heap by NUM_CPU. For a one-document index that's wasteful.
    """
    schema = _make_text_schema()
    idx = tantivy.Index(schema)
    writer = idx.writer(heap_size=15_000_000, num_threads=1)
    writer.add_document(tantivy.Document(text=" ".join(text_forms)))
    writer.commit()
    idx.reload()
    return idx


def _match_with_slop(
    haystack: list[str], needle: list[str], start: int, slop: int
) -> int | None:
    """Try to match `needle` against `haystack` starting at index `start`
    with up to `slop` intervening tokens between consecutive needle
    elements. Returns the inclusive end index of the matched span on
    success, or None if no match."""
    n = len(needle)
    j = start
    ni = 1
    used = 0
    while ni < n and j < len(haystack) - 1:
        j += 1
        if haystack[j] == needle[ni]:
            ni += 1
        else:
            used += 1
            if used > slop:
                return None
    return j if ni == n else None


def _find_token_subsequences(
    haystack: list[str], needle: list[str], slop: int = 0
) -> Iterable[tuple[int, int]]:
    """Yield (start_idx, end_idx) pairs where `needle` matches `haystack`
    with at most `slop` total intervening tokens between consecutive
    `needle` tokens.

    With `slop=0` this is a strict contiguous match
    (`end_idx == start_idx + len(needle) - 1`). With `slop > 0` the
    matched span may be longer than `len(needle)` — we return the
    inclusive bounds of the actual span in `haystack` so original-text
    offsets map correctly.
    """
    n = len(needle)
    if n == 0:
        return
    if slop == 0:
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                yield (i, i + n - 1)
        return
    # slop > 0: greedy windowed scan anchored on each needle[0] occurrence
    for i, tok in enumerate(haystack):
        if tok != needle[0]:
            continue
        end_idx = _match_with_slop(haystack, needle, i, slop)
        if end_idx is not None:
            yield (i, end_idx)


def percolate(
    schema: tantivy.Schema,
    index: tantivy.Index,
    text: str,
    slop: int = 0,
) -> list[Mention]:
    """Run percolation against the given names index.

    Returns one `Mention` per unique (start, end) span in `text`.
    """
    # 1. Tokenize the input text — NormalizedTokens carry original offsets.
    text_tokens = tokenize(text)
    if not text_tokens:
        return []
    text_forms = [t.form for t in text_tokens]
    # Read settings at call time so env-var changes apply without restart.
    settings = Settings()
    min_token_length = settings.min_token_length
    blocking_set = {f for f in text_forms if len(f) >= min_token_length}
    if not blocking_set:
        return []

    # 2. Block: query the names tantivy index for Docs whose tokens
    # overlap the input text. We use a `boolean_query` with `Should`
    # clauses (one term_query per input token) rather than a
    # `term_set_query` because the latter is a constant-score query and
    # leaves the top-K cut as essentially segment-order. With Should
    # clauses BM25 ranks docs that match more (and rarer) input tokens
    # higher, so the cap returns the relevant candidates instead of
    # whichever ones tantivy iterated first.
    Occur = tantivy.Occur
    Q = tantivy.Query
    block_q = Q.boolean_query(
        [
            (Occur.Should, Q.term_query(schema, "tokens", t))
            for t in sorted(blocking_set)
        ]
    )
    block_limit = settings.percolate_block_limit
    searcher = index.searcher()
    block_hits = searcher.search(block_q, block_limit).hits
    if not block_hits:
        return []

    # 3. Build in-memory tantivy index of just the input text.
    text_schema = _make_text_schema()
    text_index = _build_text_index(text_forms)
    text_searcher = text_index.searcher()

    # 4. For each candidate name, phrase-query the text index. Phrase
    # hit ⇒ recover original offsets via Python subsequence scan.
    mentions: list[Mention] = []
    seen: set[tuple[int, int]] = set()
    for _, addr in block_hits:
        tdoc = searcher.doc(addr)
        data = tdoc.to_dict()
        doc_names = list(data.get("names", [])) + list(data.get("aliases", []))
        schemata = data.get("schemata", []) or ["LegalEntity"]
        schema_label = get_common_schema(*schemata)

        for name in doc_names:
            # tokenize_forms is cached — across percolate calls the same
            # candidate names get tokenized only once.
            name_tokens = list(tokenize_forms(name))
            if len(name_tokens) < MIN_TOKEN_COUNT:
                continue
            # tantivy's phrase_query expects list[str | tuple[int, str]];
            # we always pass plain strings, hence the type:ignore.
            phrase_q = tantivy.Query.phrase_query(
                text_schema, "text", name_tokens, slop=slop  # type: ignore[arg-type]
            )
            if not text_searcher.search(phrase_q, 1).hits:
                continue
            for start_idx, end_idx in _find_token_subsequences(
                text_forms, name_tokens, slop=slop
            ):
                orig_start = text_tokens[start_idx].start
                orig_end = text_tokens[end_idx].end
                span = (orig_start, orig_end)
                if span in seen:
                    continue
                seen.add(span)
                mentions.append(
                    Mention(
                        text=text[orig_start:orig_end],
                        start=orig_start,
                        end=orig_end,
                        schema=schema_label,
                    )
                )
    return mentions
