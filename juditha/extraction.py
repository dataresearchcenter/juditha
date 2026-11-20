"""Aho-Corasick based entity extraction.

Builds an automaton from all normalized name forms in the aggregator,
then matches against normalized text in a single O(n) pass.

Patterns are indexed with leading/trailing whitespace so the automaton
itself enforces token-boundary alignment — no post-filtering needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import ahocorasick_rs
from anystore.io import smart_stream, smart_write
from anystore.logging import get_logger
from anystore.util import Took

from juditha.model import Doc, Mention, get_common_schema
from juditha.normalizer import tokenize
from juditha.settings import settings

log = get_logger(__name__)

# Only multi-token names with sufficient total normalized length are
# indexed. The total-length floor is derived from
# `settings.min_token_length` (env `JUDITHA_MIN_TOKEN_LENGTH`, default 4)
# so the Aho-Corasick automaton and the percolator's blocking field share
# the same notion of "too-short token". The floor is computed as
# `min_token_length * MIN_TOKEN_COUNT` (default 4 × 2 = 8, matching the
# historic `MIN_PATTERN_LENGTH`).
#
# Docs carrying the `Address` schema (places, e.g. from `load-geonames`)
# are exempt from the token-count floor: "Berlin", "Paris" or "Iran" must
# be extractable, so their floor is a single token of `min_token_length`
# characters. Noise control for places happens in the geonames loader
# (population and length floors), not here.
MIN_TOKEN_COUNT = 2
ADDRESS_SCHEMA = "Address"


def _normalize_for_extraction(name: str) -> str:
    """Normalize a name for extraction matching.

    Uses ICU normalization (NFKC_CF + Latin-ASCII folding) on each token,
    matching the text-side tokenizer normalization exactly.
    """
    tokens = tokenize(name)
    return " ".join(t.form for t in tokens)


# Token separator in the matched text and inside the automaton patterns.
# Two spaces, not one: a pattern is wrapped as " berlin ", so its trailing
# space is the first of the pair and the next pattern's leading space the
# second. With a single space the first match would consume the only
# boundary and the adjacent name ("Berlin Hamburg", "Berlin, Hamburg,
# Kiel") could never start, since leftmost-longest matches don't overlap.
TOKEN_SEP = "  "

# Per-pattern build state: (schemata, topics) unioned over its docs.
_Meta = tuple[frozenset[str], frozenset[str]]


def _build_automaton(patterns: list[str]) -> ahocorasick_rs.AhoCorasick:
    """Build Aho-Corasick automaton with whitespace-wrapped patterns.

    Each pattern is wrapped with leading/trailing space so that the
    automaton only matches at token boundaries in whitespace-padded text;
    its inner single spaces become `TOKEN_SEP` to line up with the text.
    """
    # generator: no second in-memory copy of all patterns
    wrapped = (f" {p.replace(' ', TOKEN_SEP)} " for p in patterns)
    # store_patterns=False: `extract` only needs match indexes
    return ahocorasick_rs.AhoCorasick(
        wrapped,
        matchkind=ahocorasick_rs.MATCHKIND_LEFTMOST_LONGEST,
        store_patterns=False,
    )


class AhoExtractor:
    """Aho-Corasick entity extractor."""

    def __init__(self) -> None:
        self._ac: ahocorasick_rs.AhoCorasick | None = None
        self._patterns: list[str] = []
        # pattern index -> best schema
        self._pattern_schema: list[str] = []
        # pattern index -> union of FTM topics across the docs carrying it
        self._pattern_topics: list[tuple[str, ...]] = []
        # Temporary state for incremental building: pattern -> meta,
        # interned via `_meta` (few distinct combinations)
        self._pattern_meta: dict[str, _Meta] = {}
        self._meta: dict[_Meta, _Meta] = {}

    def _intern(self, meta: _Meta) -> _Meta:
        return self._meta.setdefault(meta, meta)

    def build(self, docs: Iterable[Doc]) -> None:
        """Build the automaton from an iterable of Docs (convenience wrapper)."""
        for i, doc in enumerate(docs):
            self.add_doc(i, doc)
        self.finalize()

    def add_doc(self, idx: int, doc: Doc) -> None:
        """Add a single doc during incremental build (`idx` is unused)."""
        doc_meta = self._intern((frozenset(doc.schemata), frozenset(doc.topics)))
        # Derive the per-pattern floor from the shared min_token_length setting.
        min_tokens = 1 if ADDRESS_SCHEMA in doc.schemata else MIN_TOKEN_COUNT
        min_pattern_length = settings.min_token_length * min_tokens
        for name in doc.names | doc.aliases:
            norm = _normalize_for_extraction(name)
            tokens = norm.split()
            if len(tokens) >= min_tokens and len(norm) >= min_pattern_length:
                meta = self._pattern_meta.get(norm)
                if meta is None:
                    self._pattern_meta[norm] = doc_meta
                elif meta is not doc_meta:
                    self._pattern_meta[norm] = self._intern(
                        (meta[0] | doc_meta[0], meta[1] | doc_meta[1])
                    )

    def finalize(self) -> None:
        """Finalize the automaton after all docs have been added."""
        with Took() as t:
            patterns = sorted(self._pattern_meta)
            log.info("Finalizing extraction patterns ...", patterns=len(patterns))

            # Resolve schema and topics per pattern from collected doc state.
            # Schemata reduce to their common ancestor; topics are unioned,
            # a shared name is flagged if any cluster carrying it is.
            # Resolved once per distinct (schemata, topics) combination.
            resolved: dict[_Meta, tuple[str, tuple[str, ...]]] = {}
            for meta in self._meta:
                all_schemata, all_topics = meta
                schema = (
                    get_common_schema(*all_schemata) if all_schemata else "LegalEntity"
                )
                resolved[meta] = (schema, tuple(sorted(all_topics)))
            pattern_schema: list[str] = []
            pattern_topics: list[tuple[str, ...]] = []
            for pattern in patterns:
                schema, topics = resolved[self._pattern_meta[pattern]]
                pattern_schema.append(schema)
                pattern_topics.append(topics)

            # Free temporary build state before building the automaton (peak memory)
            self._pattern_meta = {}
            self._meta = {}
            del resolved

            self._patterns = patterns
            self._pattern_schema = pattern_schema
            self._pattern_topics = pattern_topics

            log.info("Building Aho-Corasick automaton ...", patterns=len(patterns))
            self._ac = _build_automaton(patterns)
        log.info("Aho-Corasick automaton built.", took=t.took.seconds)

    def save(self, path: Path) -> None:
        """Save patterns, schema and topics as a tab-delimited text file.

        One line per pattern: `<pattern>\\t<schema>[\\t<topic>...]`. Topics
        are optional trailing columns, so a file written before topics
        existed still loads (with empty topics).
        """
        content = "".join(
            "\t".join((p, s, *t)) + "\n"
            for p, s, t in zip(
                self._patterns, self._pattern_schema, self._pattern_topics
            )
        )
        smart_write(str(path), content, mode="w")

    def load(self, path: Path) -> bool:
        """Load patterns from text file and rebuild automaton."""
        if not path.exists():
            return False

        with Took() as t:
            log.info("Loading Aho-Corasick patterns ...")
            patterns: list[str] = []
            schemas: list[str] = []
            topics: list[tuple[str, ...]] = []
            for line in smart_stream(str(path), mode="r"):
                line = line.strip()
                if "\t" in line:
                    pattern, schema, *pattern_topics = line.split("\t")
                    patterns.append(pattern)
                    schemas.append(schema)
                    topics.append(tuple(pattern_topics))
            self._patterns = patterns
            self._pattern_schema = schemas
            self._pattern_topics = topics
            # Wrap patterns with whitespace for token-boundary matching
            self._ac = _build_automaton(patterns)
        log.info(
            "Aho-Corasick automaton loaded.",
            patterns=len(self._patterns),
            took=t.took.seconds,
        )
        return True

    def extract(self, text: str) -> list[Mention]:
        """Extract entity mentions from text."""
        if self._ac is None:
            return []

        # Tokenize text, preserving original offsets
        tokens = tokenize(text)
        if not tokens:
            return []

        # Build normalized text from tokens (TOKEN_SEP-joined forms)
        # Pad with leading/trailing space so automaton patterns
        # (which are wrapped as " pattern ") match at text boundaries
        norm_text = " " + TOKEN_SEP.join(t.form for t in tokens) + " "

        # Build char-to-token mapping for the padded text
        # Position 0 is the leading space
        char_to_token: list[int] = [-1]  # leading space
        for i, tok in enumerate(tokens):
            if i > 0:
                char_to_token.extend([-1] * len(TOKEN_SEP))  # inter-token gap
            for _ in tok.form:
                char_to_token.append(i)
        char_to_token.append(-1)  # trailing space

        # Run Aho-Corasick on padded text
        matches = self._ac.find_matches_as_indexes(norm_text)

        # Convert matches to Mentions
        # Automaton patterns are " pattern ", so match_start points to the
        # leading space and match_end points past the trailing space.
        # The actual content starts at match_start+1, ends at match_end-1.
        seen: set[tuple[int, int]] = set()
        mentions: list[Mention] = []
        for pattern_idx, match_start, match_end in matches:
            # Strip the wrapping spaces from match positions
            content_start = match_start + 1
            content_end = match_end - 1

            # Map padded positions to token indices
            start_tok = char_to_token[content_start]
            end_tok = char_to_token[content_end - 1]
            if start_tok < 0 or end_tok < 0:
                continue

            # Original text offsets from tokens
            orig_start = tokens[start_tok].start
            orig_end = tokens[end_tok].end

            # Dedup: skip if we already have a mention at this span
            span_key = (orig_start, orig_end)
            if span_key in seen:
                continue
            seen.add(span_key)

            # Surface: join the original-case forms of just the matched
            # tokens with single spaces, rather than slicing the raw
            # original text. The slice picks up whatever punctuation the
            # tokenizer stripped between tokens (e.g. `text[i:j]` for a
            # matched ["men", "so"] over `'men". So'` returns the noisy
            # `'men". So'`). Joining `t.original` reflects what was
            # actually matched. `start` / `end` still cover the
            # inclusive byte span for callers that need the raw slice.
            surface = " ".join(t.original for t in tokens[start_tok : end_tok + 1])

            mentions.append(
                Mention(
                    text=surface,
                    start=orig_start,
                    end=orig_end,
                    schema=self._pattern_schema[pattern_idx],
                    topics=set(self._pattern_topics[pattern_idx]),
                )
            )

        return mentions
