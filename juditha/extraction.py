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
from juditha.settings import Settings

log = get_logger(__name__)

# Only multi-token names with sufficient total normalized length are
# indexed. The total-length floor is derived from
# `settings.min_token_length` (env `JUDITHA_MIN_TOKEN_LENGTH`, default 4)
# so the Aho-Corasick automaton and the percolator's blocking field share
# the same notion of "too-short token". The floor is computed as
# `min_token_length * MIN_TOKEN_COUNT` (default 4 × 2 = 8, matching the
# historic `MIN_PATTERN_LENGTH`).
MIN_TOKEN_COUNT = 2


def _normalize_for_extraction(name: str) -> str:
    """Normalize a name for extraction matching.

    Uses ICU normalization (NFKC_CF + Latin-ASCII folding) on each token,
    matching the text-side tokenizer normalization exactly.
    """
    tokens = tokenize(name)
    return " ".join(t.form for t in tokens)


def _build_automaton(patterns: list[str]) -> ahocorasick_rs.AhoCorasick:
    """Build Aho-Corasick automaton with whitespace-wrapped patterns.

    Each pattern is wrapped with leading/trailing space so that the
    automaton only matches at token boundaries in whitespace-padded text.
    """
    wrapped = [f" {p} " for p in patterns]
    return ahocorasick_rs.AhoCorasick(
        wrapped,
        matchkind=ahocorasick_rs.MATCHKIND_LEFTMOST_LONGEST,
    )


class AhoExtractor:
    """Aho-Corasick entity extractor."""

    def __init__(self) -> None:
        self._ac: ahocorasick_rs.AhoCorasick | None = None
        self._patterns: list[str] = []
        # pattern index -> best schema
        self._pattern_schema: list[str] = []
        # Temporary state for incremental building
        self._pattern_docs: dict[str, set[int]] = {}
        self._doc_schemata: list[set[str]] = []

    def build(self, docs: Iterable[Doc]) -> None:
        """Build the automaton from an iterable of Docs (convenience wrapper)."""
        for i, doc in enumerate(docs):
            self.add_doc(i, doc)
        self.finalize()

    def add_doc(self, idx: int, doc: Doc) -> None:
        """Add a single doc during incremental build."""
        self._doc_schemata.append(doc.schemata)
        # Derive the per-pattern floor from the shared min_token_length
        # setting on every call. Settings() reads env vars; for build
        # loops the cost is dominated by tokenization anyway.
        min_pattern_length = Settings().min_token_length * MIN_TOKEN_COUNT
        for name in doc.names | doc.aliases:
            norm = _normalize_for_extraction(name)
            tokens = norm.split()
            if len(tokens) >= MIN_TOKEN_COUNT and len(norm) >= min_pattern_length:
                if norm not in self._pattern_docs:
                    self._pattern_docs[norm] = set()
                self._pattern_docs[norm].add(idx)

    def finalize(self) -> None:
        """Finalize the automaton after all docs have been added."""
        with Took() as t:
            patterns = sorted(self._pattern_docs)
            log.info("Finalizing extraction patterns ...", patterns=len(patterns))

            # Resolve schema per pattern from collected doc schemata
            pattern_schema: list[str] = []
            for pattern in patterns:
                doc_indices = self._pattern_docs[pattern]
                all_schemata: set[str] = set()
                for idx in doc_indices:
                    all_schemata.update(self._doc_schemata[idx])
                schema = (
                    get_common_schema(*all_schemata) if all_schemata else "LegalEntity"
                )
                pattern_schema.append(schema)

            self._patterns = patterns
            self._pattern_schema = pattern_schema

            log.info("Building Aho-Corasick automaton ...", patterns=len(patterns))
            self._ac = _build_automaton(patterns)

            # Free temporary build state
            self._pattern_docs = {}
            self._doc_schemata = []
        log.info("Aho-Corasick automaton built.", took=t.took.seconds)

    def save(self, path: Path) -> None:
        """Save patterns and schema mappings as tab-delimited text file."""
        content = "".join(
            f"{p}\t{s}\n" for p, s in zip(self._patterns, self._pattern_schema)
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
            for line in smart_stream(str(path), mode="r"):
                line = line.strip()
                if "\t" in line:
                    p, s = line.split("\t", 1)
                    patterns.append(p)
                    schemas.append(s)
            self._patterns = patterns
            self._pattern_schema = schemas
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

        # Build normalized text from tokens (space-joined forms)
        # Pad with leading/trailing space so automaton patterns
        # (which are wrapped as " pattern ") match at text boundaries
        norm_text = " " + " ".join(t.form for t in tokens) + " "

        # Build char-to-token mapping for the padded text
        # Position 0 is the leading space
        char_to_token: list[int] = [-1]  # leading space
        for i, tok in enumerate(tokens):
            if i > 0:
                char_to_token.append(-1)  # inter-token space
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
                )
            )

        return mentions
