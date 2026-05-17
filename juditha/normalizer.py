"""Name normalization, canonical key generation, and text tokenization."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rigour.names import (
    Name,
    remove_obj_prefixes,
    remove_org_prefixes,
    remove_person_prefixes,
    replace_org_types_compare,
)
from rigour.text.normalize import Normalize, normalize
from rigour.text.translit import maybe_ascii

# NFKC casefold flags applied before transliteration — matches the
# Elasticsearch ICU analysis chain (NFKC_Casefold → Latin-ASCII).
_NORMALIZE_FLAGS = Normalize.NFKC | Normalize.CASEFOLD


def icu_normalize(text: str) -> str:
    """Normalize text via rigour's NFKC casefold + Latin transliteration.

    Equivalent to the historic ICU analysis chain (NFKC_Casefold →
    Latin-ASCII) plus Cyrillic / Greek / etc. → Latin via rigour's
    maybe_ascii. Diacritics are stripped, special chars folded
    (ß→ss, ø→o), ligatures decomposed (ﬁ→fi).

    Examples:
        "Straße"    → "strasse"
        "Müller"    → "muller"
        "Élève"     → "eleve"
        "ﬁnance"   → "finance"
        "Владимир"  → "vladimir"
    """
    normalized = normalize(text, _NORMALIZE_FLAGS)
    if normalized is None:
        return ""
    return maybe_ascii(normalized)


def name_key(name: str) -> str:
    """Generate an order-independent canonical key for a name.

    Applies prefix stripping, org type canonicalization, then sorts
    comparable tokens for order independence. Returns the joined
    canonical string (e.g. "doe jane") used as both the aggregator
    cluster key and the tantivy fuzzy target.
    """
    # Strip common prefixes (Mr., Dr., The, etc.)
    cleaned = remove_person_prefixes(name)
    cleaned = remove_org_prefixes(cleaned)
    cleaned = remove_obj_prefixes(cleaned)
    # Replace org type suffixes with canonical comparable form
    cleaned = replace_org_types_compare(cleaned)
    # Parse into parts via rigour, use comparable form (accent-stripped)
    parts = Name(cleaned).parts
    # Sort comparable tokens for order independence ("Jane Doe" == "Doe, Jane")
    tokens = sorted(p.comparable for p in parts if p.comparable)
    return " ".join(tokens)


# Match word tokens (letters, digits, combining marks, hyphens within words)
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True)
class NormalizedToken:
    """A token with its position in the original text."""

    form: str  # lowercased, NFKC-normalized token
    original: str  # original surface form
    start: int  # char offset in original text
    end: int  # char offset end in original text


@dataclass(frozen=True)
class NormalizedSpan:
    """A contiguous span of normalized tokens."""

    tokens: tuple[NormalizedToken, ...]
    start: int  # char offset of first token
    end: int  # char offset after last token

    @property
    def text(self) -> str:
        """Reconstructed normalized text."""
        return " ".join(t.form for t in self.tokens)


def tokenize(text: str) -> list[NormalizedToken]:
    """Tokenize text, preserving character offsets into the original text.

    Tokenizes the original text directly so offsets are always correct.
    Each token's form is ICU-normalized (NFKC_CF + Latin-ASCII folding).
    """
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        original = match.group()
        # ICU-normalize the form for matching, but keep original offsets
        form = icu_normalize(original).strip("'-")
        if form:
            tokens.append(
                NormalizedToken(
                    form=form,
                    original=original,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tokens
