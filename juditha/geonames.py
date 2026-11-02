"""Load places from a GeoNames dump as FollowTheMoney `Address` entities.

Port of https://github.com/dataresearchcenter/geonames-tagger: the same row
and name-variant filters, but the surviving rows become `Address` entities
that flow through the regular aggregator / index / automaton instead of a
separate places file. Each entity id is the typed form `GEO:<geonameid>`.
"""

import csv
import io
import os
import zipfile
from functools import lru_cache
from typing import IO, Generator, Iterator

from anystore.io import SyncProgressBar, smart_open
from anystore.types import Uri
from followthemoney import EntityProxy
from ftmq.util import make_entity

from juditha.extraction import _normalize_for_extraction
from juditha.settings import Settings

GEONAMES_DUMP_URI = "https://download.geonames.org/export/dump/allCountries.zip"

# https://download.geonames.org/export/dump/readme.txt
SOURCE_FIELDNAMES = (
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
)

# Row filters (from geonames-tagger). Admin areas and populated places
# only; the raw `name` length floor is stricter for cities (P) to drop
# short ambiguous town names, looser for admin areas (A) so countries like
# "Iran" survive. Big cities are never noise, so from `BIG_CITY_POPULATION`
# on the name floor and the dictionary-word filter below are waived:
# otherwise "Hamburg" (7 chars) or "Kiel" are unreachable, as their admin
# rows carry the bare name only as an alternate that the substring rule
# drops, and "Berlin" or "China" are common words in every language.
FEATURES = ("A", "P")
MIN_NAME_LENGTH = 8  # P rows
MIN_NAME_LENGTH_A = 4  # A rows
MIN_POPULATION = 5_000
BIG_CITY_POPULATION = 100_000

# Dictionary-word filter (from geonames-tagger): a single-token variant of
# a place below `BIG_CITY_POPULATION` is dropped when it is a common word
# in any of these languages. Input text language is unknown at build time,
# so the check is multilingual: "auch" (a French town, German for "also")
# produced 633 false mentions in one German report. Only single tokens are
# checked, wordfreq scores phrases like "deutsche bank" as common too;
# hyphenated tokens ("saint-laurent") are exempt for the same reason.
# zipf > 4.5 is roughly the top 30k words of a language. Needs the
# `juditha[geonames]` extra (wordfreq).
COMMON_WORD_LANGS = ("de", "en", "fr", "es", "it", "nl", "pt", "ru")
COMMON_WORD_MAX_ZIPF = 4.5

# Normalized length floor for the canonical `name` / `asciiname` variants.
# Alternates use `Settings.geonames_min_alternate_length` instead.
MIN_NORM_LENGTH = 4

# Rows between two progress bar updates: `tell()` on a remote file is cheap,
# but not per row of a 13M-row dump.
PROGRESS_EVERY = 1_000


def _size(fh: IO[bytes]) -> int | None:
    """Byte size of an open file, if the backend knows it (fsspec files
    carry `.size`, local files answer `fstat`)."""
    size = getattr(fh, "size", None)
    if isinstance(size, int):
        return size
    try:
        return os.fstat(fh.fileno()).st_size
    except (OSError, AttributeError, ValueError, io.UnsupportedOperation):
        return None


def _advance(bar: SyncProgressBar, fh: IO[bytes], last: int) -> int:
    """Move the bar to the file's current byte position; returns it."""
    try:
        pos = fh.tell()
    except (OSError, AttributeError, ValueError, io.UnsupportedOperation):
        return last
    if pos > last:
        bar.advance(items=pos - last, size=pos - last)
        return pos
    return last


def _data_member(zh: zipfile.ZipFile, uri: str) -> str:
    stem = uri.rstrip("/").rsplit("/", 1)[-1][: -len(".zip")].lower()
    members = [m for m in zh.namelist() if m.lower().endswith(".txt")]
    for m in members:
        if m.lower() == f"{stem}.txt":
            return m
    members = [m for m in members if not m.lower().startswith("readme")]
    if not members:
        raise ValueError(f"No data member in zip: {uri}")
    return members[0]


def _open_dump(fh: IO[bytes], uri: str) -> Iterator[IO[str]]:
    """Yield a text handle on the tab-separated dump, unpacking a zip if
    needed. GeoNames zips (`allCountries.zip`, per-country `DE.zip`,
    `cities5000.zip`) hold the data as `<stem>.txt` next to a `readme.txt`."""
    if str(uri).lower().endswith(".zip"):
        with zipfile.ZipFile(fh) as zh:
            with zh.open(_data_member(zh, uri)) as h:
                yield io.TextIOWrapper(h, encoding="utf-8")
    else:
        yield io.TextIOWrapper(fh, encoding="utf-8")


def iter_rows(uri: Uri) -> Generator[dict[str, str], None, None]:
    """Yield the raw dump rows that pass the feature / population / name
    length floors. Shows a progress bar over the bytes read from `uri`, which
    for a remote dump is the download itself."""
    name = str(uri).rstrip("/").rsplit("/", 1)[-1]
    with (
        smart_open(uri, "rb") as fh,
        SyncProgressBar(f"Reading {name}", total=_size(fh)) as bar,
    ):
        for th in _open_dump(fh, str(uri)):
            # a zip was just seeked around for its directory: start from here
            last = _advance(bar, fh, 0)
            reader = csv.DictReader(
                th, fieldnames=SOURCE_FIELDNAMES, delimiter="\t", quoting=csv.QUOTE_NONE
            )
            for i, row in enumerate(reader, 1):
                if i % PROGRESS_EVERY == 0:
                    last = _advance(bar, fh, last)
                if row["feature_class"] not in FEATURES:
                    continue
                population = int(row.get("population") or 0)
                if population < MIN_POPULATION:
                    continue
                name_len = len(row["name"] or "")
                if population < BIG_CITY_POPULATION:
                    if row["feature_class"] == "P":
                        if name_len < MIN_NAME_LENGTH:
                            continue
                    elif name_len < MIN_NAME_LENGTH_A:
                        continue
                yield row
            _advance(bar, fh, last)


@lru_cache(maxsize=200_000)
def is_dictionary_word(norm: str) -> bool:
    """Whether `norm` (a single normalized token) is a common word in any
    of `COMMON_WORD_LANGS`."""
    try:
        from wordfreq import zipf_frequency
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "`load-geonames` needs wordfreq: pip install 'juditha[geonames]'"
        ) from e
    return any(
        zipf_frequency(norm, lang) > COMMON_WORD_MAX_ZIPF for lang in COMMON_WORD_LANGS
    )


def _is_plain_token(norm: str) -> bool:
    return " " not in norm and "-" not in norm


def _is_substring_token(norm: str, canonical_norm: str) -> bool:
    """Single token that is a strict substring of the canonical norm."""
    if " " in norm or len(norm) >= len(canonical_norm):
        return False
    return norm in canonical_norm


def select_variants(row: dict[str, str], min_alternate_length: int) -> list[str]:
    """Pick the surface forms worth indexing for one row: `name` and
    `asciiname` (floor `MIN_NORM_LENGTH`), then `alternatenames` entries
    that clear `min_alternate_length`, aren't numeric codes and aren't a
    single-token strict substring of the canonical norm ("berlin" in
    "land berlin", "kern" in "kern county": those exist as their own rows
    if they are real places). Below `BIG_CITY_POPULATION`, single-token
    variants that are dictionary words are dropped too. Returns the
    original strings, deduplicated on their normalized form, canonical
    first."""
    canonical_norm = _normalize_for_extraction(row["name"] or "")
    small = int(row.get("population") or 0) < BIG_CITY_POPULATION
    candidates: list[tuple[str, bool]] = [
        (v, False) for v in (row.get("name"), row.get("asciiname")) if v
    ]
    if row.get("alternatenames"):
        candidates.extend((v, True) for v in row["alternatenames"].split(",") if v)
    seen: set[str] = set()
    variants: list[str] = []
    for value, is_alternate in candidates:
        norm = _normalize_for_extraction(value)
        min_len = min_alternate_length if is_alternate else MIN_NORM_LENGTH
        if len(norm) < min_len or norm in seen:
            continue
        if norm.replace(" ", "").isdigit():
            continue
        if is_alternate and _is_substring_token(norm, canonical_norm):
            continue
        if small and _is_plain_token(norm) and is_dictionary_word(norm):
            continue
        seen.add(norm)
        variants.append(value)
    return variants


def make_address(row: dict[str, str], variants: list[str]) -> EntityProxy:
    name, *aliases = variants
    properties: dict[str, list[str]] = {"name": [name]}
    if aliases:
        properties["alias"] = aliases
    if row.get("country_code"):
        properties["country"] = [row["country_code"].lower()]
    return make_entity(
        {
            "id": f"GEO:{row['geonameid']}",
            "schema": "Address",
            "properties": properties,
        }
    )


def iter_entities(uri: Uri) -> Generator[EntityProxy, None, None]:
    """One `Address` entity per surviving GeoNames row."""
    min_alternate_length = Settings().geonames_min_alternate_length
    for row in iter_rows(uri):
        variants = select_variants(row, min_alternate_length)
        if variants:
            yield make_address(row, variants)
