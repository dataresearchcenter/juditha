"""
A very poor but helpful validation mechanism based on name tokens we know. This
eliminates NER noise by testing if the extracted name contains any of a known
token of a huge set of names.
"""

from collections.abc import Generator
from typing import TextIO, TypeAlias

from anystore.io import logged_items
from anystore.logging import get_logger
from anystore.store import get_store
from anystore.types import Uri
from anystore.util import join_uri
from normality import latinize_text
from rigour.names import Name

from juditha.aggregator import Aggregator
from juditha.model import NER_TAG

log = get_logger(__name__)

Tokens: TypeAlias = dict[NER_TAG, set[str]]
MIN_TOKEN_LENGTH = 5
NULL = "\x00"


def serialize_tokens(tokens: Tokens) -> bytes:
    """Serialize tokens to null-byte delimited format.

    Format: One line per tag, tag followed by null-delimited tokens.
    PER\x00token1\x00token2\x00...
    ORG\x00token1\x00token2\x00...
    LOC\x00token1\x00token2\x00...
    """
    lines = []
    for tag in ("PER", "ORG", "LOC"):
        tag_tokens = tokens.get(tag, set())
        line = NULL.join([tag] + sorted(tag_tokens))
        lines.append(line)
    return "\n".join(lines).encode("utf-8")


def parse_token_line(fp: TextIO) -> Generator[str, None, None]:
    """Yield null-delimited tokens from a single line. Stops at newline or EOF."""
    token = ""
    while True:
        c = fp.read(1)
        if not c or c == "\n":
            if token:
                yield token
            return
        if c == NULL:
            yield token
            token = ""
        else:
            token += c


def deserialize_tokens(fp: TextIO) -> Tokens:
    """Deserialize null-byte delimited format to token sets."""
    tokens: Tokens = {"PER": set(), "ORG": set(), "LOC": set()}
    while True:
        line_tokens = parse_token_line(fp)
        tag = next(line_tokens, None)
        if tag is None:
            break
        if tag in tokens:
            tokens[tag].update(line_tokens)  # type: ignore[index]
    return tokens


def _schema_to_tag(schema: str) -> NER_TAG:
    if schema == "Address":
        return "LOC"
    if schema == "Person":
        return "PER"
    return "ORG"


def _name_tokens(name: str) -> set[str]:
    tokens: set[str] = set()
    if len(name) < MIN_TOKEN_LENGTH:
        return tokens
    n = Name(name)
    for part in n.parts:
        if len(part.form) < MIN_TOKEN_LENGTH:
            continue
        if part.latinize:
            tokens.add(latinize_text(part.form))
        else:
            tokens.add(part.form)
    return tokens


def _name_tag_tokens(name: str, schema: str) -> tuple[set[str], NER_TAG]:
    return _name_tokens(name), _schema_to_tag(schema)


def build_tokens(aggregator: Aggregator) -> Tokens:
    buffer: dict[NER_TAG, set[str]] = {"PER": set(), "ORG": set(), "LOC": set()}
    for name_tokens, tag in logged_items(
        (_name_tag_tokens(n, s) for n, s in aggregator.iter_names_schema()),
        "Load",
        item_name="Token",
        logger=log,
        total=aggregator.count_rows,
    ):
        buffer[tag].update(name_tokens)
    return buffer


class Validator:
    KEY = "tokens.txt"

    def __init__(self, uri: Uri, aggregator: Aggregator) -> None:
        self.uri = uri
        self.aggregator = aggregator
        self.store = get_store(self.uri)
        self._tokens: Tokens = {}

    def get_tokens(self) -> Tokens:
        if not self._tokens:
            log.info("Loading tokens ...", uri=join_uri(self.uri, self.KEY))
            try:
                with self.store.open(self.KEY, mode="r") as io:
                    self._tokens = deserialize_tokens(io)
            except Exception:
                # Cache miss - build and store
                self._tokens = build_tokens(self.aggregator)
                self.store.put(
                    self.KEY, serialize_tokens(self._tokens), serialization_mode="raw"
                )
        return self._tokens

    def validate_name(self, name: str, tag: NER_TAG) -> bool:
        """Test if the given name shares some (~50%) normalized tokens with the
        known sets of tokens for the given tag (PER, ORG, LOC)"""
        tokens = self.get_tokens()
        name_tokens = _name_tokens(name)
        need = len(name_tokens) // 2
        seen = 0
        for token in name_tokens:
            if token in tokens.get(tag, set()):
                seen += 1
                if seen >= need:
                    return True
        return False
