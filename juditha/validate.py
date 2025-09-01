"""
A very poor but helpful validation mechanism based on name tokens we know. This
eliminates NER noise by testing if the extracted name contains any of a known
token of a huge set of names.
"""

from typing import TypeAlias

from anystore.decorators import anycache
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

Tokens: TypeAlias = dict[NER_TAG, list[str]]
MIN_TOKEN_LENGTH = 5


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
    return {k: list(v) for k, v in buffer.items()}


class Validator:
    def __init__(self, uri: Uri, aggregator: Aggregator) -> None:
        self.uri = uri
        self.aggregator = aggregator
        self.key_func = (
            lambda *args, **kwargs: f"validate_tokens_{self.aggregator.count_rows}.json"
        )
        self.cache = anycache(
            store=get_store(self.uri),
            serialization_mode="json",
            key_func=self.key_func,
        )
        self._tokens: Tokens = {}

    def get_tokens(self) -> Tokens:
        if not self._tokens:
            log.info(
                "Loading tokens ...",
                uri=join_uri(self.uri, self.key_func()),
            )
            self._tokens = self.cache(build_tokens)(self.aggregator)
        return self._tokens

    def validate_name(self, name: str, tag: NER_TAG) -> bool:
        """Test if the given name shares any normalized tokens with the known
        sets of tokens for the given tag (PER, ORG, LOC)"""
        tokens = self.get_tokens()
        name_tokens = _name_tokens(name)
        need = len(name_tokens) // 2
        seen = 0
        for token in _name_tokens(name):
            if token in tokens.get(tag, []):
                seen += 1
                if seen >= need:
                    return True
        return False
