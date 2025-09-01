import itertools
from functools import cache
from typing import Generator, Literal, Self, TypeAlias

from followthemoney import model
from followthemoney.exc import InvalidData
from pydantic import BaseModel, computed_field

NER_TAG: TypeAlias = Literal["PER", "ORG", "LOC", "OTHER"]
SCHEMA_NER: dict[str, NER_TAG] = {
    "LegalEntity": "OTHER",
    "PublicBody": "ORG",
    "Company": "ORG",
    "Organization": "ORG",
    "Person": "PER",
    "Address": "LOC",
}


@cache
def get_common_schema(*schemata: str) -> str:
    if len(schemata) == 1:
        for s in schemata:
            return s
    _schemata: set[str] = set()
    for pair in itertools.pairwise(schemata):
        try:
            s = model.common_schema(*pair)
            _schemata.add(s.name)
        except InvalidData:
            pass
    if _schemata:
        return get_common_schema(*_schemata)
    return "LegalEntity"


class Doc(BaseModel):
    caption: str
    names: set[str] = set()
    aliases: set[str] = set()
    countries: set[str] = set()
    schemata: set[str] = set()
    symbols: set[str] = set()
    score: float = 0


Docs: TypeAlias = Generator[Doc, None, None]


class Result(Doc):
    query: str

    @classmethod
    def from_doc(cls, doc: Doc, q: str, score: float) -> Self:
        return cls(
            caption=doc.caption,
            names=doc.names,
            aliases=doc.aliases,
            countries=doc.countries,
            query=q,
            score=score,
            schemata=doc.schemata,
            symbols=doc.symbols,
        )

    @computed_field
    @property
    def common_schema(self) -> str:
        return get_common_schema(*self.schemata)


class SchemaPrediction(BaseModel):
    name: str
    label: str
    score: float

    @computed_field
    @property
    def ner_tag(self) -> NER_TAG:
        return SCHEMA_NER.get(self.label, "OTHER")
