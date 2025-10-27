import itertools
from functools import cache
from typing import Generator, Literal, Self, TypeAlias

from followthemoney import model
from followthemoney.exc import InvalidData
from pydantic import BaseModel, computed_field
from rigour.names import (
    Name,
    pick_name,
    remove_obj_prefixes,
    remove_org_prefixes,
    remove_person_prefixes,
)

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


def name_key(name: str) -> str:
    key = Name(name).norm_form
    key = remove_obj_prefixes(key)
    key = remove_org_prefixes(key)
    key = remove_person_prefixes(key)
    return key


def schema_to_ner(schema: str) -> NER_TAG:
    return SCHEMA_NER.get(schema, "OTHER")


class Doc(BaseModel):
    key: str
    names: set[str] = set()
    aliases: set[str] = set()
    countries: set[str] = set()
    schemata: set[str] = set()
    score: float = 0

    @computed_field
    @property
    def ner_tags(self) -> set[NER_TAG]:
        return {schema_to_ner(s) for s in self.schemata}


Docs: TypeAlias = Generator[Doc, None, None]


class Result(Doc):
    query: str

    @classmethod
    def from_doc(cls, doc: Doc, q: str, score: float) -> Self:
        return cls(query=q, score=score, **doc.model_dump(exclude={"score"}))

    @computed_field
    @property
    def common_schema(self) -> str:
        return get_common_schema(*self.schemata)

    @computed_field
    @property
    def caption(self) -> str | None:
        return pick_name(list(self.names))


class SchemaPrediction(BaseModel):
    name: str
    label: str
    score: float

    @computed_field
    @property
    def ner_tag(self) -> NER_TAG:
        return schema_to_ner(self.label)
