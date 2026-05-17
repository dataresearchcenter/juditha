import itertools
from functools import cache
from typing import Generator, Self, TypeAlias

from followthemoney import model
from followthemoney.exc import InvalidData
from pydantic import BaseModel, ConfigDict, Field, computed_field
from rigour.names import pick_name


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
    key: str
    names: set[str] = set()
    aliases: set[str] = set()
    countries: set[str] = set()
    schemata: set[str] = set()
    score: float = 0


Docs: TypeAlias = Generator[Doc, None, None]


class Result(Doc):
    query: str
    took: float = 0  # milliseconds

    @classmethod
    def from_doc(cls, doc: Doc, q: str, score: float, took: float = 0) -> Self:
        return cls(
            query=q,
            score=score,
            took=took,
            **doc.model_dump(exclude={"score"}),
        )

    @computed_field
    @property
    def common_schema(self) -> str:
        return get_common_schema(*self.schemata)

    @computed_field
    @property
    def caption(self) -> str | None:
        return pick_name(list(self.names))


class Mention(BaseModel):
    # `schema` shadows BaseModel.schema; expose it via alias so the JSON
    # surface stays "schema" while the Python attribute is `schema_`.
    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    text: str
    start: int
    end: int
    schema_: str = Field(alias="schema")
