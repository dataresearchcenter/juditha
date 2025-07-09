import itertools
from functools import cache
from typing import Generator, Self, TypeAlias

from followthemoney import model
from followthemoney.exc import InvalidData
from pydantic import BaseModel


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
    schemata: set[str] = set()
    score: float = 0


Docs: TypeAlias = Generator[Doc, None, None]


class Result(BaseModel):
    caption: str
    names: set[str]
    query: str
    score: float
    schemata: set[str] = set()

    @classmethod
    def from_doc(cls, doc: Doc, q: str, score: float) -> Self:
        return cls(
            caption=doc.caption,
            names=doc.names,
            query=q,
            score=score,
            schemata=doc.schemata,
        )

    @property
    def common_schema(self) -> str:
        return get_common_schema(*self.schemata)
