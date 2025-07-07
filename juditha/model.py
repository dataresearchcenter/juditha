from typing import Generator, Self, TypeAlias

from followthemoney.proxy import EntityProxy
from followthemoney.types import registry
from pydantic import BaseModel, Field


class Doc(BaseModel):
    caption: str
    names: set[str] = set()
    schema_: str = Field(alias="schema", default="")
    score: float = 0

    @classmethod
    def from_proxy(cls, proxy: EntityProxy) -> Self:
        return cls(
            caption=proxy.caption,
            names=set(proxy.get_type_values(registry.name)),
            schema=proxy.schema.name,
        )


Docs: TypeAlias = Generator[Doc, None, None]


class Result(BaseModel):
    name: str
    names: set[str]
    query: str
    score: float
    schema_: str | None = Field(alias="schema", default=None)

    @classmethod
    def from_doc(cls, doc: Doc, q: str, score: float) -> Self:
        return cls(
            name=doc.caption,
            names=doc.names,
            query=q,
            score=score,
            schema=doc.schema_ or None,
        )
