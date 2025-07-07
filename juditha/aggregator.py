"""
Aggregate names from entities to caption clusters

This is needed as we could index multiple entities with similar names and we
want to canonize them all.
"""

from functools import cache
from typing import Generator, Iterable, Self, TypedDict

import duckdb
import pandas as pd
from followthemoney import EntityProxy
from rigour.names import pick_name

from juditha.model import Doc, Docs


@cache
def make_table(uri: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(uri)
    con.sql(
        "CREATE TABLE IF NOT EXISTS names (caption STRING, name STRING, schema STRING)"
    )
    return con


class Row(TypedDict):
    caption: str
    name: str
    schema: str


def unpack_entity(e: EntityProxy) -> Generator[Row, None, None]:
    names: set[str] = set()
    names.update(e.get("name"))
    names.update(e.get("alias"))
    caption = pick_name(list(names))
    if caption is not None:
        for name in names:
            yield {"caption": caption, "name": name, "schema": e.schema.name}


class Aggregator:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.table = make_table(uri)
        self.batch: list[EntityProxy] = []

    def flush(self) -> None:
        rows: Generator[Row, None, None] = (
            r for e in self.batch for r in unpack_entity(e)
        )
        df = pd.DataFrame(rows)
        duckdb.register("df", df)
        self.table.execute("INSERT INTO names SELECT * FROM df")
        self.batch = []

    def put(self, entity: EntityProxy) -> None:
        self.batch.append(entity)
        if len(self.batch) >= 10_000:
            self.flush()

    def iterate(self) -> Docs:
        for caption, names, schema in self.table.sql(
            "SELECT caption, list(name) AS names, schema FROM names "
            "GROUP BY caption, schema ORDER BY caption"
        ).fetchall():
            yield Doc(caption=caption, names=names, schema=schema)

    def load_entities(self, entities: Iterable[EntityProxy]) -> None:
        with self:
            for entity in entities:
                self.put(entity)

    @property
    def count(self) -> int:
        for (c,) in self.table.sql(
            "SELECT COUNT(DISTINCT(caption, schema)) FROM names"
        ).fetchall():
            return c
        return 0

    def __iter__(self) -> Docs:
        yield from self.iterate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.flush()
