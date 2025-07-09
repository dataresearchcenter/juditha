"""
Aggregate names from entities to caption clusters

This is needed as we could index multiple entities with similar names and we
want to canonize them all.
"""

from functools import cache
from typing import Iterable, Self, TypedDict

import duckdb
import pandas as pd
from anystore.logging import get_logger
from followthemoney import EntityProxy, registry
from rigour.names import pick_name

from juditha.model import Doc, Docs

log = get_logger(__name__)


@cache
def make_table(uri: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(uri)
    con.sql(
        "CREATE TABLE IF NOT EXISTS names (caption STRING, schema STRING, names STRING[])"
    )
    return con


class Row(TypedDict):
    caption: str
    names: set[str]
    schema: str


def unpack_entity(e: EntityProxy) -> Row | None:
    names: set[str] = set()
    names.update(e.get("name"))
    caption = pick_name(list(names))
    if caption is not None:
        names.update(e.get("alias"))
        return {"caption": caption, "schema": e.schema.name, "names": names}


class Aggregator:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.buffer: list[EntityProxy] = []

    def flush(self) -> None:
        rows = filter(bool, map(unpack_entity, self.buffer))
        df = pd.DataFrame(rows)
        df["names"] = df["names"].map(list)
        duckdb.register("df", df)
        self.table.execute("INSERT INTO names SELECT * FROM df")
        self.buffer = []

    def put(self, entity: EntityProxy) -> None:
        if not entity.get_type_values(registry.name):
            return
        self.buffer.append(entity)
        if len(self.buffer) >= 10_000:
            self.flush()

    def iterate(self) -> Docs:
        current_caption = None
        schemata: set[str] = set()
        names_: set[str] = set()
        res = self.table.execute("SELECT * FROM names ORDER BY caption")
        while rows := res.fetchmany(100_000):
            for caption, schema, names in rows:
                if current_caption is None:
                    current_caption = caption
                if current_caption != caption:
                    yield Doc(caption=current_caption, names=names_, schemata=schemata)
                    current_caption = caption
                    names_ = set()
                    schemata = set()
                schemata.add(schema)
                names_.update(names)

    def load_entities(self, entities: Iterable[EntityProxy]) -> None:
        with self:
            for entity in entities:
                self.put(entity)

    @property
    def table(self) -> duckdb.DuckDBPyConnection:
        return make_table(self.uri)

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
