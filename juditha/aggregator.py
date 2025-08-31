"""
Aggregate names from entities to caption clusters

This is needed as we could index multiple entities with similar names and we
want to canonize them all.
"""

from functools import cache
from typing import Generator, Iterable, Self, TypedDict

import duckdb
import pandas as pd
from anystore.logging import get_logger
from anystore.types import StrGenerator
from followthemoney import EntityProxy, registry
from ftmq.enums import Schemata
from ftmq.util import get_symbols, select_symbols
from rigour.names import pick_name

from juditha.model import Doc, Docs

log = get_logger(__name__)


@cache
def make_table(uri: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(uri)
    con.sql(
        """CREATE TABLE IF NOT EXISTS names (
            caption STRING,
            schema STRING,
            names STRING[],
            aliases STRING[],
            countries STRING[],
            symbols STRING[]
            )"""
    )
    return con


class Row(TypedDict):
    caption: str
    names: set[str]
    aliases: set[str]
    countries: set[str]
    symbols: set[str]
    schema: str


def unpack_entity(e: EntityProxy) -> Row | None:
    names = set(e.get("name"))
    caption = e.caption
    if caption in Schemata:
        caption = pick_name(list(names))
    if caption is not None and caption not in Schemata and len(caption) > 5:
        symbols = select_symbols(e)
        if not symbols:
            symbols = get_symbols(e)
        return {
            "caption": caption,
            "schema": e.schema.name,
            "names": names,
            "aliases": set(e.get("alias")),
            "countries": set(e.countries),
            "symbols": {str(s) for s in symbols},
        }


class Aggregator:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.buffer: list[EntityProxy] = []

    def flush(self) -> None:
        rows = filter(bool, map(unpack_entity, self.buffer))
        df = pd.DataFrame(rows)
        df["names"] = df["names"].map(list)
        df["aliases"] = df["aliases"].map(list)
        df["countries"] = df["countries"].map(list)
        df["symbols"] = df["symbols"].map(list)
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
        aliases_: set[str] = set()
        countries_: set[str] = set()
        symbols_: set[str] = set()
        res = self.table.execute("SELECT * FROM names ORDER BY caption")
        while rows := res.fetchmany(100_000):
            for caption, schema, names, aliases, countries, symbols in rows:
                if current_caption is None:
                    current_caption = caption
                if current_caption != caption:
                    yield Doc(
                        caption=current_caption,
                        names=names_,
                        aliases=aliases_,
                        countries=countries_,
                        schemata=schemata,
                        symbols=symbols_,
                    )
                    current_caption = caption
                    names_ = set()
                    aliases_ = set()
                    countries_ = set()
                    symbols_ = set()
                    schemata = set()
                schemata.add(schema)
                names_.update(names)
                aliases_.update(aliases)
                countries_.update(countries)
                symbols_.update(symbols)
        # don't forget the last (or only) one
        if current_caption:
            yield Doc(
                caption=current_caption,
                names=names_,
                aliases=aliases_,
                countries=countries_,
                schemata=schemata,
                symbols=symbols_,
            )

    def load_entities(self, entities: Iterable[EntityProxy]) -> None:
        with self:
            for entity in entities:
                self.put(entity)

    def iter_names(self) -> StrGenerator:
        res = self.table.execute("SELECT names FROM names")
        while rows := res.fetchmany(100_000):
            for (names,) in rows:
                yield from names

    def iter_names_schema(self) -> Generator[tuple[str, str], None, None]:
        res = self.table.execute("SELECT schema, names FROM names")
        while rows := res.fetchmany(100_000):
            for schema, names in rows:
                for name in names:
                    yield name, schema

    @property
    def table(self) -> duckdb.DuckDBPyConnection:
        return make_table(self.uri)

    @property
    def count(self) -> int:
        for (c,) in self.table.sql(
            "SELECT COUNT(DISTINCT caption) FROM names"
        ).fetchall():
            return c
        return 0

    @property
    def count_rows(self) -> int:
        for (c,) in self.table.sql("SELECT COUNT(*) FROM names").fetchall():
            return c
        return 0

    def __iter__(self) -> Docs:
        yield from self.iterate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.flush()
