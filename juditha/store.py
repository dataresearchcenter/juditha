import multiprocessing
import os
from functools import cache, lru_cache
from typing import Generator, Self

import jellyfish
import tantivy
from anystore.util import model_dump
from followthemoney.proxy import EntityProxy
from followthemoney.types import registry
from pydantic import BaseModel, Field
from rapidfuzz import process

from juditha.logging import get_logger
from juditha.settings import Settings

NUM_CPU = multiprocessing.cpu_count()


log = get_logger(__name__)
settings = Settings()


@cache
def make_schema() -> tantivy.Schema:
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("schema", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("caption", stored=True)
    schema_builder.add_text_field("names", stored=True)
    return schema_builder.build()


class Doc(BaseModel):
    caption: str
    names: list[str] = []
    schema_: str = Field(alias="schema", default="")
    score: float = 0

    @classmethod
    def from_proxy(cls, proxy: EntityProxy) -> Self:
        return cls(
            caption=proxy.caption,
            names=proxy.get_type_values(registry.name),
            schema=proxy.schema.name,
        )


class Result(BaseModel):
    name: str
    query: str
    score: float
    schema_: str | None = Field(alias="schema", default=None)

    @classmethod
    def from_doc(cls, doc: Doc, q: str, score: float) -> Self:
        return cls(name=doc.caption, query=q, score=score, schema=doc.schema_ or None)


class Store:
    def __init__(self, uri: str | None):
        settings = Settings()
        schema = make_schema()

        uri = uri or settings.uri
        self.buffer: list[tantivy.Document] = []
        if uri.startswith("memory"):
            self.index = tantivy.Index(schema)
        else:
            os.makedirs(uri, exist_ok=True)
            self.index = tantivy.Index(schema, uri)

        self.uri = uri
        self.index.reload()
        log.info("👋", store=uri)

    def put(self, doc: Doc) -> None:
        self.buffer.append(tantivy.Document(**model_dump(doc)))
        if len(self.buffer) == 100_000:
            self.flush()

    def flush(self) -> None:
        log.info("Flushing %d items..." % len(self.buffer), store=self.uri)
        writer = self.index.writer(heap_size=15000000 * NUM_CPU, num_threads=NUM_CPU)
        for doc in self.buffer:
            writer.add_document(doc)
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()
        self.buffer = []

    def _search(
        self, q: str, query: tantivy.Query, limit: int, threshold: float
    ) -> Generator[Result, None, None]:
        searcher = self.index.searcher()
        result = searcher.search(query, limit)
        _q = q.lower()
        for item in result.hits:
            doc = searcher.doc(item[1])
            data = doc.to_dict()
            data["caption"] = doc.get_first("caption")
            data["schema"] = doc.get_first("schema")
            doc = Doc(**data)
            score = jellyfish.jaro_similarity(_q, doc.caption.lower())
            if score > threshold:
                yield Result.from_doc(doc, q, score)
            res = process.extractOne(_q, [n.lower() for n in doc.names])
            if res is not None:
                score = res[:2][1] / 100
                if score > threshold:
                    yield Result.from_doc(doc, q, score)

    def search(
        self, q: str, threshold: float | None = None, limit: int | None = None
    ) -> Result | None:
        threshold = threshold or settings.fuzzy_threshold
        limit = limit or settings.limit

        # 1. try exact
        query = self.index.parse_query(
            f'"{q}"',
            field_boosts={"caption": 2},
        )
        for res in self._search(q, query, limit, threshold):
            return res

        # 2. more fuzzy
        # FIXME seems not to work
        query = tantivy.Query.fuzzy_term_query(
            self.index.schema, "names", q, prefix=True
        )
        for res in self._search(q, query, limit, threshold):
            return res

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.flush()


@cache
def get_store(uri: str | None = None) -> Store:
    return Store(uri)


@lru_cache(100_000)
def lookup(q: str, threshold: float | None = None) -> Result | None:
    store = get_store()
    return store.search(q, threshold)
