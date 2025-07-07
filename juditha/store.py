import multiprocessing
from functools import cache, lru_cache
from typing import Generator, Self

import jellyfish
import tantivy
from anystore.io import logged_items
from anystore.logging import get_logger
from anystore.types import Uri
from anystore.util import join_uri, model_dump, path_from_uri, rm_rf
from rapidfuzz import process

from juditha.aggregator import Aggregator
from juditha.model import Doc, Result
from juditha.settings import Settings

NUM_CPU = multiprocessing.cpu_count()
INDEX = "tantivy.db"
NAMES = "names.db"

log = get_logger(__name__)
settings = Settings()


@cache
def make_schema() -> tantivy.Schema:
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("schema", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("caption", stored=True)
    schema_builder.add_text_field("names", stored=True)
    return schema_builder.build()


@cache
def ensure_db_path(uri: Uri) -> str:
    path = path_from_uri(uri) / NAMES
    path.parent.mkdir(exist_ok=True, parents=True)
    return str(path)


@cache
def ensure_index_path(uri: Uri) -> str:
    path = path_from_uri(uri) / INDEX
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


class Store:
    def __init__(self, uri: str | None):
        schema = make_schema()

        self.uri = uri or settings.uri

        if self.uri.startswith("memory"):
            self.index = tantivy.Index(schema)
            self.aggregator = Aggregator(":memory:")
        else:
            self.index = tantivy.Index(schema, ensure_index_path(self.uri))
            self.aggregator = Aggregator(ensure_db_path(self.uri))

        self.buffer: list[tantivy.Document] = []

        self.index.reload()
        log.info("👋", store=self.uri)

    def put(self, doc: Doc) -> None:
        self.buffer.append(tantivy.Document(**model_dump(doc)))
        if len(self.buffer) == 100_000:
            self.flush()

    def build(self) -> None:
        if not self.uri.startswith("memory"):
            uri = join_uri(self.uri, NAMES)
            log.info("Cleaning up outdated store ...", uri=uri)
            rm_rf(uri)
        with self as store:
            count = self.aggregator.count
            for doc in logged_items(
                self.aggregator, "Write", item_name="Doc", logger=log, total=count
            ):
                store.put(doc)

    def flush(self) -> None:
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
    settings = Settings()
    return Store(uri or settings.uri)


@lru_cache(100_000)
def lookup(
    q: str, threshold: float | None = None, uri: Uri | None = None
) -> Result | None:
    store = get_store(uri)
    return store.search(q, threshold)
