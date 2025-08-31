import multiprocessing
from functools import cache, lru_cache
from typing import Generator, Iterable, Self

import jellyfish
import tantivy
from anystore.decorators import error_handler
from anystore.io import logged_items
from anystore.logging import get_logger
from anystore.types import Uri
from anystore.util import join_uri, model_dump, path_from_uri, rm_rf
from rapidfuzz import process
from rigour.names import Name

from juditha.aggregator import Aggregator
from juditha.model import NER_TAG, Doc, Result
from juditha.settings import Settings
from juditha.validate import Validator

NUM_CPU = multiprocessing.cpu_count()
INDEX = "tantivy.db"
NAMES = "names.db"
TOKENS = "tokens"

log = get_logger(__name__)
settings = Settings()


def clean_name(name: str) -> str:
    n = Name(name)
    return n.norm_form


@cache
def make_schema() -> tantivy.Schema:
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("schemata", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("caption", stored=True)
    schema_builder.add_text_field("names", stored=True)
    schema_builder.add_text_field("aliases", stored=True)
    schema_builder.add_text_field("countries", tokenizer_name="raw", stored=True)
    schema_builder.add_json_field("symbols", tokenizer_name="raw", stored=True)
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


@cache
def ensure_tokens_path(uri: Uri) -> str:
    path = path_from_uri(uri) / TOKENS
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


class Store:
    def __init__(self, uri: str | None):
        schema = make_schema()

        self.uri = uri or settings.uri

        if self.uri.startswith("memory"):
            self.index = tantivy.Index(schema)
            self.aggregator = Aggregator(":memory:")
            self.validator = Validator("memory://", self.aggregator)
        else:
            self.index = tantivy.Index(schema, ensure_index_path(self.uri))
            self.aggregator = Aggregator(ensure_db_path(self.uri))
            self.validator = Validator(ensure_tokens_path(self.uri), self.aggregator)

        self.buffer: list[tantivy.Document] = []

        self.index.reload()
        log.info("👋", store=self.uri)

    def put(self, doc: Doc) -> None:
        self.buffer.append(tantivy.Document(**model_dump(doc)))
        if len(self.buffer) == 100_000:
            self.flush()

    def build(self) -> None:
        if not self.uri.startswith("memory"):
            uri = join_uri(self.uri, INDEX)
            log.info("Cleaning up outdated store ...", uri=uri)
            rm_rf(uri)
        with self as store:
            count = self.aggregator.count
            for doc in logged_items(
                self.aggregator, "Write", item_name="Doc", logger=log, total=count
            ):
                store.put(doc)
        # build validation tokens
        _ = self.validator.get_tokens()

    def flush(self) -> None:
        writer = self.index.writer(heap_size=15000000 * NUM_CPU, num_threads=NUM_CPU)
        for doc in self.buffer:
            writer.add_document(doc)
        writer.commit()
        writer.wait_merging_threads()
        self.index.reload()
        self.buffer = []

    def _search(
        self, q: str, clean_q: str, query: tantivy.Query, limit: int, threshold: float
    ) -> Generator[Result, None, None]:
        searcher = self.index.searcher()
        result = searcher.search(query, limit)
        docs: list[Doc] = []
        for item in result.hits:
            doc = searcher.doc(item[1])
            data = doc.to_dict()
            data["caption"] = doc.get_first("caption")
            data["schema"] = doc.get_first("schema")
            doc = Doc(**data)
            score = jellyfish.jaro_similarity(clean_q, doc.caption.lower())
            if score > threshold:
                yield Result.from_doc(doc, q, score)
            else:
                docs.append(doc)
        # now try other names
        for doc in docs:
            res = process.extractOne(clean_q, [n.lower() for n in doc.names])
            if res is not None:
                score = res[:2][1] / 100
                if score > threshold:
                    yield Result.from_doc(doc, q, score)

    @error_handler(max_retries=0)
    def search(
        self,
        q: str,
        threshold: float | None = None,
        limit: int | None = None,
        schemata: Iterable[str] | None = None,
    ) -> Result | None:
        threshold = threshold or settings.fuzzy_threshold
        limit = limit or settings.limit
        clean_q = clean_name(q)
        if not clean_q or len(clean_q) < settings.min_length:
            return

        # Build schema filter if provided
        schema_filter = ""
        if schemata:
            schema_terms = " OR ".join(f'schemata:"{schema}"' for schema in schemata)
            schema_filter = f" AND ({schema_terms})"

        # 1. try exact caption
        query_str = f'"{clean_q}"{schema_filter}'
        query = self.index.parse_query(query_str, ["caption"])
        for res in self._search(q, clean_q, query, limit, threshold):
            return res

        # 2. lookup other names
        query_str = f'"{clean_q}"{schema_filter}'
        query = self.index.parse_query(query_str, ["names", "aliases"])
        for res in self._search(q, clean_q, query, limit, threshold):
            return res

        # 3. more fuzzy - broad term search with OR  FIXME doesn't seem to work
        terms = clean_q.split()
        terms = " OR ".join(f"{term}~1" for term in terms)
        query_str = f"{terms}{schema_filter}"
        query = self.index.parse_query(query_str, ["caption", "names", "aliases"])
        for res in self._search(q, clean_q, query, limit, threshold):
            return res

    def validate(self, name: str, tag: NER_TAG) -> bool:
        return self.validator.validate_name(name, tag)

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
    q: str,
    threshold: float | None = None,
    uri: Uri | None = None,
    schemata: tuple[str, ...] | None = None,
) -> Result | None:
    store = get_store(uri)
    return store.search(q, threshold, schemata=set(schemata) if schemata else None)


@lru_cache(100_000)
def validate_name(name: str, tag: NER_TAG, uri: Uri | None = None) -> bool:
    store = get_store(uri)
    return store.validate(name, tag)
