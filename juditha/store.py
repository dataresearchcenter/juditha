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

from juditha.aggregator import Aggregator
from juditha.model import NER_TAG, Doc, Result, name_key
from juditha.settings import Settings
from juditha.validate import Validator

NUM_CPU = multiprocessing.cpu_count()
INDEX = "tantivy.db"
NAMES = "names.db"
TOKENS = "tokens"

log = get_logger(__name__)
settings = Settings()


@cache
def make_schema() -> tantivy.Schema:
    schema_builder = tantivy.SchemaBuilder()
    # Use raw tokenizer for names so they're indexed as complete strings for fuzzy matching
    schema_builder.add_text_field("key", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("schemata", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("names", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("aliases", tokenizer_name="raw", stored=True)
    schema_builder.add_text_field("countries", tokenizer_name="raw", stored=True)
    return schema_builder.build()


def ensure_db_path(uri: Uri) -> str:
    path = path_from_uri(uri) / NAMES
    path.parent.mkdir(exist_ok=True, parents=True)
    return str(path)


def ensure_index_path(uri: Uri) -> str:
    path = path_from_uri(uri) / INDEX
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


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
            self.index = tantivy.Index(make_schema(), ensure_index_path(self.uri))
        with self as store:
            count = self.aggregator.count
            for doc in logged_items(
                self.aggregator, "Indexing", item_name="Doc", logger=log, total=count
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
        self,
        q: str,
        clean_q: str,
        query: tantivy.Query,
        limit: int,
        threshold: float,
        schemata: Iterable[str] | None = None,
    ) -> Generator[Result, None, None]:
        searcher = self.index.searcher()
        result = searcher.search(query, limit)
        docs: list[Doc] = []
        for item in result.hits:
            tdoc = searcher.doc(item[1])
            data = tdoc.to_dict()

            # Convert lists to sets and handle fields properly
            doc = Doc(
                key=tdoc.get_first("key") or "",
                names=set(data.get("names", [])),
                aliases=set(data.get("aliases", [])),
                countries=set(data.get("countries", [])),
                schemata=set(data.get("schemata", [])),
            )

            # Schema filtering: skip docs that don't match requested schemata
            if schemata and not doc.schemata.intersection(schemata):
                continue

            # Compare normalized forms for scoring
            score = jellyfish.jaro_similarity(clean_q, doc.key)
            if score > threshold:
                yield Result.from_doc(doc, q, score)
            else:
                docs.append(doc)
        # now try other names
        for doc in docs:
            res = process.extractOne(clean_q, [name_key(n) for n in doc.names])
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
        threshold = threshold if threshold is not None else settings.fuzzy_threshold
        limit = limit if limit is not None else settings.limit
        clean_q = name_key(q)
        if not clean_q or len(clean_q) < settings.min_length:
            return

        # Build fuzzy query for the complete name with field boosting
        # key field boosted 3x, names 2x, aliases 1x
        field_boosts = {"key": 3.0, "names": 2.0, "aliases": 1.0}

        # Create fuzzy query for each field matching the complete name
        field_queries = []
        for field, boost in field_boosts.items():
            fuzzy_q = tantivy.Query.fuzzy_term_query(
                self.index.schema,
                field,
                clean_q,
                distance=2,
                transposition_cost_one=True,
                prefix=False,
            )
            # Boost the field query
            boosted_q = tantivy.Query.boost_query(fuzzy_q, boost)
            field_queries.append((tantivy.Occur.Should, boosted_q))

        # Combine all field queries with OR - Tantivy's BM25 will rank by relevance
        query = tantivy.Query.boolean_query(field_queries)

        for res in self._search(q, clean_q, query, limit, threshold, schemata):
            return res

    def validate(self, name: str, tag: NER_TAG) -> bool:
        return self.validator.validate_name(name, tag)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
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
    store = get_store(uri) if uri is not None else get_store()
    return store.search(q, threshold, schemata=set(schemata) if schemata else None)


@lru_cache(100_000)
def validate_name(name: str, tag: NER_TAG, uri: Uri | None = None) -> bool:
    store = get_store(uri) if uri is not None else get_store()
    return store.validate(name, tag)
