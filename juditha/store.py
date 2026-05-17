"""Tantivy-backed name lookup store.

Replaces the previous sparse-matrix index. The schema uses
`tokenizer_name="raw"` on every name field so multi-word names index as a
single term — FuzzyTermQuery then matches against the whole normalized
name without splitting on whitespace.

Query shape (per refactor plan):
    BooleanQuery:
      should: fuzzy(key,     clean_q, d=2, transp=1) * 3.0
      should: fuzzy(names,   clean_q, d=2, transp=1) * 2.0
      should: fuzzy(aliases, clean_q, d=2, transp=1) * 1.0
      should: bool(should=[term(phonetic, c) for c in metaphone(clean_q)]) * 0.5
      [must: symbols ⊇ extracted_symbols]
      [must: qid = qid_hint]
      [must: schemata ∈ requested_schemata]

Rerank: tantivy returns top-K → jaro on hit.key → rapidfuzz fallback over
hit.names. Phonetic-only hits that are semantically wrong get filtered by
the rerank stage.
"""

import multiprocessing
import time
from functools import cache, lru_cache
from typing import Iterable, Self

import jellyfish
import tantivy
from anystore.io import logged_items
from anystore.logging import get_logger
from anystore.types import Uri
from anystore.util import join_uri, path_from_uri, rm_rf
from rapidfuzz import process
from rigour.names import Name, NameTypeTag, SymbolCategory, analyze_names

from juditha.aggregator import Aggregator
from juditha.extraction import AhoExtractor
from juditha.model import Doc, Mention, Result
from juditha.normalizer import name_key, tokenize_forms
from juditha.percolator import MIN_TOKEN_LENGTH, percolate
from juditha.settings import Settings

NUM_CPU = multiprocessing.cpu_count()
INDEX = "tantivy.db"
NAMES = "names.db"
AHO = "automaton.txt"


log = get_logger(__name__)
settings = Settings()


@cache
def make_schema() -> tantivy.Schema:
    """Build the tantivy schema for name lookup.

    Every name field uses `tokenizer_name="raw"` so multi-word names are
    indexed as a single term — that's what makes FuzzyTermQuery work
    across the whole name without splitting on whitespace.
    """
    b = tantivy.SchemaBuilder()
    # primary normalized form — fuzzy target, BM25 anchor
    b.add_text_field("key", tokenizer_name="raw", stored=True)
    # surface forms (multi-value)
    b.add_text_field("names", tokenizer_name="raw", stored=True)
    b.add_text_field("aliases", tokenizer_name="raw", stored=True)
    # narrowing / pushdown filters
    b.add_text_field("schemata", tokenizer_name="raw", stored=True)
    b.add_text_field("countries", tokenizer_name="raw", stored=True)
    b.add_text_field("qid", tokenizer_name="raw", stored=True)
    b.add_text_field("symbols", tokenizer_name="raw", stored=True)
    # phonetic blocking for names unreachable by fuzzy distance 2;
    # index-only, not stored
    b.add_text_field("phonetic", tokenizer_name="raw", stored=False)
    # percolator blocking: per-token inverted index over names + aliases
    b.add_text_field("tokens", tokenizer_name="raw", stored=False)
    return b.build()


def ensure_db_path(uri: Uri) -> str:
    path = path_from_uri(uri) / NAMES
    path.parent.mkdir(exist_ok=True, parents=True)
    return str(path)


def ensure_index_path(uri: Uri) -> str:
    path = path_from_uri(uri) / INDEX
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


# ----- Index-time feature extractors (rigour 2.x) -----


@lru_cache(maxsize=100_000)
def _phonetic_codes(name: str) -> tuple[str, ...]:
    """Dedup metaphone codes per name part, for phonetic blocking.

    Cached because the same names recur across docs/aliases and across
    percolate calls; rigour's `Name(name).parts` parse is the cost.
    """
    codes: set[str] = set()
    for p in Name(name).parts:
        if p.metaphone:
            codes.add(p.metaphone)
    return tuple(sorted(codes))


@lru_cache(maxsize=100_000)
def _name_features(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Run rigour analyze_names and split symbols into (qids, other_symbols).

    NAME-category symbols (rigour's person-name cluster IDs) go to `qids`;
    everything else (ORG_CLASS, NICK, SYMBOL, DOMAIN, LOCATION, …) goes to
    `symbols` as f"{category.name}:{symbol.id}" so categories don't collide.

    Cached: rigour runs both PER and ORG analyzers on the name (two
    Rust round-trips). With 10M-name corpora the same surface forms
    recur often enough to make memoization worthwhile.
    """
    qids: set[str] = set()
    symbols: set[str] = set()
    for type_tag in (NameTypeTag.ORG, NameTypeTag.PER):
        for n in analyze_names(type_tag, [name]):
            for s in n.symbols:
                if s.category == SymbolCategory.NAME:
                    qids.add(s.id)
                else:
                    # SymbolCategory is rigour's Rust enum; .value is the
                    # short label (e.g. "ORG_CLASS", "SYMBOL", "NICK").
                    symbols.add(f"{s.category.value}:{s.id}")
    return tuple(sorted(qids)), tuple(sorted(symbols))


# ----- Query construction -----


def build_query(
    schema: tantivy.Schema,
    clean_q: str,
    *,
    symbols: set[str] | None = None,
    qid_hint: str | None = None,
    schemata: set[str] | None = None,
    phonetic_codes: Iterable[str] | None = None,
) -> tantivy.Query:
    """Build the BooleanQuery per the refactor plan."""
    Occur = tantivy.Occur
    Q = tantivy.Query

    clauses: list[tuple[tantivy.Occur, tantivy.Query]] = []

    # tier 1: fuzzy across name fields with juditha's boost scheme
    for field, boost in (("key", 3.0), ("names", 2.0), ("aliases", 1.0)):
        fq = Q.fuzzy_term_query(
            schema,
            field,
            clean_q,
            distance=2,
            transposition_cost_one=True,
            prefix=False,
        )
        clauses.append((Occur.Should, Q.boost_query(fq, boost)))

    # tier 2: phonetic blocking — OR across tokens at lower boost
    codes = [c for c in (phonetic_codes or []) if c]
    if codes:
        phon = Q.boolean_query(
            [(Occur.Should, Q.term_query(schema, "phonetic", c)) for c in codes]
        )
        clauses.append((Occur.Should, Q.boost_query(phon, 0.5)))

    # narrowing (hard filters via posting-list intersection)
    if symbols:
        sym = Q.boolean_query(
            [(Occur.Should, Q.term_query(schema, "symbols", s)) for s in symbols]
        )
        clauses.append((Occur.Must, sym))

    if qid_hint:
        clauses.append((Occur.Must, Q.term_query(schema, "qid", qid_hint)))

    if schemata:
        sch = Q.boolean_query(
            [(Occur.Should, Q.term_query(schema, "schemata", s)) for s in schemata]
        )
        clauses.append((Occur.Must, sch))

    return Q.boolean_query(clauses)


class Store:
    def __init__(self, uri: str | None):
        self.uri = uri or settings.uri
        self._schema = make_schema()
        # Tantivy mmap on disk so the OS page cache is shared across all
        # procrastinate workers on the host (memory footprint is 1×, not N×).
        self.index = tantivy.Index(self._schema, ensure_index_path(self.uri))
        self.aggregator = Aggregator(ensure_db_path(self.uri))

        self.buffer: list[tantivy.Document] = []
        self._extractor: AhoExtractor | None = None
        # Lazy persistent writer: tantivy IndexWriter is heavy
        # (heap_size × num_threads bytes); reuse across flush() calls
        # instead of allocating one per batch.
        self._writer: tantivy.IndexWriter | None = None
        self.index.reload()
        log.info("👋", store=self.uri)

    def _get_writer(self) -> tantivy.IndexWriter:
        if self._writer is None:
            self._writer = self.index.writer(
                heap_size=15_000_000 * NUM_CPU, num_threads=NUM_CPU
            )
        return self._writer

    def _release_writer(self) -> None:
        """Wait for in-flight merges and drop the writer reference."""
        if self._writer is not None:
            self._writer.wait_merging_threads()
            self._writer = None

    @property
    def extractor(self) -> AhoExtractor:
        if self._extractor is None:
            self._extractor = AhoExtractor()
            aho_path = path_from_uri(self.uri) / AHO
            self._extractor.load(aho_path)
        return self._extractor

    def put(self, doc: Doc) -> None:
        """Buffer a Doc for tantivy indexing; flushes every 100k."""
        names_all = doc.names | doc.aliases
        qids: set[str] = set()
        symbols: set[str] = set()
        phonetic: set[str] = set()
        # tokens powers the percolator blocker — every normalized token
        # ≥ MIN_TOKEN_LENGTH chars across names + aliases.
        tokens: set[str] = set()
        for n in names_all:
            q_n, s_n = _name_features(n)
            qids.update(q_n)
            symbols.update(s_n)
            phonetic.update(_phonetic_codes(n))
            for t in tokenize_forms(n):
                if len(t) >= MIN_TOKEN_LENGTH:
                    tokens.add(t)

        fields: dict[str, str | list[str]] = {
            "key": doc.key,
            "names": sorted(doc.names),
            "aliases": sorted(doc.aliases),
            "schemata": sorted(doc.schemata),
            "countries": sorted(doc.countries),
            "qid": sorted(qids),
            "symbols": sorted(symbols),
            "phonetic": sorted(phonetic),
            "tokens": sorted(tokens),
        }
        self.buffer.append(tantivy.Document(**fields))
        if len(self.buffer) >= 100_000:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        writer = self._get_writer()
        for d in self.buffer:
            writer.add_document(d)
        writer.commit()
        # NOTE: wait_merging_threads() intentionally NOT called here —
        # background merges can run while we accumulate the next batch.
        # close() drains them at shutdown.
        self.index.reload()
        self.buffer = []

    def build(self) -> None:
        """Rebuild tantivy index + AhoExtractor from the aggregator.

        Single iteration through the aggregator — both consumers see each
        doc as it streams through.
        """
        uri = join_uri(self.uri, INDEX)
        log.info("Cleaning up outdated store ...", uri=uri)
        # The persistent writer (if any) is tied to the about-to-be-deleted
        # index. Drain its merges and drop it before recreating the index.
        self._release_writer()
        rm_rf(uri)
        self.index = tantivy.Index(self._schema, ensure_index_path(self.uri))

        self._extractor = AhoExtractor()
        count = self.aggregator.count
        with self as store:
            for j, doc in enumerate(
                logged_items(
                    self.aggregator,
                    "Indexing",
                    item_name="Doc",
                    logger=log,
                    total=count,
                )
            ):
                store.put(doc)
                self._extractor.add_doc(j, doc)

        # Drain merges from the build so the index is fully quiescent
        # before subsequent searches.
        self._release_writer()
        self._extractor.finalize()
        aho_path = path_from_uri(self.uri) / AHO
        self._extractor.save(aho_path)

    def search(
        self,
        q: str,
        threshold: float | None = None,
        limit: int | None = None,
        schemata: Iterable[str] | None = None,
    ) -> Result | None:
        t0 = time.perf_counter()
        threshold = threshold if threshold is not None else settings.fuzzy_threshold
        limit = limit if limit is not None else settings.limit
        clean_q = name_key(q)
        if not clean_q or len(clean_q) < settings.min_length:
            return None

        schemata_set = set(schemata) if schemata else None
        phon_codes = _phonetic_codes(q)
        query = build_query(
            self._schema,
            clean_q,
            schemata=schemata_set,
            phonetic_codes=phon_codes,
        )

        searcher = self.index.searcher()
        hits = searcher.search(query, limit).hits

        deferred: list[Doc] = []
        for _, addr in hits:
            tdoc = searcher.doc(addr)
            data = tdoc.to_dict()
            doc = Doc(
                key=tdoc.get_first("key") or "",
                names=set(data.get("names", [])),
                aliases=set(data.get("aliases", [])),
                countries=set(data.get("countries", [])),
                schemata=set(data.get("schemata", [])),
            )
            score = jellyfish.jaro_similarity(clean_q, doc.key)
            if score > threshold:
                took = (time.perf_counter() - t0) * 1000
                return Result.from_doc(doc, q, score, took=took)
            deferred.append(doc)

        # Second pass: rapidfuzz over each hit's full surface-form set, to
        # catch alias-driven matches that don't rank on `key`.
        for doc in deferred:
            res = process.extractOne(clean_q, [name_key(n) for n in doc.names])
            if res is not None:
                score = res[1] / 100
                if score > threshold:
                    took = (time.perf_counter() - t0) * 1000
                    return Result.from_doc(doc, q, score, took=took)
        return None

    def extract(self, text: str) -> list[Mention]:
        return self.extractor.extract(text)

    def percolate(self, text: str, slop: int = 0) -> list[Mention]:
        """Reverse-search the names index for mentions in `text`.

        Thin wrapper around `juditha.percolator.percolate`. See that
        module for the algorithm.
        """
        return percolate(self._schema, self.index, text, slop=slop)

    def close(self) -> None:
        """Flush pending writes and release tantivy writer + LevelDB handles.

        Call this in one-shot scripts that want a deterministic shutdown
        (the cached Store registry otherwise keeps refs alive until
        process exit). Long-running workers don't need to call it
        explicitly.
        """
        self.flush()
        self._release_writer()
        self.aggregator.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        # Context-manager exit flushes pending writes but does NOT close —
        # build() uses `with self as store:` and we still want lookup /
        # percolate to work afterwards.
        self.flush()


@cache
def _store_for_uri(uri: str) -> Store:
    """Cache one Store per resolved URI.

    plyvel only allows one open handle per LevelDB path, so the registry
    is effectively a per-URI singleton.
    """
    return Store(uri)


def get_store(uri: Uri | None = None) -> Store:
    """Return the cached Store for `uri` (or for the env-resolved URI).

    Unlike a direct `@cache` decorator, this resolves None against the
    current `Settings()` *at call time*, so changes to `JUDITHA_URI`
    between calls pick up the new URI without needing `cache_clear()`.
    """
    return _store_for_uri(str(uri) if uri is not None else Settings().uri)


# Backward-compat alias for code (and tests) that calls
# `get_store.cache_clear()`. Clears the underlying per-URI registry.
get_store.cache_clear = _store_for_uri.cache_clear  # type: ignore[attr-defined]


@lru_cache(100_000)
def lookup(
    q: str,
    threshold: float | None = None,
    uri: Uri | None = None,
    schemata: tuple[str, ...] | None = None,
) -> Result | None:
    store = get_store(uri) if uri is not None else get_store()
    return store.search(q, threshold, schemata=set(schemata) if schemata else None)
