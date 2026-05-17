"""
Aggregate names from entities to canonical name clusters in LevelDB.

Names are clustered by name_key() which produces order-independent,
accent-normalized keys with Q-ID resolution. The sparse vector index
handles all candidate retrieval; LevelDB only stores and iterates clusters.
"""

from typing import Iterable, Self

import plyvel
from anystore.logging import get_logger
from followthemoney import EntityProxy, registry
from ftmq.enums import Schemata
from rigour.names import pick_name

from juditha.model import Doc, Docs
from juditha.normalizer import name_key as n

log = get_logger(__name__)


NAMES = ("name", "previousName")


def unpack_entity(e: EntityProxy) -> dict | None:
    names: set[str] = set()
    for prop in NAMES:
        names.update(e.get(prop))
    caption = e.caption
    if caption in Schemata:
        caption = pick_name(list(names))
    if caption is not None and caption not in Schemata and len(caption) > 5:
        key = n(caption)
        if key:
            return {
                "key": key,
                "schema": e.schema.name,
                "names": names,
                "aliases": set(e.get("alias")),
                "countries": set(e.countries),
            }


class Aggregator:
    """
    LevelDB-based write-optimized storage for canonical name clusters.

    Key structure (null byte \\x00 delimiter, empty values):
        _d_\\x00<name_key>\\x00<schema>\\x00name\\x00<original_name>
        _d_\\x00<name_key>\\x00<schema>\\x00alias\\x00<original_alias>
        _d_\\x00<name_key>\\x00<schema>\\x00country\\x00<country_code>

    Benefits:
    - Automatic deduplication via LevelDB key overwriting
    - O(n) iteration for all docs, O(log n) prefix lookup for single doc
    """

    D = "\x00"  # Null byte delimiter
    EMPTY = b""
    PREFIX = "_d_"

    def __init__(self, uri: str) -> None:
        self.uri = str(uri)  # Convert Path to str if needed
        self.buffer: list[EntityProxy] = []
        # Open LevelDB (creates if doesn't exist)
        self.db = plyvel.DB(self.uri, create_if_missing=True)

    def make_key(self, *parts: str) -> bytes:
        return self.D.join(parts).encode()

    def flush(self) -> None:
        """Flush buffered entities to LevelDB."""
        with self.db.write_batch() as db:
            for entity in self.buffer:
                row = unpack_entity(entity)
                if row is None:
                    continue

                norm_key = row["key"]
                schema = row["schema"]

                for name in row["names"]:
                    key = self.make_key(self.PREFIX, norm_key, schema, "name", name)
                    db.put(key, self.EMPTY)

                for alias in row["aliases"]:
                    key = self.make_key(self.PREFIX, norm_key, schema, "alias", alias)
                    db.put(key, self.EMPTY)

                for country in row["countries"]:
                    country = country.lower()
                    key = self.make_key(
                        self.PREFIX, norm_key, schema, "country", country
                    )
                    db.put(key, self.EMPTY)

        self.buffer = []

    def put(self, entity: EntityProxy) -> None:
        """Add entity to buffer."""
        if not entity.get_type_values(registry.name):
            return
        self.buffer.append(entity)
        if len(self.buffer) >= 10_000:
            self.flush()

    def iterate(self, norm_key: str | None = None) -> Docs:
        """
        Iterate over aggregated docs.

        Args:
            norm_key: Optional key to filter by. If None, iterates all docs.
        """
        current_key = None
        schemata: set[str] = set()
        names_: set[str] = set()
        aliases_: set[str] = set()
        countries_: set[str] = set()

        # Build prefix for iteration
        if norm_key:
            # Iterate only keys for specific norm
            prefix = f"{self.PREFIX}{self.D}{norm_key}{self.D}".encode("utf-8")
        else:
            # Iterate all main index keys
            prefix = f"{self.PREFIX}{self.D}".encode("utf-8")

        # LevelDB stores keys in sorted order, so we can group by norm
        # Key format: _d_\x00<norm>\x00<schema>\x00type\x00value
        for key, _ in self.db.iterator(prefix=prefix):
            # Decode key
            key_str = key.decode("utf-8")

            # Parse main key: _main_\x00<norm>\x00<schema>\x00type\x00value
            _, norm, schema, entry_type, entry_value = key_str.split(self.D)

            # If we're filtering by norm and this doesn't match, stop
            if norm_key and norm != norm_key:
                break

            # Check if we've moved to a new norm group
            if current_key is None:
                current_key = norm

            if current_key != norm:
                # Yield the accumulated doc for previous norm
                yield Doc(
                    key=current_key,
                    names=names_,
                    aliases=aliases_,
                    countries=countries_,
                    schemata=schemata,
                )
                # Reset for new norm group
                current_key = norm
                names_ = set()
                aliases_ = set()
                countries_ = set()
                schemata = set()

            # Accumulate data for current norm
            schemata.add(schema)

            # Add name, alias, or country based on type
            if entry_type == "name":
                names_.add(entry_value)
            elif entry_type == "alias":
                aliases_.add(entry_value)
            elif entry_type == "country":
                countries_.add(entry_value)

        # Don't forget the last (or only) one
        if current_key:
            yield Doc(
                key=current_key,
                names=names_,
                aliases=aliases_,
                countries=countries_,
                schemata=schemata,
            )

    def load_entities(self, entities: Iterable[EntityProxy]) -> None:
        with self:
            for entity in entities:
                self.put(entity)

    def get_doc(self, norm_key: str) -> Doc | None:
        """Retrieve a Doc by norm_key via prefix iteration."""
        for doc in self.iterate(norm_key=norm_key):
            return doc

    @property
    def count(self) -> int:
        """Count distinct name_keys (aggregated docs)."""
        seen = set()
        prefix = f"{self.PREFIX}{self.D}".encode("utf-8")
        for key, _ in self.db.iterator(prefix=prefix):
            key_str = key.decode("utf-8")
            parts = key_str.split(self.D)
            if len(parts) >= 2:
                seen.add(parts[1])
        return len(seen)

    def __iter__(self) -> Docs:
        yield from self.iterate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        self.flush()
        # Don't close DB - keep it open for subsequent use via cached Index

    def close(self) -> None:
        """Close the LevelDB connection."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def __del__(self) -> None:
        """Ensure DB is closed when object is garbage collected"""
        self.close()
