"""
Aggregate names from entities to caption's norm_form clusters

This is needed as we could index multiple entities with similar names and we
want to canonize them all.
"""

from typing import Generator, Iterable, Self, TypedDict

import plyvel
from anystore.logging import get_logger
from followthemoney import EntityProxy, registry
from ftmq.enums import Schemata
from rigour.names import pick_name

from juditha.model import Doc, Docs
from juditha.model import name_key as n
from juditha.model import schema_to_ner

log = get_logger(__name__)


NAMES = ("name", "previousName")


class Row(TypedDict):
    key: str
    schema: str
    names: set[str]
    aliases: set[str]
    countries: set[str]


def unpack_entity(e: EntityProxy) -> Row | None:
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
    Aggregator using LevelDB for fast write-optimized storage.

    Storage format (using null byte \x00 as delimiter):

    Main index (all data in keys, empty values):
    - _main_\x00<key>\x00<schema>\x00name\x00<name> → b''
    - _main_\x00<key>\x00<schema>\x00alias\x00<alias> → b''
    - _main_\x00<key>\x00<schema>\x00country\x00<country> → b''

    Inverted indexes (for fast lookup):
    - _idx_name\x00<name>\x00<key> → b''
    - _idx_alias\x00<alias>\x00<key> → b''

    Filter indexes (for filtering):
    - _idx_ner\x00<ner_tag>\x00name\x00<name>\x00<key> → b''
    - _idx_ner\x00<ner_tag>\x00alias\x00<alias>\x00<key> → b''
    - _idx_country\x00<country>\x00<key> → b''

    Example:
    Main:
    - _main_\x00john_smith\x00John Smith\x00Person\x00name\x00John Smith → b''
    - _main_\x00john_smith\x00John Smith\x00Person\x00alias\x00Johnny → b''
    - _main_\x00john_smith\x00John Smith\x00Person\x00country\x00US → b''

    Inverted:
    - _idx_name\x00john smith\x00john_smith → b''
    - _idx_alias\x00johnny\x00john_smith → b''

    Filter:
    - _idx_ner\x00PER\x00name\x00john smith\x00john_smith → b''
    - _idx_country\x00US\x00john_smith → b''

    Benefits:
    - Automatic deduplication: LevelDB overwrites duplicate keys
    - Fast iteration: O(n) to reconstruct all docs, O(log n) for single doc via prefix
    - Fast filtering: O(log n) lookups via filter indexes
    - All data in keys: No JSON parsing needed
    """

    D = "\x00"  # Null byte delimiter
    EMPTY = b""
    MAIN = "_main_"
    IDX_NAME = "_idx_name"
    IDX_ALIAS = "_idx_alias"
    IDX_NER = "_idx_ner"
    IDX_COUNTRY = "_idx_country"

    def __init__(self, uri: str) -> None:
        self.uri = str(uri)  # Convert Path to str if needed
        self.buffer: list[EntityProxy] = []

        # Open LevelDB (creates if doesn't exist)
        if self.uri == ":memory:":
            # In-memory mode not supported by LevelDB, use temp file
            import tempfile

            self._temp_dir = tempfile.mkdtemp()
            self.db = plyvel.DB(self._temp_dir, create_if_missing=True)
        else:
            self.db = plyvel.DB(self.uri, create_if_missing=True)

    def make_key(self, *parts: str) -> bytes:
        return self.D.join(parts).encode()

    def flush(self) -> None:
        """
        Flush buffered entities to LevelDB.

        Creates all indexes:
        - Main index: <key>:<schema>:name/alias/country:<value>
        - Inverted indexes: _idx_name/alias:<value>:<key>
        - Filter indexes: _idx_ner:<tag>:name/alias:<value>:<key>, _idx_country:<country>:<key>
        - All values are empty (automatic deduplication via key overwriting)
        """
        with self.db.write_batch() as db:
            for entity in self.buffer:
                row = unpack_entity(entity)
                if row is None:
                    continue

                norm_key = row["key"]
                schema = row["schema"]
                ner_tag = schema_to_ner(schema)

                # Create indexes for each name
                for name in row["names"]:
                    norm_name = n(name)
                    key = self.make_key(self.MAIN, norm_key, schema, "name", name)
                    db.put(key, self.EMPTY)

                    # Inverted index: _idx_name:<norm_name>:<norm> (uses cleaned
                    # name for lookups)
                    key = self.make_key(self.IDX_NAME, norm_name, norm_key)
                    db.put(key, self.EMPTY)

                    # NER filter: _idx_ner:<ner_tag>:name:<norm_name>:<norm>
                    key = self.make_key(
                        self.IDX_NER, ner_tag, "name", norm_name, norm_key
                    )
                    db.put(key, self.EMPTY)

                # Create indexes for each alias
                for alias in row["aliases"]:
                    norm_alias = n(alias)
                    # Main index (uses original alias)
                    key = self.make_key(self.MAIN, norm_key, schema, "alias", alias)
                    db.put(key, self.EMPTY)

                    # Inverted index: _idx_alias:<norm_alias>:<norm> (uses
                    # cleaned alias for lookups)
                    key = self.make_key(self.IDX_ALIAS, norm_alias, norm_key)
                    db.put(key, self.EMPTY)

                    # NER filter: _idx_ner:<ner_tag>:alias:<norm_alias>:<norm>
                    key = self.make_key(
                        self.IDX_NER, ner_tag, "alias", norm_alias, norm_key
                    )
                    db.put(key, self.EMPTY)

                # Create indexes for each country
                for country in row["countries"]:
                    country = country.lower()
                    # Main index: _main_:<norm>:<schema>:country:<country>
                    key = self.make_key(self.MAIN, norm_key, schema, "country", country)
                    db.put(key, self.EMPTY)

                    # Country filter: _idx_country:<country>:<norm>
                    key = self.make_key(self.IDX_COUNTRY, country, norm_key)
                    db.put(key, self.EMPTY)

        self.buffer = []

    def put(self, entity: EntityProxy) -> None:
        """Add entity to buffer"""
        if not entity.get_type_values(registry.name):
            return
        self.buffer.append(entity)
        if len(self.buffer) >= 10_000:
            self.flush()

    def iterate(self, norm_key: str | None = None) -> Docs:
        """
        Iterate over aggregated docs.

        Args:
            key: Optional key to filter by. If provided, only returns the Doc
                for that specific norm. If None, iterates all docs.

        LevelDB stores keys in sorted order, so we can group by norm.
        Main keys: _main_\x00<key>\x00<schema>\x00name/alias/country\x00<value>
        All data is in the keys, no values to parse.
        """
        current_key = None
        schemata: set[str] = set()
        names_: set[str] = set()
        aliases_: set[str] = set()
        countries_: set[str] = set()

        # Build prefix for iteration
        if norm_key:
            # Iterate only keys for specific norm
            prefix = f"{self.MAIN}{self.D}{norm_key}{self.D}".encode("utf-8")
        else:
            # Iterate all main index keys
            prefix = f"{self.MAIN}{self.D}".encode("utf-8")

        # Iterate through main index with prefix
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

    def get_key(self, name_key: str) -> str | None:
        """
        Look up the first norm key for a given name key using inverted index.

        Returns the first matching key, or None if not found.
        """
        for idx in (self.IDX_NAME, self.IDX_ALIAS):
            prefix = self.make_key(idx, name_key) + self.D.encode()
            for key, _ in self.db.iterator(prefix=prefix):
                key_str = key.decode("utf-8")
                # Format: _idx_name\x00<name_key>\x00<norm_key>
                parts = key_str.split(self.D)
                if len(parts) == 3 and parts[1] == key:
                    return parts[2]

    def get_doc(self, norm: str) -> Doc | None:
        """
        Retrieve a Doc by norm.

        Uses iterate() with norm_prefix to efficiently retrieve a single doc.
        Returns None if norm not found.
        """
        for doc in self.iterate(norm_key=norm):
            return doc
        # lookup by inverted idx
        key = self.get_key(norm)
        if key is not None:
            return self.get_doc(key)

    def iter_names_schema(self) -> Generator[tuple[str, str], None, None]:
        """Iterate over (name, schema) tuples from main index"""
        prefix = f"{self.MAIN}{self.D}".encode("utf-8")
        for key, _ in self.db.iterator(prefix=prefix):
            key_str = key.decode("utf-8")

            # Parse main index: _main_\x00<norm>\x00<schema>\x00name/alias\x00<value>
            parts = key_str.split(self.D)

            if len(parts) == 5:
                schema = parts[2]
                name_type = parts[3]
                name_value = parts[4]

                # Only yield names, not aliases
                if name_type == "name":
                    yield name_value, schema

    @property
    def count(self) -> int:
        """Count distinct norms (aggregated docs)"""
        seen = set()
        prefix = f"{self.MAIN}{self.D}".encode("utf-8")
        for key, _ in self.db.iterator(prefix=prefix):
            key_str = key.decode("utf-8")

            # Extract norm from key (format: "_main_\x00norm\x00schema\x00type\x00value")
            parts = key_str.split(self.D)
            if len(parts) >= 2:
                norm = parts[1]
                seen.add(norm)
        return len(seen)

    @property
    def count_rows(self) -> int:
        """Count total main keys (names + aliases + countries)"""
        count = 0
        prefix = f"{self.MAIN}{self.D}".encode("utf-8")
        for _ in self.db.iterator(prefix=prefix):
            count += 1
        return count

    def __iter__(self) -> Docs:
        yield from self.iterate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        self.flush()
        # Don't close DB - keep it open for subsequent use via cached Index

    def close(self) -> None:
        """Close the LevelDB connection"""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def __del__(self) -> None:
        """Ensure DB is closed when object is garbage collected"""
        self.close()
