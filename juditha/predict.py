"""
Predict schemata of names with a fasttext model. Ref.:
https://github.com/alephdata/followthemoney-typepredict
"""

import random
import tempfile
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Generator, Literal, TypeAlias, cast

import fasttext
from anystore.logging import get_logger
from anystore.util import Took
from icu import Transliterator, UnicodeString
from rigour.names import Name

from juditha.aggregator import Aggregator
from juditha.model import SCHEMA_NER, SchemaPrediction
from juditha.settings import Settings

log = get_logger(__name__)
Label: TypeAlias = Literal[
    "__label__PublicBody",
    "__label__Organization",
    "__label__Company",
    "__label__Person",
    "__label__Address",
    "__label__UNK",
]
FT: TypeAlias = tuple[Label, str]

_model_cache = {}
_cache_lock = threading.Lock()

# Cache transliterator instance (thread-safe, created once)
_transliterator = None
_transliterator_lock = threading.Lock()


def _get_transliterator():
    """Get cached ICU transliterator instance"""
    global _transliterator
    if _transliterator is None:
        with _transliterator_lock:
            if _transliterator is None:
                # Any-Latin converts any script to Latin, Latin-ASCII removes diacritics
                _transliterator = Transliterator.createInstance(
                    "Any-Latin; Latin-ASCII"
                )
    return _transliterator


def _get_cached_model(model_path: str):
    """Thread-safe model caching"""
    with _cache_lock:
        if model_path not in _model_cache:
            log.info("Loading FastText model ...", path=model_path)
            _model_cache[model_path] = fasttext.load_model(model_path)
        return _model_cache[model_path]


def default_normalize(x: str) -> str:
    """Normalize name for FastText using rigour + ICU for multi-language support

    The model accuracy increases about 2-4% points with icu.

    Uses rigour.Name.comparable which handles:
    - Latin, Cyrillic, Greek → Latin transliteration
    - Diacritics removal (café → cafe)
    - Case folding

    Falls back to ICU for additional transliteration of:
    - Arabic → Latin
    - Chinese → Pinyin
    - Other non-Latin scripts
    """
    # Use rigour's comparable form (handles most European languages efficiently)
    name = Name(x)
    result = name.comparable

    # If rigour didn't transliterate (non-Latin chars remain), use ICU
    # Check if result contains non-ASCII chars beyond basic Latin range
    if any(ord(c) > 127 for c in result):
        transliterator = _get_transliterator()
        unicode_str = UnicodeString(result)
        transliterated = transliterator.transliterate(unicode_str)
        result = str(transliterated).casefold()

    return result


def add_noise(text: str) -> str:
    """Add synthetic noise to text for data augmentation"""
    if len(text) < 3:
        return text

    noise_type = random.choice(["char_swap", "char_drop", "char_add", "word_duplicate"])

    if noise_type == "char_swap" and len(text) > 2:
        # Swap two adjacent characters
        pos = random.randint(0, len(text) - 2)
        chars = list(text)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)

    elif noise_type == "char_drop" and len(text) > 3:
        # Drop a random character
        pos = random.randint(0, len(text) - 1)
        return text[:pos] + text[pos + 1 :]

    elif noise_type == "char_add":
        # Add a random character
        pos = random.randint(0, len(text))
        char = random.choice("abcdefghijklmnopqrstuvwxyz ")
        return text[:pos] + char + text[pos:]

    elif noise_type == "word_duplicate" and " " in text:
        # Duplicate a random word
        words = text.split()
        if len(words) > 1:
            word_to_dup = random.choice(words)
            pos = random.randint(0, len(words))
            words.insert(pos, word_to_dup)
            return " ".join(words)

    return text


class SampleAggregator:
    def __init__(
        self,
        aggregator: Aggregator,
        limit: int | None = 100_000,
        train_ratio: float | None = 0.8,
        normalizer: Callable[..., str] | None = None,
    ):
        self.aggregator = aggregator
        self.limit = limit or 100_000
        self.normalizer = normalizer or default_normalize
        self.names: dict[str, set[str]] = defaultdict(set)
        # Allocate samples proportionally
        self.schema_allocation = int(self.limit * 0.5)  # 50% for schema diversity
        self.country_allocation = int(self.limit * 0.3)  # 30% for country diversity
        self.random_allocation = (
            self.limit - self.schema_allocation - self.country_allocation
        )  # 20% random
        self.train_ratio = train_ratio or 0.8
        self.collected = 0

    def make_sample(self) -> None:
        """Get representative sample data across all schemata and countries"""

        log.info("Sampling data from LevelDB ...", uri=self.aggregator.uri)
        self.collected = 0
        with Took() as t:
            # First pass: iterate through all docs to build indexes
            schema_items: dict[str, list[tuple[str, str]]] = defaultdict(list)
            country_items: dict[str, list[tuple[str, str]]] = defaultdict(list)
            all_items: list[tuple[str, str]] = []

            for doc in self.aggregator:
                # Get the first name from each doc
                if doc.names:
                    name = next(iter(doc.names))
                    # Get the first schema
                    if doc.schemata:
                        schema = next(iter(doc.schemata))
                        item = (name, schema)

                        # Add to all items
                        all_items.append(item)

                        # Group by schema
                        schema_items[schema].append(item)

                        # Group by countries
                        for country in doc.countries:
                            country_items[country].append(item)

            # Sample by schema (distributed across all schemas)
            if schema_items:
                samples_per_schema = max(1, self.schema_allocation // len(schema_items))
                for schema, items in schema_items.items():
                    random.shuffle(items)
                    sampled = items[:samples_per_schema]
                    collected = self._collect_items(sampled)
                    log.info(f"Collected {collected} names for schema `{schema}`.")

            # Sample by country (distributed across top countries)
            if country_items:
                # Sort countries by number of items (descending)
                sorted_countries = sorted(
                    country_items.items(), key=lambda x: len(x[1]), reverse=True
                )
                # Take top 50 countries
                top_countries = sorted_countries[:50]

                samples_per_country = max(
                    1, self.country_allocation // len(top_countries)
                )
                for country, items in top_countries:
                    random.shuffle(items)
                    sampled = items[:samples_per_country]
                    collected = self._collect_items(sampled)
                    log.info(f"Collected {collected} names for country `{country}`.")

            # Random samples to fill remaining quota
            if self.collected < self.limit:
                remaining = min(self.random_allocation, self.limit - self.collected)
                random.shuffle(all_items)
                sampled = all_items[:remaining]
                collected = self._collect_items(sampled)
                log.info(f"Collected {collected} random other names")

            log.info(
                "Sample data complete.",
                took=t.took,
                collected=self.collected,
                unique_names=len(self.names),
            )

    def _collect_items(self, items: list[tuple[str, str]]) -> int:
        """Collect (name, schema) tuples into the names dict"""
        count = 0
        for name, schema in items:
            normalized_name = self.normalizer(name)
            self.names[normalized_name].add(schema)
            count += 1
        self.collected += count
        return count

    def _build_ft(self, name: str, schemata: set[str]) -> FT:
        if len(schemata) == 1:
            schema = list(schemata)[0]
            if schema in SCHEMA_NER:
                return cast(Label, f"__label__{schema}"), name
        return "__label__UNK", name

    def iterate(self) -> Generator[FT, None, None]:
        """Iterate names with added 10% synthetic noise. If a name or
        token has more than 1 schemata, the label will be UNK"""
        names = list(self.names.keys())
        random.shuffle(names)
        for name in names:
            yield self._build_ft(name, self.names[name])
            if random.randint(0, 100) < 11:
                yield self._build_ft(add_noise(name), self.names[name])

    def create_training_data(self) -> tuple[Path, Path]:
        """Create training and validation data files with 10% synthetic noise"""
        train_file = Path(tempfile.mktemp(suffix=".txt"))
        val_file = Path(tempfile.mktemp(suffix=".txt"))

        with open(train_file, "w") as train, open(val_file, "w") as val:
            for label, text in self.iterate():
                if (random.randint(0, 100) / 100) > self.train_ratio:
                    train.write(f"{label} {text}\n")
                else:
                    val.write(f"{label} {text}\n")
        return train_file, val_file


def get_sample(
    aggregator: Aggregator,
    limit: int | None = 100_000,
) -> Generator[FT, None, None]:
    """Get sample training data from aggregator"""
    sampler = SampleAggregator(aggregator, limit)
    sampler.make_sample()
    yield from sampler.iterate()


def train_model(
    aggregator: Aggregator,
    model_path: str | None = None,
    limit: int | None = 100_000,
    epoch: int = 25,
    lr: float = 0.1,
    wordNgrams: int = 2,
    dim: int = 100,
    ws: int = 5,
    minCount: int = 1,
    verbose: int = 2,
):
    """Train a simple FastText model for schema classification"""
    settings = Settings()
    if model_path is None:
        model_path = str(settings.make_path("schema_classifier.bin"))

    sampler = SampleAggregator(aggregator, limit)
    sampler.make_sample()
    train_file, val_file = sampler.create_training_data()

    try:
        model = fasttext.train_supervised(
            input=str(train_file),
            epoch=epoch,
            lr=lr,
            wordNgrams=wordNgrams,
            dim=dim,
            ws=ws,
            minCount=minCount,
            verbose=verbose,
        )

        # Test on validation data
        print(f"Validation accuracy: {model.test(str(val_file))[1]:.4f}")

        # Save model
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

    finally:
        # Clean up temp files
        train_file.unlink(missing_ok=True)
        val_file.unlink(missing_ok=True)


@lru_cache(100_000)
def predict_schema(
    text: str,
    model_path: str | None = None,
    normalizer: Callable[..., str] | None = None,
) -> Generator[SchemaPrediction, None, None]:
    """Predict schema for a given text using the trained FastText model"""
    settings = Settings()
    if model_path is None:
        model_path = str(settings.make_path("schema_classifier.bin"))

    if normalizer is None:
        normalizer = default_normalize

    model = _get_cached_model(model_path)
    normalized_text = normalizer(text)
    labels, scores = model.predict(normalized_text, k=3)

    for label, score in zip(labels, scores):
        if score > 0.5:
            label = label.replace("__label__", "")
            yield SchemaPrediction(name=text, label=label, score=round(float(score), 4))
