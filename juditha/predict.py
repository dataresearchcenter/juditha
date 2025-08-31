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
from typing import Callable, Generator, Iterable, Literal, TypeAlias, cast

import fasttext
from anystore.logging import get_logger
from anystore.util import Took
from rigour.names import tokenize_name

from juditha.aggregator import Aggregator, Row
from juditha.model import SCHEMA_NER, SchemaPrediction
from juditha.settings import Settings

log = get_logger(__name__)
COLUMNS = "caption, schema, names, aliases, countries, symbols"
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


def _get_cached_model(model_path: str):
    """Thread-safe model caching"""
    with _cache_lock:
        if model_path not in _model_cache:
            log.info("Loading FastText model ...", path=model_path)
            _model_cache[model_path] = fasttext.load_model(model_path)
        return _model_cache[model_path]


def default_normalize(x: str) -> str:
    return x.lower()


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
        self.tokens: dict[str, set[str]] = defaultdict(set)
        # Allocate samples proportionally
        self.schema_allocation = int(self.limit * 0.5)  # 60% for schema diversity
        self.country_allocation = int(self.limit * 0.3)  # 20% for country diversity
        self.random_allocation = (
            self.limit - self.schema_allocation - self.country_allocation
        )  # 20% random
        self.train_ratio = train_ratio or 0.8
        self.collected = 0

    def make_sample(self) -> None:
        """Get representative sample data across all schemata and countries"""

        log.info("Querying sample data ...", uri=self.aggregator.uri)
        self.collected = 0
        with Took() as t:
            # 1. Get schema samples (distributed across all schemas)
            schema_query = "SELECT DISTINCT schema FROM names"
            result = self.aggregator.table.execute(schema_query)
            schemas = [row[0] for row in result.fetchall()]
            if schemas:
                samples_per_schema = max(1, self.schema_allocation // len(schemas))
                for schema in schemas:
                    query = f"""
                    SELECT {COLUMNS}
                    FROM names
                    WHERE schema = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                    """
                    result = self.aggregator.table.execute(
                        query, [schema, samples_per_schema]
                    )
                    collected = self._collect_result(result.fetchall())
                    log.info(f"Collected {collected} names for schema `{schema}`.")

            # 2. Get country samples (distributed across countries)
            country_query = """
            SELECT array_to_string(countries, ',') as country_str, COUNT(*) as cnt
            FROM names
            WHERE len(countries) > 0
            GROUP BY array_to_string(countries, ',')
            ORDER BY cnt DESC
            LIMIT 50
            """
            result = self.aggregator.table.execute(country_query)
            countries = [row[0] for row in result.fetchall()]
            if countries:
                samples_per_country = max(1, self.country_allocation // len(countries))
                for country_str in countries:
                    query = f"""
                    SELECT {COLUMNS}
                    FROM names
                    WHERE array_to_string(countries, ',') = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                    """
                    result = self.aggregator.table.execute(
                        query, [country_str, samples_per_country]
                    )
                    collected = self._collect_result(result.fetchall())
                    log.info(
                        f"Collected {collected} names for country `{country_str}`."
                    )

            # 3. Get random samples to fill remaining quota
            if self.collected < self.limit:
                remaining = min(self.random_allocation, self.limit - self.collected)
                random_query = f"""
                SELECT {COLUMNS}
                FROM names
                ORDER BY RANDOM()
                LIMIT ?
                """
                result = self.aggregator.table.execute(random_query, [remaining])
                collected = self._collect_result(result.fetchall())
                log.info(f"Collected {collected} random other names")

            # collect longer tokens
            log.info("Building tokens ...")
            for name, schemata in self.names.items():
                for token in tokenize_name(name, 8):
                    self.tokens[token].update(schemata)

            log.info(
                "Query sample data complete.",
                took=t.took,
                collected=self.collected,
                names=len(self.names),
                tokens=len(self.tokens),
            )

    def _collect_result(self, rows: Iterable[Row]) -> int:
        count = 0
        for caption, schema, names, aliases, _, _ in rows:
            self.names[self.normalizer(caption)].add(schema)
            for name in names:
                name = self.normalizer(name)
                self.names[name].add(schema)
                count += 1

            # Include a few aliases
            for i, alias in enumerate(aliases):
                if i < len(names) // 3:  # Include ~1/3 as many aliases as names
                    alias = self.normalizer(alias)
                    self.names[alias].add(schema)
        self.collected += count
        return count

    def _build_ft(self, name: str, schemata: set[str]) -> FT:
        if len(schemata) == 1:
            schema = list(schemata)[0]
            if schema in SCHEMA_NER:
                return cast(Label, f"__label__{schema}"), name
        return "__label__UNK", name

    def iterate(self) -> Generator[FT, None, None]:
        """Iterate names and tokens with added 10% synthetic noise. If a name or
        token has more than 1 schemata, the label will be UNK"""
        names = list(self.names.keys())
        random.shuffle(names)
        tokens = list(self.tokens.keys())
        random.shuffle(tokens)
        for name in names:
            yield self._build_ft(name, self.names[name])
            if random.randint(0, 100) < 11:
                yield self._build_ft(add_noise(name), self.names[name])
        for token in tokens:
            yield self._build_ft(token, self.tokens[token])
            if random.randint(0, 100) < 11:
                yield self._build_ft(add_noise(token), self.tokens[token])

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
