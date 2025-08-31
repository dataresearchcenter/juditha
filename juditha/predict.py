"""
Predict schemata of names with a fasttext model. Ref.:
https://github.com/alephdata/followthemoney-typepredict
"""

import random
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Generator, Literal, TypeAlias, cast

import fasttext
from anystore.logging import get_logger
from anystore.util import Took

from juditha.aggregator import Aggregator
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


def get_sample(
    aggregator: Aggregator,
    limit: int | None = 100_000,
    normalizer: Callable[..., str] | None = None,
) -> Generator[FT, None, None]:
    """Get representative sample data across all schemata and countries"""

    log.info("Querying sample data ...", uri=aggregator.uri)
    with Took() as t:
        normalizer = normalizer or default_normalize
        actual_limit = limit or 100_000

        # Allocate samples proportionally
        schema_allocation = int(actual_limit * 0.6)  # 60% for schema diversity
        country_allocation = int(actual_limit * 0.2)  # 20% for country diversity
        random_allocation = (
            actual_limit - schema_allocation - country_allocation
        )  # 20% random

        seen_rows = set()  # For deduplication using caption+schema as key
        yielded_count = 0

        # 1. Get schema samples (distributed across all schemas)
        schema_query = "SELECT DISTINCT schema FROM names"
        schema_result = aggregator.table.execute(schema_query)
        schemas = [row[0] for row in schema_result.fetchall()]

        if schemas:
            samples_per_schema = max(1, schema_allocation // len(schemas))

            for schema in schemas:
                if yielded_count >= actual_limit:
                    break

                schema_sample_query = f"""
                SELECT {COLUMNS}
                FROM names
                WHERE schema = ?
                ORDER BY RANDOM()
                LIMIT ?
                """
                schema_result = aggregator.table.execute(
                    schema_sample_query, [schema, samples_per_schema]
                )

                for row in schema_result.fetchall():
                    if yielded_count >= actual_limit:
                        break

                    caption, row_schema, names, aliases, countries, symbols = row
                    dedup_key = (caption, row_schema)

                    if dedup_key in seen_rows:
                        continue
                    seen_rows.add(dedup_key)

                    if row_schema in SCHEMA_NER:
                        label: Label = cast(Label, f"__label__{row_schema}")
                        for name in names:
                            if yielded_count >= actual_limit:
                                break
                            yield label, normalizer(name)
                            yielded_count += 1

                        # Include fewer aliases than names
                        for i, alias in enumerate(aliases):
                            if yielded_count >= actual_limit:
                                break
                            if (
                                i < len(names) // 3
                            ):  # Include ~1/3 as many aliases as names
                                yield label, normalizer(alias)
                                yielded_count += 1

        # 2. Get country samples (distributed across countries)
        if yielded_count < actual_limit:
            country_query = """
            SELECT array_to_string(countries, ',') as country_str, COUNT(*) as cnt
            FROM names
            WHERE len(countries) > 0
            GROUP BY array_to_string(countries, ',')
            ORDER BY cnt DESC
            LIMIT 50
            """
            country_result = aggregator.table.execute(country_query)
            countries = [row[0] for row in country_result.fetchall()]

            if countries:
                samples_per_country = max(1, country_allocation // len(countries))

                for country_str in countries:
                    if yielded_count >= actual_limit:
                        break

                    country_sample_query = f"""
                    SELECT {COLUMNS}
                    FROM names
                    WHERE array_to_string(countries, ',') = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                    """
                    country_result = aggregator.table.execute(
                        country_sample_query, [country_str, samples_per_country]
                    )

                    for row in country_result.fetchall():
                        if yielded_count >= actual_limit:
                            break

                        caption, row_schema, names, aliases, countries, symbols = row
                        dedup_key = (caption, row_schema)

                        if dedup_key in seen_rows:
                            continue
                        seen_rows.add(dedup_key)

                        if row_schema in SCHEMA_NER:
                            label: Label = cast(Label, f"__label__{row_schema}")
                            for name in names:
                                if yielded_count >= actual_limit:
                                    break
                                yield label, normalizer(name)
                                yielded_count += 1

                            # Include fewer aliases than names
                            for i, alias in enumerate(aliases):
                                if yielded_count >= actual_limit:
                                    break
                                if (
                                    i < len(names) // 3
                                ):  # Include ~1/3 as many aliases as names
                                    yield label, normalizer(alias)
                                    yielded_count += 1

        # 3. Get random samples to fill remaining quota
        if yielded_count < actual_limit:
            remaining = min(random_allocation, actual_limit - yielded_count)
            random_query = f"""
            SELECT {COLUMNS}
            FROM names
            ORDER BY RANDOM()
            LIMIT ?
            """
            random_result = aggregator.table.execute(
                random_query, [remaining * 2]
            )  # Get extra to account for dedup

            for row in random_result.fetchall():
                if yielded_count >= actual_limit:
                    break

                caption, row_schema, names, aliases, countries, symbols = row
                dedup_key = (caption, row_schema)

                if dedup_key in seen_rows:
                    continue
                seen_rows.add(dedup_key)

                if row_schema in SCHEMA_NER:
                    label: Label = cast(Label, f"__label__{row_schema}")
                    for name in names:
                        if yielded_count >= actual_limit:
                            break
                        yield label, normalizer(name)
                        yielded_count += 1

                    # Include fewer aliases than names
                    for i, alias in enumerate(aliases):
                        if yielded_count >= actual_limit:
                            break
                        if i < len(names) // 3:  # Include ~1/3 as many aliases as names
                            yield label, normalizer(alias)
                            yielded_count += 1

        log.info("Query sample data complete.", took=t.took)


def create_training_data(
    aggregator: Aggregator, train_ratio: float = 0.8, limit: int | None = 100_000
) -> tuple[Path, Path]:
    """Create training and validation data files with 10% synthetic noise"""
    train_file = Path(tempfile.mktemp(suffix=".txt"))
    val_file = Path(tempfile.mktemp(suffix=".txt"))

    samples = list(get_sample(aggregator, limit))
    split_idx = int(len(samples) * train_ratio)

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    with open(train_file, "w") as f:
        # Write original training samples
        for label, text in train_samples:
            f.write(f"{label} {text}\n")

        # Add 10% synthetic noise samples
        noise_count = int(len(train_samples) * 0.1)
        noise_samples = random.sample(
            train_samples, min(noise_count, len(train_samples))
        )

        for label, text in noise_samples:
            noisy_text = add_noise(text)
            f.write(f"{label} {noisy_text}\n")

    with open(val_file, "w") as f:
        for label, text in val_samples:
            f.write(f"{label} {text}\n")

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

    train_file, val_file = create_training_data(aggregator, limit=limit)

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
            schema = label.replace("__label__", "")
            yield SchemaPrediction(
                name=text, schema_name=schema, score=round(float(score), 4)
            )
