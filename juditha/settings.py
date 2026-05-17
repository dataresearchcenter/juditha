from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_URI = "juditha.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="juditha_", env_nested_delimiter="_", extra="ignore"
    )

    debug: bool = Field(alias="debug", default=False)
    uri: str = Field(default=DEFAULT_URI)
    fuzzy_threshold: float = 0.97
    limit: int = 10
    min_length: int = 4
    # Per-token length floor shared by the percolator (filters the
    # blocking set + the `tokens` field at index time) and the
    # Aho-Corasick extractor (derives its pattern-length floor as
    # `min_token_length * MIN_TOKEN_COUNT`). Lowering this admits
    # shorter / stopword-ish tokens, which improves recall on
    # single-strong-token names like "Abu Bakr" or "XI Jinping" at the
    # cost of a larger blocking-set fan-out. Changing this value
    # requires `juditha build` to re-emit both the `tokens` field and
    # the Aho automaton.
    min_token_length: int = 4
    # Maximum candidate Docs the percolator's blocking stage returns
    # (BM25-ranked). Raise for long input texts whose token set otherwise
    # crowds out true positives.
    percolate_block_limit: int = 10_000
