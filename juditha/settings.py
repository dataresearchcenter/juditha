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
    # Per-token length floor used by the Aho-Corasick extractor to
    # derive its pattern-length floor (`min_token_length *
    # MIN_TOKEN_COUNT`). The percolator does NOT consult this – it
    # indexes every name token in the `tokens` field and relies on
    # `percolate_min_should_match` to keep the candidate pool small.
    # Changing this value requires `juditha build` to re-emit the Aho
    # automaton.
    min_token_length: int = 4
    # Maximum candidate Docs the percolator's blocking stage returns
    # (BM25-ranked). Raise for long input texts whose token set otherwise
    # crowds out true positives.
    percolate_block_limit: int = 10_000
    # `minimum_number_should_match` passed to the percolator blocking
    # `boolean_query` (tantivy 0.26+). Default 2 is recall-safe for
    # names whose tokens all clear `MIN_TOKEN_CHARS` (hardcoded at 2):
    # percolatable names already require `MIN_TOKEN_COUNT == 2` tokens
    # downstream, and the `tokens` field is populated symmetrically
    # with the blocking_set filter, so every such name contributes
    # >= 2 tokens to the index. Names with a single-char token (e.g.
    # "A Lee") silently miss at MSM=2 since only "lee" survives the
    # MIN_TOKEN_CHARS floor – that's accepted noise-vs-recall trade.
    percolate_min_should_match: int = 2
