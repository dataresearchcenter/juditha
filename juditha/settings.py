from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_URI = "juditha.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="juditha_",
        env_nested_delimiter="_",
        nested_model_default_partial_update=True,
    )

    debug: bool = Field(alias="debug", default=False)
    uri: str = Field(default=DEFAULT_URI)
    fuzzy_threshold: float = 0.97
    limit: int = 10
    min_length: int = 4
