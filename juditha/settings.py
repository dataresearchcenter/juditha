from pathlib import Path

from anystore.util import join_uri, uri_to_path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_URI = "juditha.db"


class ApiContact(BaseSettings):
    name: str = "Data and Research Center – DARC"
    url: str = "https://github.com/dataresearchcenter/juditha/"
    email: str = "hi@dataresearchcenter.org"


class ApiSettings(BaseSettings):
    title: str = "Juditha"
    contact: ApiContact = ApiContact()
    description_uri: str = "README.md"


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

    api: ApiSettings = ApiSettings()

    def make_path(self, path: str) -> Path:
        return uri_to_path(join_uri(self.uri, path))
