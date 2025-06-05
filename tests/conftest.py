from pathlib import Path

import pytest
from ftmq.io import smart_read_proxies

from juditha.store import get_store

FIXTURES_PATH = (Path(__file__).parent / "fixtures").absolute()


@pytest.fixture(scope="module")
def fixtures_path():
    return FIXTURES_PATH


@pytest.fixture(scope="module")
def eu_authorities():
    return [x for x in smart_read_proxies(FIXTURES_PATH / "eu_authorities.ftm.json")]


@pytest.fixture()
def store(tmp_path):
    # return get_store(str(tmp_path / "juditha"))
    return get_store()
