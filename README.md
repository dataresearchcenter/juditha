[![juditha on pypi](https://img.shields.io/pypi/v/juditha)](https://pypi.org/project/juditha/) [![Python test and package](https://github.com/dataresearchcenter/juditha/actions/workflows/python.yml/badge.svg)](https://github.com/dataresearchcenter/juditha/actions/workflows/python.yml) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Coverage Status](https://coveralls.io/repos/github/dataresearchcenter/juditha/badge.svg?branch=main)](https://coveralls.io/github/dataresearchcenter/juditha?branch=main) [![MIT License](https://img.shields.io/pypi/l/juditha)](./LICENSE)

# juditha

A super-fast lookup service for canonical names based on [tantivy](https://github.com/quickwit-oss/tantivy).

`juditha` wants to solve the noise/garbage problem occurring when working with [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition). Given the availability of huge lists of *known names*, such as company registries or lists of persons of interest, one could canonize `ner`-results against this service to check if they are known.

The implementation uses a pre-populated tantivy index. Data is either [FollowTheMoney](https://followthemoney.tech) entities or simply list of names.

## quickstart

    pip install juditha

### populate

    echo "Jane Doe\nAlice" | juditha load-names

### lookup

    juditha lookup "jane doe"
    "Jane Doe"

To match more fuzzy, reduce the threshold (default 0.97):

    juditha lookup "doe, jane" --threshold 0.5
    "Jane Doe"

## data import

### from ftm entities

    cat entities.ftm.json | juditha load-entities

### from anywhere

    juditha load-names -i s3://my_bucket/names.txt
    juditha load-entities -i https://data.ftm.store/eu_authorities/entities.ftm.json

### a complete dataset or catalog

Following the [`nomenklatura`](https://github.com/opensanctions/nomenklatura) specification, a dataset json config needs `names.txt` or `entities.ftm.json` in its resources.

    juditha load-dataset https://data.ftm.store/eu_authorities/index.json
    juditha load-catalog https://data.ftm.store/investigraph/catalog.json

## use in python applications

```python
from juditha import lookup

assert lookup("jane doe") == "Jane Doe"
assert lookup("doe, jane") is None
assert lookup("doe, jane", threshold=0.5) == "Jane Doe"
```

## run as api

    uvicorn --port 8000 juditha.api:app --workers 8

### api calls

Just do head requests to check if a name is known:

    curl -I "http://localhost:8000/jane%20doe"
    HTTP/1.1 200 OK

    curl -I "http://localhost:8000/John"
    HTTP/1.1 404 Not Found

An actual request returns the canonized name and optional [FollowTheMoney Schema](https://followthemoney.tech/explorer/):

    curl "http://localhost:8000/jane doe"
    {"name": "Jane Doe", "schema": "Person", "score": 1}

Tweak threshold (default: 0.97) via `?threshold=0.6`

## the name

**Juditha Dommer** was the daughter of a coppersmith and raised seven children, while her husband Johann Pachelbel wrote a *canon*.


## License and Copyright

`juditha`, (C) 2024 investigativedata.io

`juditha`, (C) 2025 [Data and Research Center – DARC](https://dataresearchcenter.org)

`juditha` is licensed under the AGPLv3 or later license.

see [NOTICE](./NOTICE) and [LICENSE](./LICENSE)
