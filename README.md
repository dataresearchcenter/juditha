[![juditha on pypi](https://img.shields.io/pypi/v/juditha)](https://pypi.org/project/juditha/)
[![PyPI Downloads](https://static.pepy.tech/badge/juditha/month)](https://pepy.tech/projects/juditha)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/juditha)](https://pypi.org/project/juditha/)
[![Python test and package](https://github.com/dataresearchcenter/juditha/actions/workflows/python.yml/badge.svg)](https://github.com/dataresearchcenter/juditha/actions/workflows/python.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Coverage Status](https://coveralls.io/repos/github/dataresearchcenter/juditha/badge.svg?branch=main)](https://coveralls.io/github/dataresearchcenter/juditha?branch=main)
[![AGPLv3+ License](https://img.shields.io/pypi/l/juditha)](./LICENSE)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)

# Juditha

A super-fast in-process lookup service for canonical names, backed by [tantivy](https://github.com/quickwit-oss/tantivy).

`juditha` exists to tame the noise that follows from [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition): given a huge list of *known names* (company registries, persons of interest, sanctions lists), it tells you whether a span produced by your NER pipeline corresponds to one of them, even when the casing, accents, token order, or spelling differs.

The implementation uses a pre-populated names database and index. Data is either [FollowTheMoney](https://followthemoney.tech) entities or simply list of names.

## Run as a service

`juditha` is an in-process library first, but the read-only half of the store can be served over gRPC so several workers share one built corpus:

```bash
JUDITHA_URI=/var/lib/juditha juditha serve --host 0.0.0.0
```

Clients just point `JUDITHA_URI` at it, and `lookup` / `extract` / `percolate` go over the wire unchanged:

```bash
JUDITHA_URI=grpc://localhost:50051 juditha lookup "Jane Doe"
```

There is no authentication, so keep it on a private network. Docker image: `ghcr.io/dataresearchcenter/juditha`, mount a built store at `/data`.

## Documentation

https://docs.investigraph.dev/lib/juditha

## The name

**Juditha Dommer** was the daughter of a coppersmith and raised seven children, while her husband Johann Pachelbel wrote a *canon*.

## Versioning

To mark the compatibility with [followthemoney](https://followthemoney.tech), `juditha` follows the same major version, which is currently 4.x.x.

## License and copyright

`juditha`, (C) 2024 investigativedata.io. (C) 2025, 2026 [Data and Research Center – DARC](https://dataresearchcenter.org). Licensed under AGPLv3 or later. See [NOTICE](https://github.com/dataresearchcenter/juditha/blob/main/NOTICE) and [LICENSE](https://github.com/dataresearchcenter/juditha/blob/main/LICENSE).
