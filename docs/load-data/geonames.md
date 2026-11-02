# Load data / GeoNames

[GeoNames](https://www.geonames.org/) is a gazetteer of several million places with their alternate names in many languages. `juditha load-geonames` turns a GeoNames dump into [FollowTheMoney](https://followthemoney.tech) `Address` entities and pushes them into the aggregator like any other loader, so places become available to `lookup`, `extract` and `percolate` next to persons and organisations. This replaces the standalone [geonames-tagger](https://github.com/dataresearchcenter/geonames-tagger).

The loader needs the `geonames` extra, which pulls in [wordfreq](https://github.com/rspeer/wordfreq) (about 60 MB of word frequency data) for the dictionary-word filter described below. `lookup`, `extract` and `serve` do not need it.

```bash
pip install "juditha[geonames]"
```

!!! note "Data licence"

    GeoNames data is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Make sure your use complies.

## Input

`-i` takes any [anystore](https://docs.investigraph.dev/lib/anystore/) uri pointing at a GeoNames dump: the full [`allCountries.zip`](https://download.geonames.org/export/dump/allCountries.zip) (the default, roughly 400 MB), a per-country zip such as `DE.zip`, a subset like `cities5000.zip`, or the unpacked tab-separated `.txt`. Zips are read in memory, so a local copy is faster than streaming the full dump from geonames.org.

```bash
juditha load-geonames                                   # downloads allCountries.zip
juditha load-geonames -i ./allCountries.zip
juditha load-geonames -i https://download.geonames.org/export/dump/DE.zip
juditha load-geonames -i ./DE.txt
juditha build
```

As with every loader, nothing is searchable until `juditha build` has run.

## What gets loaded

Only a fraction of the dump is useful for name matching. The row and name filters are the ones geonames-tagger settled on; measured on a German intelligence report, the dictionary-word filter alone removes more than a third of all place mentions, every one of them a false positive:

- Feature classes `A` (countries, states, regions) and `P` (cities, villages) only.
- Population of at least 5 000.
- The raw `name` must have at least 8 characters for `P` rows and 4 for `A` rows. This drops short, ambiguous town names ("Rolle", "Kern"). Places with at least 100 000 inhabitants are exempt from this floor: "Berlin", "Hamburg" or "Kiel" are never noise, and without the exemption they would be unreachable, since their administrative rows carry the bare name only as an alternate that the substring rule below drops.
- `name` and `asciiname` are kept when their normalized form has at least 4 characters.
- Entries from `alternatenames` are kept when their normalized form has at least [`JUDITHA_GEONAMES_MIN_ALTERNATE_LENGTH`](../reference/settings.md#juditha_geonames_min_alternate_length) characters (default 6), are not purely numeric (postal codes, admin codes) and are not a single token that is a substring of the canonical name ("Berlin" as an alternate of "Land Berlin", "Kern" of "Kern County"). Short alternates are where most false positives come from; raise the floor for stricter matching, lower it toward 4 to catch transliterations like "Roma".
- Below 100 000 inhabitants, a single-token name or alternate that is a common dictionary word is dropped. The check runs against wordfreq's German, English, French, Spanish, Italian, Dutch, Portuguese and Russian lists (zipf frequency above 4.5, roughly the 30 000 most common words), because the language of the text that will be tagged is unknown at load time. This is what keeps "Auch" (a French town, German for "also"), "Rolle", "Lage" or "Mission" out of the automaton, while "Berlin", "China" or "Essen" survive through their population. Multi-token and hyphenated names are not checked: wordfreq scores phrases like "Deutsche Bank" as common too. The price is a few real mid-size places whose name is a word, such as Pisa or Como.

Every variant is normalized the same way the automaton normalizes text (casefold, accent and script folding), and deduplicated on that form: the ASCII name "Sant Julia de Loria" is the same variant as "Sant Julià de Lòria".

## The entity shape

Each surviving row becomes one `Address`:

```json
{"id": "GEO:2950159", "schema": "Address", "properties": {"name": ["Berlin"], "alias": ["Berlino", "Berlín", "..."], "country": ["de"]}}
```

The entity id is the GeoNames id with a `GEO:` prefix, but juditha is a name matcher, not a resolver: ids, coordinates and admin codes are not kept, the aggregator only stores names, aliases, countries, schemata and topics. Rows that share a canonical name (the city and the state of Berlin, several GeoNames entries for "Sant Julià de Lòria") are clustered like any other entities with the same name key.

## Single-token places

The Aho-Corasick extractor normally skips names with fewer than two tokens, see [Extract](../extras/extract.md). Clusters that carry the `Address` schema are exempt: "Berlin" or "Iran" are extractable, and the aggregator's caption floor is relaxed for them too, so "Rome" or "Bonn" survive. The percolator keeps its two-token floor, so single-token places are found by `extract` and `lookup` but not by `percolate`.

## From Python

```python
from juditha import get_build_store
from juditha import io

store = get_build_store()
io.load_geonames("DE.zip", store)
store.build()

store.extract("Ich war in Berlin.")
# [Mention(text='Berlin', start=11, end=17, schema_='Address', topics=set())]
```

`juditha.geonames.iter_entities(uri)` yields the `Address` proxies without loading them, if you want to write them to an FTM store instead.
