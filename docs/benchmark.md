# Benchmark

`contrib/benchmark/run.py` compares the two extraction paths, `Store.extract` (Aho-Corasick automaton) and `Store.percolate` (tantivy reverse-search), on a real input document against name corpora of three different sizes. This page captures what was measured, how each method scales, and where they disagree.

## Running it

```bash
.venv/bin/python contrib/benchmark/run.py                                # default names + text
.venv/bin/python contrib/benchmark/run.py vsbericht_2023.txt             # different text
.venv/bin/python contrib/benchmark/run.py --names peps.txt               # different corpus
.venv/bin/python contrib/benchmark/run.py --names peps.txt --rebuild     # force a fresh load + build
JUDITHA_PERCOLATE_BLOCK_LIMIT=100000 .venv/bin/python contrib/benchmark/run.py
```

Each names file maps to its own store directory at `contrib/benchmark/store-{stem}/`, so different corpora cohabit and switching between them costs nothing once each has been built. The script times the `from juditha import …` step separately so cold-start cost is explicit, then queries through a fresh read-only `Store` (as a worker process would). It warms the Aho-Corasick automaton with an empty `Store.extract("")` and runs one throw-away `percolate` so the reported per-call times are steady state, runs `percolate` once per slop value (0, 1, 2), and prints the symmetric difference of the unique-name sets so recall asymmetries are obvious.

Settings are read once at import, so `JUDITHA_*` environment variables have to be set before the script starts, not changed inside it.

## Setup

| Element | Value |
| --- | --- |
| Input text | `vsbericht_2024.txt`, the German federal *Verfassungsschutzbericht 2024* (annual report on constitutional protection), sourced from [verfassungsschutzberichte.de](https://verfassungsschutzberichte.de/) |
| Input size | 760 370 chars / 90 275 whitespace tokens |
| Hardware | 16 vCPU Xeon, 14 GB RAM, no GPU |
| Versions | juditha 4.8.1, tantivy 0.27.0, rigour 2.6.2, ahocorasick-rs 1.0.3, Python 3.14 |

Three corpora, in order of size, all pulled from the OpenSanctions public datasets (each ships a flat `names.txt` resource alongside the FTM entities):

| Corpus | Source | Names in file | Clusters after `name_key` merge | File size |
| --- | --- | ---: | ---: | --- |
| `peps_de` | [German subset of OpenSanctions PEPs](https://www.opensanctions.org/datasets/de_abgeordnetenwatch/) | 2 532 | 2 532 | 39 KB |
| `peps` | [OpenSanctions PEPs](https://www.opensanctions.org/datasets/peps/) | 983 184 | 926 758 | 23.9 MB |
| `names` | [OpenSanctions default](https://www.opensanctions.org/datasets/default/) (sanctions + PEPs + debarred + more) | 3 073 460 | 2 749 766 | 109.7 MB |

The cluster count is lower than the names count because `juditha.normalizer.name_key` (order-independent, accent-stripped canonical form) collapses surface variants of the same name into one aggregator cluster.

To reproduce: download the `names.txt` resource from the relevant dataset page (the OpenSanctions site links it under "Resources"), drop it next to `contrib/benchmark/run.py`, and pass `--names peps.txt` or `--names names.txt` to the script.

## Results

Timings are wall-clock. Build cost (one-time per corpus) is split from warmup cost (per process) and per-call cost. Per-call and warmup figures are from a second run with the store already in the OS page cache.

### Build (`juditha load-names` + `juditha build`, one-time per corpus)

| Stage | `peps_de` | `peps` | `names` |
| --- | ---: | ---: | ---: |
| `load_names` (LevelDB aggregator) | <0.1 s | 23 s | 84 s |
| Tantivy indexing + pattern collection | 4 s | 95 s @ 9.8 K docs/s | 337 s @ 8.2 K docs/s |
| Aho-Corasick finalize + save | <1 s | 6 s | 37 s |
| **Build total** | **4.8 s** | **125 s** | **459 s** |

`build()` makes a single pass over the aggregator: every doc is written to the tantivy index and handed to the extractor (`AhoExtractor.add_doc`) in the same loop, so pattern collection is part of the indexing row. The last row is resolving each pattern's schema and topics, building the automaton and writing `automaton.txt`.

### Per-process warmup cost (every process that opens the store)

| Stage | `peps_de` | `peps` | `names` |
| --- | ---: | ---: | ---: |
| `import juditha` | 1.1 s | 1.1 s | 1.1 s |
| Aho-Corasick warmup (first `Store.extract` call) | 5 ms | 4.2 s | 17.6 s |
| First `percolate` call (page cache warm) | 214 ms | 834 ms | 1.33 s |

The Aho-Corasick warmup is linear in the pattern count. On a cold page cache (right after the build) it was 6.1 s for `peps` and 24.5 s for `names`.

### Per-call cost (each extraction / percolation against an input text)

Input: `vsbericht_2024.txt`, 760 K chars / 90 K whitespace tokens. `percolate` at `slop=0`.

| Metric | `peps_de` (2.5 K) | `peps` (927 K) | `names` (2.75 M) |
| --- | ---: | ---: | ---: |
| `extract` (per call) | 445 ms | 453 ms | 532 ms |
| `percolate` block-limit 10 K | 211 ms | 569 ms | 1.23 s |
| `percolate` block-limit 100 K | (same) | 1.31 s | 4.02 s |
| Unique names found by `extract` | 6 | 25 | 89 |
| Unique names found by `percolate` | 6 | 26 | 90 (limit 10 K), 98 (limit 100 K) |
| Names in both | 6 | 25 | 81 (limit 10 K), 88 (limit 100 K) |
| `extract`-only | 0 | 0 | 8 (limit 10 K), 1 (limit 100 K) |
| `percolate`-only | 0 | 1 | 9 (limit 10 K), 10 (limit 100 K) |

Run-to-run variance on the per-call numbers is roughly ±20 %.

## Scaling characteristics

### `extract` (Aho-Corasick)

The warmup cost is linear in the corpus: 2.5 K patterns load in 5 ms, 947 K in 4.2 s, 2.75 M in 17.6 s. The patterns file is read and the automaton is built in memory; both scale with the pattern count. Once the automaton is warm, per-call cost is essentially constant in the corpus size (it depends on the input text length, not the pattern count): 445 ms, 453 ms, 532 ms across a 1000× corpus-size range.

This makes Aho-Corasick the right choice for a long-running process that extracts many documents against the same corpus: warmup amortizes, and per-call cost stays flat. It is the wrong choice when each invocation is short-lived (CLI scripts, on-demand serverless workers) because every fresh process pays the full warmup again.

### `percolate` (tantivy reverse-search)

There is no equivalent warmup. The tantivy index is mmap'd so the first searcher call lands on the OS page cache and pays only for the segments it touches; the in-memory text index for the input is rebuilt every call but is single-threaded and pinned to tantivy's minimum 15 MB heap, so it adds tens of milliseconds at most. The variable cost is the number of candidates surviving the blocking stage: 211 ms for 2.5 K candidates, 0.6 to 1.2 s for ~10 K (clamped by `JUDITHA_PERCOLATE_BLOCK_LIMIT`), 1.3 to 4 s when the cap is lifted to 100 K. Each candidate runs a tantivy phrase query against the input text plus a Python token-subsequence scan for offset recovery; both are linear in the candidate count.

This makes `percolate` the right choice for short-lived workers, for procrastinate-style fanout where many processes share the same on-disk index via the page cache, and for any context where Aho-Corasick's multi-second automaton load is the bottleneck.

### Where the two converge

At small corpora (a few thousand clusters) the two methods deliver the same answers, with `percolate` slightly faster per call and the automaton having a negligible warmup. At large corpora the automaton is faster per call but costs seconds to load; `percolate` costs nothing to start but more per call. The crossover depends on workload shape (documents per process), not on a single corpus-size threshold.

### A note on build cost

Build cost is paid once per corpus version, not per query. At 2.75 M names the full build takes ~7.5 minutes, of which the single tantivy indexing pass (~5.5 minutes at ~8 K docs/s) is by far the longest. This is acceptable as a one-shot when refreshing the corpus on a schedule (daily, weekly), but it does mean that swapping corpora at query time is not a thing: switching from `peps` to `names` means a multi-minute outage if you do it in-place. `contrib/benchmark/run.py` uses one store directory per names-file stem (`store-{stem}/`) so multiple corpora can cohabit and switching is free once both have been built.

## Recall comparison

At small corpora (`peps_de`, `peps`) the two methods agree on everything `extract` finds; on `peps`, `percolate` adds `To Lam` (see below). On the `names` corpus they diverge in both directions.

### `extract`-only (mostly vanishes when the cap is raised)

With `JUDITHA_PERCOLATE_BLOCK_LIMIT=10000` (the default) percolate misses 8 names that Aho finds:

- `Abu Bakr`, `Abu Yasin`, `Ahmad Sa'adat`, `Central Committee`, `Intelligence Organization`, `Die Partei`, `die Partei`: two common tokens each, so under BM25 they are out-ranked by candidates that match more or rarer input tokens and fall below the 10 K cap on a 2.75 M-cluster corpus. Raising the cap to 100 K brings every one of them in.
- `al-Aqsa e V`: `e` and `v` are below the percolator's hardcoded `MIN_TOKEN_CHARS = 2`, so only `al-aqsa` reaches the blocking query and the candidate can never clear `JUDITHA_PERCOLATE_MIN_SHOULD_MATCH=2`. This one stays `extract`-only at any cap.

### `percolate`-only

Nine to ten names are found by `percolate` and not by `extract`. They fall into three groups:

- **Shadowed by a longer pattern** (`Donald Trump`, `Karl Lauterbach`, `Revolutionary Guard`, `Revolutionary Guard Corps`, `Reconnaissance General Bureau`, `Palestinian Prisoner Solidarity Network`): these patterns are in the automaton, but the corpus also holds a longer name that covers the same span, and the automaton's leftmost-longest matching reports that one instead (`Präsident Donald Trump`, `Dr Karl Lauterbach`, `Islamic Revolutionary Guard Corps`). `percolate` checks each candidate independently, so it reports both the short and the long form. This is a difference in the reported mention text, not a missed entity.
- **Below the Aho pattern-length floor** (`APT 28`, `TKP ML`, `To Lam`): the normalized pattern is 6 characters, below `min_token_length × MIN_TOKEN_COUNT = 8` at defaults, so it never enters the automaton. The percolator has no such floor.
- **Noise** (`men So`): a two-token stored name matching ordinary text, which the automaton's length floor happens to filter.

The practical reading: on a large heterogeneous corpus `percolate` surfaces short names and overlapping variants that the automaton drops by design, while `extract` is more robust for two-token names built from common words, as long as the block limit is not raised.

## Tuning `JUDITHA_PERCOLATE_BLOCK_LIMIT`

This is the only knob worth turning between the defaults and a production deployment. The trade-off:

| Setting | When | Cost |
| --- | --- | --- |
| `10_000` (default) | Corpora up to ~1 M clusters, latency-sensitive workloads | ~0.5 to 1.2 s on 90 K-token input |
| `100_000` | Multi-million-cluster corpora where parity with Aho matters | ~1.3 s (`peps`) to ~4 s (`names`) on 90 K-token input |
| `1_000_000` | Investigative / batch jobs where recall trumps latency | linear in candidates surviving blocking |

The default is conservative on purpose. At a few hundred thousand clusters the cap barely bites (on `peps`, 100 K finds the same names at slop 0 for twice the time); at multi-million scale you start seeing recall asymmetry and bumping the cap is the cheapest fix.

## Slop: trades precision for recall, not runtime

`percolate(text, slop=N)` allows up to `N` intervening tokens between consecutive name parts when matching candidate phrases against the input text. `slop=0` is strict adjacency; `slop=1` lets `Jane Doe` match `Jane M. Doe`; `slop=2` allows two intervening tokens, and so on.

On the `names` corpus (block limit 10 K):

| Input | `slop` | unique names | percolate-only vs extract | runtime |
| --- | ---: | ---: | ---: | ---: |
| `vsbericht_2023.txt` | 0 | 90 | 9 | 1.23 s |
| `vsbericht_2023.txt` | 1 | 93 | 12 | 0.75 s |
| `vsbericht_2023.txt` | 2 | 98 | 17 | 0.85 s |
| `vsbericht_2024.txt` | 0 | 90 | 9 | 1.23 s |
| `vsbericht_2024.txt` | 1 | 99 | 18 | 0.73 s |
| `vsbericht_2024.txt` | 2 | 107 | 26 | 0.75 s |

Two observations worth knowing:

- **Runtime does not grow with slop.** Tantivy's positional intersect at `slop>0` is theoretically more expensive than strict adjacency, but the difference is below run-to-run noise here. The `slop=0` row is consistently the slowest only because it is the first measured call after the warm-up. Treat the cost as constant when choosing a slop value.
- **Recall grows monotonically with slop, precision drops.** Going from 0 to 1 picks up legitimate variants (line-break and punctuation splits like `Combat 18 Deutschland (C18)`, `Babbar Khalsa International`). Going to 2 starts catching noise: on the 2023 report it adds spans such as `den Staat Israel kein` and `die postautonome Interventionistische Linke`, which are adjective- or article-padded text, not references to a stored entity. With the block limit at 100 K the effect is stronger (2024 report: 10, 30, 49 percolate-only names at slop 0, 1, 2).

Practical guidance: `slop=0` is the right default for entity verification against a closed reference set. Use `slop=1` only when you know the input has line-break or punctuation noise (OCR output, scraped HTML). `slop=2` is rarely justified outside very controlled corpora.
