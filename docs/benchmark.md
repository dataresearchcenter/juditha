# Benchmark

`contrib/benchmark/run.py` compares the two extraction paths, `Store.extract` (Aho-Corasick automaton) and `Store.percolate` (tantivy reverse-search), on a real input document against name corpora of three different sizes. This page captures what was measured, how each method scales, and where they disagree.

## Running it

```bash
.venv/bin/python contrib/benchmark/run.py                                # default names + text
.venv/bin/python contrib/benchmark/run.py vsbericht_2024.txt             # different text
.venv/bin/python contrib/benchmark/run.py --names peps.txt               # different corpus
JUDITHA_PERCOLATE_BLOCK_LIMIT=100000 .venv/bin/python contrib/benchmark/run.py
```

Each names file maps to its own store directory at `contrib/benchmark/store-{stem}/`, so different corpora cohabit and switching between them costs nothing once each has been built. The script times the `from juditha import …` step separately so cold-start cost is explicit, warms the Aho-Corasick automaton with an empty `Store.extract("")` so the reported `extract` time is per-call only, and prints the symmetric difference of the two unique-name sets so recall asymmetries are obvious.

## Setup

| Element | Value |
| --- | --- |
| Input text | `vsbericht_2024.txt`, the German federal *Verfassungsschutzbericht 2024* (annual report on constitutional protection), sourced from [verfassungsschutzberichte.de](https://verfassungsschutzberichte.de/) |
| Input size | 760 369 chars / 90 275 whitespace tokens |
| Hardware | local laptop, no GPU |
| juditha version | tantivy 0.25, rigour 2.1.1, pyicu transitively |

Three corpora, in order of size, all pulled from the OpenSanctions public datasets (each ships a flat `names.txt` resource alongside the FTM entities):

| Corpus | Source | Names in file | Clusters after `name_key` merge | File size |
| --- | --- | ---: | ---: | --- |
| `peps_de` | [German subset of OpenSanctions PEPs](https://www.opensanctions.org/datasets/de_abgeordnetenwatch/) | 2 532 | 2 532 | 39 KB |
| `peps` | [OpenSanctions PEPs](https://www.opensanctions.org/datasets/peps/) | 983 184 | 928 037 | 23.9 MB |
| `names` | [OpenSanctions default](https://www.opensanctions.org/datasets/default/) (sanctions + PEPs + debarred + more) | 3 073 460 | 2 764 365 | 109.7 MB |

The cluster count is lower than the names count because `juditha.normalizer.name_key` (order-independent, accent-stripped canonical form) collapses surface variants of the same name into one aggregator cluster.

To reproduce: download the `names.txt` resource from the relevant dataset page (the OpenSanctions site links it under "Resources"), drop it next to `contrib/benchmark/run.py`, and pass `--names peps.txt` or `--names names.txt` to the script.

## Results

Timings are wall-clock on a single thread. Build cost (one-time per corpus) is split from cold-start cost (per process) and per-call cost.

### Build (`juditha load-names` + `juditha build`, one-time per corpus)

| Stage | `peps_de` | `peps` | `names` |
| --- | ---: | ---: | ---: |
| `load_names` (LevelDB aggregator) | <1 s | 26.02 s | 97.09 s |
| Tantivy indexing | <1 s | 68 s @ 13.6 K docs/s | 268 s @ 10.3 K docs/s |
| Aho-Corasick automaton build | <1 s | 3 s | 14 s |
| **Build total** | **<2 s** | **100.24 s** | **403.33 s** |

The aggregator dominates load time at small scale and tantivy indexing dominates at large scale. Aho-Corasick automaton construction is essentially free relative to the other two stages, but its persisted `automaton.txt` is what the cold-start warmup later has to read back in.

### Per-process warmup cost (every process that opens the store)

| Stage | `peps_de` | `peps` | `names` |
| --- | ---: | ---: | ---: |
| Aho-Corasick warmup (first `Store.extract` call) | 4 ms | 3.24 s | 13.95 s |

The Aho-Corasick warmup is linear in the pattern count.

### Per-call cost (each extraction / percolation against an input text)

Input: `vsbericht_2024.txt`, 760 K chars / 90 K whitespace tokens.

| Metric | `peps_de` (2.5 K) | `peps` (928 K) | `names` (2.76 M) |
| --- | ---: | ---: | ---: |
| `extract` (per call) | 416 ms | 459 ms | 559 ms |
| `percolate` block-limit 10 K | 397 ms | 1.30 s | 1.59 s |
| `percolate` block-limit 100 K | (same) | (same) | 4.56 s |
| Unique names found by `extract` | 4 | 25 | 90 |
| Unique names found by `percolate` | 4 | 25 | 96 (limit 100 K) |
| Names in both | 4 | 25 | 90 |
| `extract`-only | 0 | 0 | 0 (limit 100 K), 1 (limit 10 K) |
| `percolate`-only | 0 | 0 | 6 |

## Scaling characteristics

### `extract` (Aho-Corasick)

The cold-start cost is linear in the corpus: 2.5 K patterns load in 4 ms, 928 K in 3.2 s, 2.76 M in 14 s. The patterns file is read and an in-memory FST automaton is built; both scale with the pattern count. Once the automaton is warm, per-call cost is essentially constant in the corpus size (it depends on the input text length, not the pattern count): 416 ms, 459 ms, 559 ms across a 1000× corpus-size range.

This makes Aho-Corasick the right choice for a long-running process that extracts many documents against the same corpus: warmup amortizes, and per-call cost stays flat. It is the wrong choice when each invocation is short-lived (CLI scripts, on-demand serverless workers) because every fresh process pays the full warmup again.

### `percolate` (tantivy reverse-search)

There is no equivalent warmup. The tantivy index is mmap'd so the first searcher call lands on the OS page cache and pays only for the segments it touches; the in-memory text index for the input is rebuilt every call but is single-threaded and pinned to tantivy's minimum 15 MB heap, so it adds tens of milliseconds at most. The variable cost is the number of candidates surviving the blocking stage: 397 ms for 2.5 K candidates, 1.30 s for ~10 K (clamped by `PERCOLATE_BLOCK_LIMIT`), 4.56 s when the cap is lifted to 100 K. Each candidate runs a tantivy phrase query against the input text plus a Python token-subsequence scan for offset recovery; both are linear in the candidate count.

This makes `percolate` the right choice for short-lived workers, for procrastinate-style fanout where many processes share the same on-disk index via the page cache, and for any context where Aho-Corasick's gigabyte-scale automaton load is the bottleneck.

### Where the two converge

At small corpora (a few thousand clusters) the two methods deliver the same answers in comparable per-call time, with the automaton having a negligible warmup. At one extreme of cost the automaton dominates (large corpora, short-lived processes); at the other extreme the percolator dominates (small corpora, persistent processes). The crossover depends on workload shape, not on a single corpus-size threshold.

### A note on build cost

Build cost is paid once per corpus version, not per query. At 2.76 M names the full build takes ~6.7 minutes on a laptop, of which the tantivy indexing phase (~4.5 minutes at ~10 K docs/s) is by far the longest. This is acceptable as a one-shot when refreshing the corpus on a schedule (daily, weekly), but it does mean that swapping corpora at query time is not a thing: switching from `peps` to `names` means a 7-minute outage if you do it in-place. `contrib/benchmark/run.py` uses one store directory per names-file stem (`store-{stem}/`) so multiple corpora can cohabit and switching is free once both have been built.

## Recall comparison

At small corpora (`peps_de`, `peps`) the two methods agree perfectly: every unique name found by one is found by the other. On the `names` corpus they diverge in both directions.

### `extract`-only (vanishes when the cap is raised)

With `JUDITHA_PERCOLATE_BLOCK_LIMIT=10000` (the default at this writing) percolate misses 8 names that Aho finds. Every one of them has at most a single token of length ≥ 4 going into the percolator's `tokens` blocking field:

- `Abu Bakr`, `Abu Yasin`: `abu` and `bakr`/`yasin` after normalisation; only `bakr` / `yasin` (≥ 4 chars) make it into the blocking index.
- `Combat 18`, `Hamburg 41`, `Hamburg (41`: digits are tokens too but get filtered by `settings.min_token_length=4`, leaving just `combat` or `hamburg`.
- `XI Jinping`: `xi` is 2 chars; only `jinping` is indexed.
- `al-Aqsa TV`, `al-Aqsa e.V`: only `al-aqsa` is ≥ 4 chars.

With a single-token blocking signature these candidates score weakly under BM25 and get out-ranked by docs that match more / rarer input tokens. At 2.76 M clusters they fall below the 10 K cap; raising the cap to 100 K brings every one of them in and the `extract`-only set drops to zero.

### `percolate`-only (consistent across cap settings)

Six names are found by `percolate` and missed by `extract`, regardless of the block-limit value:

- `Donald Trump`
- `Karl Lauterbach`
- `Palestinian Prisoner Solidarity Network`
- `Reconnaissance General Bureau`
- `Revolutionary Guard`
- `Revolutionary Guard Corps`

All are multi-token (≥ 2) and well above the derived Aho pattern-length floor (`min_token_length × MIN_TOKEN_COUNT = 8` at defaults), so the Aho thresholds aren't filtering them. The discrepancy is structural: percolate evaluates each candidate's `names ∪ aliases` as a tantivy phrase query against the input text, while Aho-Corasick matches against the patterns explicitly added to its automaton during `Store.build`. If a stored cluster carries a longer canonical name (`Donald J. Trump`) but no plain `Donald Trump` alias, percolate can still surface the cluster (via a different cluster's aliases or via tokens shared across the text) but Aho cannot, because the literal pattern `" donald trump "` is not in its automaton.

The practical reading: `percolate` gives modestly higher recall on a large heterogeneous corpus where the same person or organisation is referenced under multiple captions, and where the canonical caption is not the form most commonly used in running text.

## Tuning `JUDITHA_PERCOLATE_BLOCK_LIMIT`

This is the only knob worth turning between the defaults and a production deployment. The trade-off:

| Setting | When | Cost |
| --- | --- | --- |
| `10_000` (default) | Corpora up to ~1 M clusters, latency-sensitive workloads | ~1 s on 90 K-token input |
| `100_000` | Multi-million-cluster corpora where parity with Aho matters | ~4 to 5 s on 90 K-token input |
| `1_000_000` | Investigative / batch jobs where recall trumps latency | linear in candidates surviving blocking |

The default is conservative on purpose. At a few hundred thousand clusters the cap never bites; at multi-million scale you start seeing recall asymmetry and bumping the cap is the cheapest fix.

## Slop: trades precision for recall, not runtime

`percolate(text, slop=N)` allows up to `N` intervening tokens between consecutive name parts when matching candidate phrases against the input text. `slop=0` is strict adjacency; `slop=1` lets `Jane Doe` match `Jane M. Doe`; `slop=2` allows two intervening tokens, and so on.

In practice on the `names` corpus and `vsbericht_2023.txt`:

| `slop` | unique names | percolate-only vs extract | runtime |
| ---: | ---: | ---: | ---: |
| 0 | 126 | 16 | ~1.2 s |
| 1 | 137 | 27 | ~1.2 s |
| 2 | 147 | 37 | ~1.1 s |

Two observations worth knowing:

- **Runtime is essentially flat across slop values.** Tantivy's positional intersect at `slop>0` is theoretically more expensive than strict adjacency, but the difference disappears below run-to-run noise in this benchmark. Treat the cost as constant when choosing a slop value.
- **Recall grows monotonically with slop, precision drops.** Going from 0 to 1 picks up legitimate variants (line-break splits like `"Palestinian Prisoner\nSolidarity Network"`, abbreviations, single-token inserts). Going from 1 to 2 starts catching noisier matches: in German text where `Die Partei` is a stored name, `slop=2` surfaces `"Die Führungsebene der Partei"`, `"die Migrationspolitik. Die Partei"`, and similar adjective-padded forms that aren't genuine references to the stored entity.

Practical guidance: `slop=0` is the right default for entity verification against a closed reference set. Use `slop=1` only when you know the input has line-break or punctuation noise (OCR output, scraped HTML). `slop=2` is rarely justified outside very controlled corpora.
