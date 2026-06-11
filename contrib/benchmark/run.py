"""Benchmark juditha extract vs percolate on a name corpus.

Reads a names list (default `contrib/benchmark/names.txt`), builds a
juditha store at `contrib/benchmark/store-{names-stem}/` so different
corpora cohabit, runs `Store.extract` (Aho-Corasick) once and
`Store.percolate` (tantivy reverse-search) once per slop value in
`SLOP_VALUES`, then prints wall-clock timings, unique-name counts, and
the symmetric difference against extract for each slop run.

Usage:
    .venv/bin/python contrib/benchmark/run.py
    .venv/bin/python contrib/benchmark/run.py path/to/text.txt
    .venv/bin/python contrib/benchmark/run.py --names path/to/names.txt
    .venv/bin/python contrib/benchmark/run.py --rebuild
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_NAMES_PATH = BENCH_DIR / "names.txt"
DEFAULT_TEXT = BENCH_DIR / "vsbericht_2024.txt"
# Percolate slop values to compare in every run. slop=0 is the strict
# baseline; slop=1 lets "Jane M. Doe" match "Jane Doe"; slop=2 catches
# names with up to two intervening tokens.
SLOP_VALUES = (0, 1, 2)


def _hms(seconds: float) -> str:
    """Compact ms/s formatter."""
    if seconds < 1.0:
        return f"{seconds * 1000:8.1f} ms"
    return f"{seconds:8.2f} s "


def store_uri_for(names_path: Path) -> Path:
    """One store directory per names-file stem so corpora cohabit."""
    return BENCH_DIR / f"store-{names_path.stem}"


def ensure_names(names_path: Path) -> Path:
    """Verify the names file exists; fail fast otherwise."""
    if not names_path.exists():
        sys.exit(f"names file not found: {names_path}")
    size_mb = names_path.stat().st_size / (1024 * 1024)
    print(f"names:       {names_path} ({size_mb:.1f} MB)")
    return names_path


def time_import() -> tuple[float, object, object]:
    """Time the import of the juditha public surface used here.

    Returns (elapsed_seconds, get_store, io_module).
    """
    t = time.perf_counter()
    from juditha import get_store as _get_store
    from juditha import io as _io

    elapsed = time.perf_counter() - t
    return elapsed, _get_store, _io


def ensure_store(
    get_store,
    io_module,
    names_path: Path,
    store_uri: Path,
    force: bool = False,
) -> tuple[float, object]:
    """Build the store if needed; return (elapsed_seconds, store).

    Elapsed is zero if the store was already populated and not forced.
    """
    os.environ["JUDITHA_URI"] = str(store_uri)
    get_store.cache_clear()
    store = get_store()

    count = store.aggregator.count
    if count > 0 and not force:
        print(f"store:       {store_uri} already populated ({count} clusters)")
        return 0.0, store

    t = time.perf_counter()
    print(f"Loading names from {names_path} ...")
    io_module.load_names(str(names_path), store)
    load_t = time.perf_counter() - t
    print(f"  loaded in {_hms(load_t)}")

    print("Building tantivy index + extractor ...")
    t_build = time.perf_counter()
    store.build()
    build_t = time.perf_counter() - t_build
    print(f"  built in {_hms(build_t)}")
    print(f"store:       {store.aggregator.count} clusters")
    return load_t + build_t, store


def fmt_names(names: set[str], limit: int = 30) -> str:
    items = sorted(names)
    head = items[:limit]
    body = "\n".join(f"  {n!r}" for n in head)
    if len(items) > limit:
        body += f"\n  ... and {len(items) - limit} more"
    return body


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "text",
        type=Path,
        nargs="?",
        default=DEFAULT_TEXT,
        help=(
            "Input text path (positional, optional; default: "
            f"{DEFAULT_TEXT.relative_to(BENCH_DIR.parent.parent)})"
        ),
    )
    p.add_argument(
        "--names",
        type=Path,
        default=DEFAULT_NAMES_PATH,
        help=(
            "Names list file (one name per line; default: "
            f"{DEFAULT_NAMES_PATH.relative_to(BENCH_DIR.parent.parent)}). "
            "Each names file maps to its own store directory at "
            "contrib/benchmark/store-{stem}/."
        ),
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-build of the store for the chosen names file",
    )
    p.add_argument(
        "--diff-limit",
        type=int,
        default=30,
        help="Max number of names to print in each diff block (default: 30)",
    )
    args = p.parse_args()

    if not args.text.exists():
        sys.exit(f"input text not found: {args.text}")

    names_path = ensure_names(args.names)
    store_uri = store_uri_for(names_path)

    # Time the import step explicitly: cold-start cost matters for short
    # CLI invocations and for procrastinate workers.
    import_t, get_store, io_module = time_import()
    print(f"import:      {_hms(import_t)}")

    build_t, store = ensure_store(
        get_store, io_module, names_path, store_uri, force=args.rebuild
    )
    if build_t > 0:
        print(f"build total: {_hms(build_t)}")

    text = args.text.read_text(encoding="utf-8")
    print()
    print(
        f"input:       {args.text}  ({len(text):,} chars, "
        f"{len(text.split()):,} ws-tokens)"
    )

    print()
    print("=== extract (Aho-Corasick) ===")
    # Warm-up: an empty-string extract triggers the lazy AhoExtractor
    # property, which reads the saved patterns from disk and builds the
    # automaton in memory. We measure that separately so the real
    # extraction time reflects only the per-call scan, not the one-shot
    # load cost.
    t = time.perf_counter()
    store.extract("")
    warmup_t = time.perf_counter() - t
    print(f"  warmup:       {_hms(warmup_t)}")
    t = time.perf_counter()
    ex_mentions = store.extract(text)
    ex_time = time.perf_counter() - t
    ex_names = {m.text for m in ex_mentions}
    print(f"  time:         {_hms(ex_time)}")
    print(f"  mentions:     {len(ex_mentions):8d}")
    print(f"  unique names: {len(ex_names):8d}")

    # Warm-up: the tantivy index is mmap'd, so the first percolate call
    # after a fresh process faults cold pages from `tantivy.db/` into the
    # OS page cache. We do one throw-away percolate against the same
    # text so each per-slop timing below reflects the warm steady state,
    # not the one-off page-fault cost. Reported separately for visibility.
    print()
    print("=== percolate ===")
    t = time.perf_counter()
    store.percolate(text)
    pe_warmup_t = time.perf_counter() - t
    print(f"  warmup:       {_hms(pe_warmup_t)}")

    # Run percolate once per SLOP_VALUES entry. Each run gets its own
    # time + name-set diff against extract so the cost / recall
    # trade-off as slop grows is visible side-by-side.
    pe_results: list[tuple[int, float, list, set[str]]] = []
    for slop in SLOP_VALUES:
        print()
        print(f"=== percolate (slop={slop}) ===")
        t = time.perf_counter()
        pe_mentions = store.percolate(text, slop=slop)
        pe_time = time.perf_counter() - t
        pe_names = {m.text for m in pe_mentions}
        print(f"  time:         {_hms(pe_time)}")
        print(f"  mentions:     {len(pe_mentions):8d}")
        print(f"  unique names: {len(pe_names):8d}")
        pe_results.append((slop, pe_time, pe_mentions, pe_names))

    print()
    print("=== name-set comparison vs extract ===")
    for slop, _, _, pe_names in pe_results:
        both = ex_names & pe_names
        only_extract = ex_names - pe_names
        only_percolate = pe_names - ex_names
        print(
            f"  slop={slop:<2} | in both={len(both):4d} | "
            f"extract-only={len(only_extract):4d} | "
            f"percolate-only={len(only_percolate):4d}"
        )

    # Per-slop diff blocks (only emitted when there's something to show)
    for slop, _, _, pe_names in pe_results:
        only_extract = ex_names - pe_names
        only_percolate = pe_names - ex_names
        if only_extract:
            print()
            print(
                f"-- slop={slop} | found by extract but NOT percolate "
                f"({len(only_extract)}) --"
            )
            print(fmt_names(only_extract, limit=args.diff_limit))
        if only_percolate:
            print()
            print(
                f"-- slop={slop} | found by percolate but NOT extract "
                f"({len(only_percolate)}) --"
            )
            print(fmt_names(only_percolate, limit=args.diff_limit))

    return 0


if __name__ == "__main__":
    sys.exit(main())
