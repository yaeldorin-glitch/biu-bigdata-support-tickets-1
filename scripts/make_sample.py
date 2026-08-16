"""Build the committed sample CSV from the full dataset.

The full export is CC BY-NC 4.0 and 26MB, so it is not committed. A stratified
sample is, which keeps the repository runnable out of the box while respecting
the licence and keeping git history small.

Stratified by queue *and* language so the sample preserves the shape of the
corpus: a uniform random 300 rows would likely contain zero "General Inquiry"
tickets (1.4% of the data) and skew the demo.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tickets.core.schema import RAW_COLUMNS  # noqa: E402


def build_sample(source: Path, destination: Path, size: int = 300, seed: int = 42) -> int:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("queue", ""), row.get("language", ""))].append(row)

    rng = random.Random(seed)
    per_group = max(1, size // max(len(groups), 1))

    sampled: list[dict] = []
    for group in sorted(groups):
        candidates = groups[group]
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_group])

    # Top up to the requested size from whatever is left, so a few tiny strata
    # do not leave the sample short.
    if len(sampled) < size:
        chosen = {id(r) for r in sampled}
        remaining = [r for r in rows if id(r) not in chosen]
        rng.shuffle(remaining)
        sampled.extend(remaining[: size - len(sampled)])

    rng.shuffle(sampled)
    sampled = sampled[:size]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row in sampled:
            writer.writerow({column: row.get(column, "") for column in RAW_COLUMNS})

    return len(sampled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw" / "tickets.csv")
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "sample" / "tickets_sample.csv")
    parser.add_argument("--size", type=int, default=300)
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source CSV not found at {args.source}; see data/README.md")

    count = build_sample(args.source, args.destination, args.size)
    print(f"wrote {count} rows to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
