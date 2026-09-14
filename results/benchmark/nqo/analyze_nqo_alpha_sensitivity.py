#!/usr/bin/env python3
"""Summarize JOB Random fixed-alpha and learned-alpha results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "nqo_alpha_sensitivity.csv"
POLICIES = (
    ("fixed_0", "Fixed alpha=0.0"),
    ("fixed_0_5", "Fixed alpha=0.5"),
    ("fixed_1", "Fixed alpha=1.0"),
    ("learned", "Learned alpha_t"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "fold",
        "sql_path",
        "schedule_policy",
        "runtime_ms",
        "pg_runtime_ms",
        "status",
        "cache_hit",
        "actions_json",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def metrics(rows: list[dict[str, str]]) -> tuple[float, float, int, int, float]:
    pairs = [
        (float(row["pg_runtime_ms"]), float(row["runtime_ms"]))
        for row in rows
    ]
    ws = sum(pg for pg, _ in pairs) / sum(runtime for _, runtime in pairs)
    gs = math.exp(statistics.fmean(math.log(pg / runtime) for pg, runtime in pairs))
    improved = sum(runtime < pg for pg, runtime in pairs)
    return ws, gs, improved, len(pairs), sum(runtime for _, runtime in pairs)


def alpha_counts(rows: list[dict[str, str]]) -> Counter[float]:
    counts: Counter[float] = Counter()
    for row in rows:
        for decision in json.loads(row["actions_json"] or "[]"):
            if decision.get("phase") == "select":
                counts[float(decision["action"]["schedule_alpha"])] += 1
    return counts


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(value) for value in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    rows = read_rows(args.input)
    grouped = {
        policy: [row for row in rows if row["schedule_policy"] == policy]
        for policy, _ in POLICIES
    }
    learned_total = metrics(grouped["learned"])[4]

    print("Alpha scheduling sensitivity on JOB Random\n")
    result_rows = []
    for policy, label in POLICIES:
        ws, gs, improved, total, runtime = metrics(grouped[policy])
        result_rows.append(
            (
                label,
                f"{ws:.6f}",
                f"{gs:.6f}",
                f"{improved}/{total} ({100.0 * improved / total:.2f}%)",
                f"{runtime / 1000.0:.3f}",
                f"{runtime / learned_total:.4f}",
            )
        )
    print_table(
        ("Schedule policy", "WS", "GS", "Imp", "NQO total (s)", "Runtime / learned"),
        result_rows,
    )

    print("\nFold-level WS\n")
    fold_rows = []
    for policy, label in POLICIES:
        values = []
        for fold in ("a", "b", "c"):
            current = [row for row in grouped[policy] if row["fold"] == fold]
            values.append(f"{metrics(current)[0]:.6f}")
        fold_rows.append((label, *values))
    print_table(("Schedule policy", "Fold a", "Fold b", "Fold c"), fold_rows)

    print("\nRealized Select decisions\n")
    frequency_rows = []
    for policy, label in POLICIES:
        counts = alpha_counts(grouped[policy])
        total = sum(counts.values())
        frequency_rows.append(
            (
                label,
                str(total),
                f"{counts[0.0]}/{total}" if total else "0/0",
                f"{counts[0.5]}/{total}" if total else "0/0",
                f"{counts[1.0]}/{total}" if total else "0/0",
            )
        )
    print_table(
        ("Schedule policy", "#Select", "alpha=0.0", "alpha=0.5", "alpha=1.0"),
        frequency_rows,
    )

    fixed_rows = [row for row in rows if row["schedule_policy"] != "learned"]
    hits = sum(row["cache_hit"] == "1" for row in fixed_rows)
    print(
        f"\nFixed-alpha cache coverage: {hits}/{len(fixed_rows)} hits; "
        f"{len(fixed_rows) - hits} physical executions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
