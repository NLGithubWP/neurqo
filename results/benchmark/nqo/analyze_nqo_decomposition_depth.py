#!/usr/bin/env python3
"""Summarize the fixed-depth query-decomposition microbenchmark."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parent / "nqo_decomposition_depth.csv"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(value) for value in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)).rstrip(),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pg = {
        (row["dataset"], row["query_id"]): float(row["runtime_ms"])
        for row in rows
        if row["method"] == "PostgreSQL"
    }
    fixed = [row for row in rows if row["method"] == "Fixed-depth"]
    print("Decomposition-depth microbenchmark")
    print("alpha=0.5, Search=default, Low=none")
    print(f"Cache hits: {sum(row['cache_hit'].lower() == 'true' for row in fixed)}/{len(fixed)}")
    print()

    by_query: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in fixed:
        by_query[(row["dataset"], row["query_id"])].append(row)

    print("Per-query curves")
    detail: list[list[str]] = []
    for (dataset, query_id), query_rows in sorted(by_query.items()):
        for row in sorted(query_rows, key=lambda item: int(item["requested_split_rounds"])):
            detail.append(
                [
                    dataset,
                    query_id,
                    row["requested_split_rounds"],
                    row["actual_split_rounds"],
                    row["executed_units"],
                    f"{float(row['charged_runtime_ms']) / 1000.0:.6f}",
                    f"{float(row['speedup']):.6f}",
                    f"{1.0 / float(row['speedup']):.6f}",
                    f"{float(row['split_total_ms']) / 1000.0:.6f}",
                    f"{float(row['materialized_bytes']) / 1024.0:.2f}",
                    row["status"],
                ]
            )
    print(
        format_table(
            [
                "Dataset",
                "Query",
                "Requested splits",
                "Actual splits",
                "Executed units",
                "Charged runtime (s)",
                "Speedup",
                "Normalized runtime",
                "Split cost (s)",
                "Materialized (KiB)",
                "Status",
            ],
            detail,
        )
    )
    print()

    print("Best depth per query")
    best_table: list[list[str]] = []
    for (dataset, query_id), query_rows in sorted(by_query.items()):
        best = min(query_rows, key=lambda row: float(row["charged_runtime_ms"]))
        deepest = max(query_rows, key=lambda row: int(row["requested_split_rounds"]))
        best_table.append(
            [
                dataset,
                query_id,
                best["requested_split_rounds"],
                best["executed_units"],
                f"{float(best['speedup']):.6f}",
                deepest["requested_split_rounds"],
                f"{float(deepest['speedup']):.6f}",
            ]
        )
    print(
        format_table(
            [
                "Dataset",
                "Query",
                "Best splits",
                "Best units",
                "Best speedup",
                "Deepest splits",
                "Deepest speedup",
            ],
            best_table,
        )
    )
    print()

    print("Workload aggregate at common depths")
    aggregate: list[list[str]] = []
    for dataset in sorted({key[0] for key in by_query}):
        query_ids = sorted(query_id for ds, query_id in by_query if ds == dataset)
        depths = sorted(
            set.intersection(
                *(
                    {
                        int(row["requested_split_rounds"])
                        for row in by_query[(dataset, query_id)]
                    }
                    for query_id in query_ids
                )
            )
        )
        for depth in depths:
            selected = [
                next(
                    row
                    for row in by_query[(dataset, query_id)]
                    if int(row["requested_split_rounds"]) == depth
                )
                for query_id in query_ids
            ]
            pg_total = sum(pg[(dataset, query_id)] for query_id in query_ids)
            method_total = sum(float(row["charged_runtime_ms"]) for row in selected)
            aggregate.append(
                [
                    dataset,
                    str(depth),
                    str(len(query_ids)),
                    f"{method_total / pg_total:.6f}",
                    f"{pg_total / method_total:.6f}",
                    f"{sum(float(row['split_total_ms']) for row in selected) / 1000.0:.6f}",
                ]
            )
    print(
        format_table(
            [
                "Dataset",
                "Requested splits",
                "Queries",
                "Normalized runtime",
                "WS",
                "Split cost (s)",
            ],
            aggregate,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
