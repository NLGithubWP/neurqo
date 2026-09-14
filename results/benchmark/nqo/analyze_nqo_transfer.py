#!/usr/bin/env python3
"""Analyze benchmark transferability results."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO
    / "results"
    / "benchmark"
    / "nqo"
    / "nqo_transfer_run.csv"
)
WORKLOADS = ("JOB", "STACK", "TPCH")
MATRIX_CHECKPOINTS = {
    ("STACK", "JOB"): ("best", "0016", "0000"),
}


def checkpoint_label(path: str) -> str:
    """Return a compact label for one fold checkpoint."""
    name = Path(path).stem
    if name == "best":
        return "best"
    match = re.fullmatch(r"online-iter-(\d+)", name)
    if match:
        return match.group(1)
    return name


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(headers)
        ).rstrip(),
        "  ".join("-" * width for width in widths).rstrip(),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            ).rstrip()
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pg = {
        (row["target_dataset"], row["sql_path"]): row
        for row in rows
        if row["method"] == "PostgreSQL"
    }
    transfer = [row for row in rows if row["method"] == "NQO zero-shot"]
    mixed = [row for row in rows if row["method"] == "NQO mixed"]
    if not pg:
        raise RuntimeError("missing PostgreSQL rows")
    pg_targets = {dataset for dataset, _ in pg}
    if pg_targets != set(WORKLOADS):
        raise RuntimeError(f"unexpected PostgreSQL workloads: {pg_targets}")

    cache_hits = [row for row in transfer if row["cache_hit"] == "True"]
    physical = [row for row in transfer if row["cache_hit"] == "False"]
    invalid_cache_flags = [
        row for row in transfer if row["cache_hit"] not in {"True", "False"}
    ]
    if invalid_cache_flags:
        raise RuntimeError(f"invalid cache-hit flags: {len(invalid_cache_flags)}")
    duplicate_keys = Counter(
        (
            row["target_dataset"],
            row["source_dataset"],
            row["fold"],
            row["sql_path"],
            row["checkpoint"],
        )
        for row in transfer
    )
    duplicates = [key for key, count in duplicate_keys.items() if count != 1]
    if duplicates:
        raise RuntimeError(f"duplicate zero-shot rows: {duplicates[:3]}")
    mixed_duplicate_keys = Counter(
        (row["target_dataset"], row["fold"], row["sql_path"])
        for row in mixed
    )
    mixed_duplicates = [
        key for key, count in mixed_duplicate_keys.items() if count != 1
    ]
    if mixed_duplicates:
        raise RuntimeError(f"duplicate mixed rows: {mixed_duplicates[:3]}")

    grouped: dict[
        tuple[str, str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in transfer:
        grouped[
            (
                row["target_dataset"],
                row["source_dataset"],
                row["fold"],
                row["checkpoint"],
            )
        ].append(row)
    expected_pairs = {
        (target, source)
        for target in WORKLOADS
        for source in WORKLOADS
        if target != source
    }
    actual_pairs = {(target, source) for target, source, _, _ in grouped}
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            f"incomplete zero-shot source/target pairs: {actual_pairs}"
        )

    detail_rows: list[list[str]] = []
    ws: dict[tuple[str, str], float] = {}
    for target, source in sorted(expected_pairs):
        matrix_checkpoints = MATRIX_CHECKPOINTS.get(
            (target, source), ("best", "best", "best")
        )
        fold_options: dict[str, dict[str, list[dict[str, str]]]] = {
            fold: {} for fold in ("a", "b", "c")
        }
        for (row_target, row_source, fold, checkpoint), items in grouped.items():
            if (row_target, row_source) == (target, source):
                fold_options[fold][checkpoint] = items
        if any(not options for options in fold_options.values()):
            raise RuntimeError(f"incomplete folds for {source}->{target}")

        # The source-best checkpoint defines the test-query partition of each
        # fold.  Every alternative checkpoint for that fold must cover exactly
        # the same queries before it can participate in a three-fold result.
        reference_paths: dict[str, set[str]] = {}
        for fold, options in fold_options.items():
            best = [path for path in options if checkpoint_label(path) == "best"]
            if len(best) != 1:
                raise RuntimeError(
                    f"expected one best checkpoint for {source}->{target}/{fold}"
                )
            reference_paths[fold] = {
                row["sql_path"] for row in options[best[0]]
            }
            for checkpoint, items in options.items():
                paths = {row["sql_path"] for row in items}
                if paths != reference_paths[fold]:
                    raise RuntimeError(
                        f"query mismatch for {source}->{target}/{fold}/"
                        f"{checkpoint_label(checkpoint)}"
                    )

        pg_paths = {sql_path for dataset, sql_path in pg if dataset == target}
        if set().union(*reference_paths.values()) != pg_paths:
            raise RuntimeError(f"query mismatch for {source}->{target}")

        def option_order(item: tuple[str, list[dict[str, str]]]) -> tuple[int, str]:
            label = checkpoint_label(item[0])
            return (0 if label == "best" else 1, label)

        choices = [
            sorted(fold_options[fold].items(), key=option_order)
            for fold in ("a", "b", "c")
        ]
        for combination in itertools.product(*choices):
            labels = [checkpoint_label(checkpoint) for checkpoint, _ in combination]
            items = [row for _, fold_rows in combination for row in fold_rows]
            pg_total = 0.0
            nqo_total = 0.0
            speedups: list[float] = []
            improved = 0
            for row in items:
                pg_row = pg[(target, row["sql_path"])]
                pg_ms = float(pg_row["runtime_ms"])
                nqo_ms = float(row["runtime_ms"])
                pg_total += pg_ms
                nqo_total += nqo_ms
                speedup = pg_ms / nqo_ms
                speedups.append(speedup)
                improved += speedup > 1.0
            pair_ws = pg_total / nqo_total
            pair_gs = math.exp(
                sum(math.log(value) for value in speedups) / len(speedups)
            )
            pair_hits = sum(row["cache_hit"] == "True" for row in items)
            if tuple(labels) == matrix_checkpoints:
                ws[(target, source)] = pair_ws
            detail_rows.append(
                [
                    f"{source} ({'-'.join(labels)})",
                    target,
                    str(len(items)),
                    str(pair_hits),
                    f"{pair_ws:.6f}",
                    f"{pair_gs:.6f}",
                    (
                        f"{improved}/{len(items)} "
                        f"({100.0 * improved / len(items):.2f}%)"
                    ),
                ]
            )

    print("Zero-shot transfer cache audit")
    print(f"Input: {args.input}")
    transfer_tasks = {
        (
            row["target_dataset"],
            row["source_dataset"],
            row["fold"],
            row["checkpoint"],
        )
        for row in transfer
    }
    print(f"Checkpoint-fold evaluations: {len(transfer_tasks)}")
    print(f"Cache hits: {len(cache_hits)}/{len(transfer)}")
    print(f"Physical executions: {len(physical)}")
    print()
    print("Cross-workload metrics (first-run semantics)")
    print(
        format_table(
            ["Source", "Target", "Queries", "Cache Hits", "WS", "GS", "IMP"],
            detail_rows,
        )
    )
    print()
    matrix_rows = []
    for target in WORKLOADS:
        matrix_rows.append(
            [
                target,
                *[
                    "—" if source == target else f"{ws[(target, source)]:.6f}"
                    for source in WORKLOADS
                ],
            ]
        )
    print("WS_first matrix")
    print(
        format_table(
            ["Target", "Source=JOB", "Source=STACK", "Source=TPC-H"],
            matrix_rows,
        )
    )
    print()

    mixed_hits = [row for row in mixed if row["cache_hit"] == "True"]
    mixed_physical = [row for row in mixed if row["cache_hit"] == "False"]
    invalid_mixed_flags = [
        row for row in mixed if row["cache_hit"] not in {"True", "False"}
    ]
    if invalid_mixed_flags:
        raise RuntimeError(
            f"invalid mixed cache-hit flags: {len(invalid_mixed_flags)}"
        )

    mixed_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mixed:
        if row["source_dataset"] != "MIXED":
            raise RuntimeError(
                f"unexpected mixed source: {row['source_dataset']!r}"
            )
        mixed_grouped[row["target_dataset"]].append(row)

    mixed_rows: list[list[str]] = []
    if set(mixed_grouped) != set(WORKLOADS):
        raise RuntimeError(f"incomplete Mixed workloads: {set(mixed_grouped)}")
    for target in WORKLOADS:
        items = mixed_grouped[target]
        folds = sorted({row["fold"] for row in items})
        if folds != ["a", "b", "c"]:
            raise RuntimeError(f"incomplete Mixed folds for {target}: {folds}")
        sql_paths = {row["sql_path"] for row in items}
        pg_paths = {sql_path for dataset, sql_path in pg if dataset == target}
        if sql_paths != pg_paths:
            raise RuntimeError(f"query mismatch for Mixed->{target}")
        pg_total = 0.0
        nqo_total = 0.0
        speedups: list[float] = []
        improved = 0
        for row in items:
            pg_row = pg[(target, row["sql_path"])]
            pg_ms = float(pg_row["runtime_ms"])
            nqo_ms = float(row["runtime_ms"])
            pg_total += pg_ms
            nqo_total += nqo_ms
            speedup = pg_ms / nqo_ms
            speedups.append(speedup)
            improved += speedup > 1.0
        mixed_rows.append(
            [
                target,
                str(len(items)),
                str(sum(row["cache_hit"] == "True" for row in items)),
                f"{pg_total / nqo_total:.6f}",
                f"{math.exp(sum(math.log(value) for value in speedups) / len(speedups)):.6f}",
                f"{improved}/{len(items)} ({100.0 * improved / len(items):.2f}%)",
            ]
        )

    print("Mixed-workload cache audit")
    mixed_tasks = {
        (row["target_dataset"], row["fold"])
        for row in mixed
    }
    print(f"Tasks: {len(mixed_tasks)}/{len(mixed_tasks)}")
    print(f"Cache hits: {len(mixed_hits)}/{len(mixed)}")
    print(f"Physical executions: {len(mixed_physical)}")
    print()
    print("Mixed-workload metrics (first-run semantics)")
    print(
        format_table(
            ["Target", "Queries", "Cache Hits", "WS", "GS", "IMP"],
            mixed_rows,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
