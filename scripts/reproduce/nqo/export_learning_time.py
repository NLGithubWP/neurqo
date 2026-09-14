#!/usr/bin/env python3
"""Export cumulative NQO learning times from workload matrix reports."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO / "results" / "benchmark" / "nqo" / "nqo_learning_time.csv"
)
CSV_FIELDS = ("dataset", "protocol", "iteration", "elapsed_min")


def matrix_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use DATASET=path/to/learning_curve.csv")
    dataset, raw_path = value.split("=", 1)
    dataset = dataset.upper()
    if dataset not in {"JOB", "STACK", "TPCH"}:
        raise argparse.ArgumentTypeError(f"unknown dataset: {dataset}")
    return dataset, Path(raw_path)


def load_rows(inputs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows = []
    seen_datasets = set()
    for dataset, path in inputs:
        if dataset in seen_datasets:
            raise ValueError(f"duplicate matrix input for {dataset}")
        seen_datasets.add(dataset)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"protocol", "iteration", "elapsed_training_s"}
            if not required <= set(reader.fieldnames or ()):
                raise RuntimeError(
                    f"matrix curve lacks cumulative timing fields: {path}"
                )
            for source in reader:
                rows.append(
                    {
                        "dataset": dataset,
                        "protocol": source["protocol"],
                        "iteration": int(source["iteration"]),
                        "elapsed_min": (
                            f"{float(source['elapsed_training_s']) / 60.0:.3f}"
                        ),
                    }
                )
    if seen_datasets != {"JOB", "STACK", "TPCH"}:
        raise ValueError("provide exactly one JOB, STACK, and TPCH matrix curve")
    return sorted(
        rows,
        key=lambda row: (row["dataset"], row["protocol"], row["iteration"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=matrix_input,
        action="append",
        required=True,
        help="repeat as JOB=path, STACK=path, and TPCH=path",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists; pass --overwrite: {args.output}")
    rows = load_rows(args.matrix)
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
