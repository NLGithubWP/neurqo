#!/usr/bin/env python3
"""Summarize versioned NQO learning traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("nqo_learning_trace.csv")
DEFAULT_TIME_INPUT = Path(__file__).with_name("nqo_learning_time.csv")
FOLDS = ("a", "b", "c")
PROTOCOLS = {
    "TPCH": ("random",),
    "JOB": ("base_query", "leave_one_out", "random"),
    "STACK": ("base_query", "leave_one_out", "random"),
}
PROTOCOL_LABELS = {
    "base_query": "Base-query",
    "leave_one_out": "Leave-one-out",
    "random": "Random",
}


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            row["iteration"] = int(raw["iteration"])
            row["query_count"] = int(raw["query_count"])
            row["pg_total_ms"] = float(raw["pg_total_ms"])
            row["nqo_total_ms"] = float(raw["nqo_total_ms"])
            row["ws"] = float(raw["ws"])
            row["inverse_ws"] = float(raw["inverse_ws"])
            rows.append(row)
    return rows


def load_time_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            row["iteration"] = int(raw["iteration"])
            row["elapsed_min"] = float(raw["elapsed_min"])
            rows.append(row)
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("empty learning trace")
    keys = [
        (row["dataset"], row["protocol"], row["fold"], row["iteration"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate dataset/protocol/fold/iteration rows")
    measured = [row for row in rows if int(row["iteration"]) >= 0]
    for dataset, protocols in PROTOCOLS.items():
        for protocol in protocols:
            folds = {
                str(row["fold"])
                for row in rows
                if row["dataset"] == dataset and row["protocol"] == protocol
            }
            if folds != set(FOLDS):
                raise RuntimeError(f"missing folds: {dataset}/{protocol}: {folds}")
    for row in measured:
        if str(row["checkpoint"]).endswith("/best.pt"):
            continue
        if str(row["checkpoint"]).endswith("/learning_curve.jsonl"):
            continue
        cache_hits = int(str(row["cache_hits"]))
        if not 0 < cache_hits <= row["query_count"]:
            raise RuntimeError(
                f"invalid replay count: {row['dataset']}/"
                f"{row['protocol']}/{row['fold']}/{row['iteration']}"
            )


def validate_time_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("empty elapsed-time input")
    keys = [
        (row["dataset"], row["protocol"], row["iteration"]) for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate dataset/protocol/iteration elapsed-time rows")
    for dataset, protocols in PROTOCOLS.items():
        iteration_sets = []
        for protocol in protocols:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["protocol"] == protocol
            ]
            if not selected:
                raise RuntimeError(f"missing elapsed times: {dataset}/{protocol}")
            ordered = sorted(selected, key=lambda row: int(row["iteration"]))
            elapsed = [float(row["elapsed_min"]) for row in ordered]
            if any(right < left for left, right in zip(elapsed, elapsed[1:])):
                raise RuntimeError(f"non-monotonic elapsed times: {dataset}/{protocol}")
            iteration_sets.append({int(row["iteration"]) for row in selected})
        if any(values != iteration_sets[0] for values in iteration_sets[1:]):
            raise RuntimeError(f"misaligned elapsed iterations: {dataset}")


def group_rows(
    rows: list[dict[str, object]], dataset: str, protocol: str
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["dataset"] == dataset and row["protocol"] == protocol
    ]


def iteration_label(dataset: str, protocol: str, iteration: int) -> str:
    return str(iteration)


def best_so_far(
    rows: list[dict[str, object]], cutoff: int
) -> tuple[float, dict[str, dict[str, object]]]:
    chosen: dict[str, dict[str, object]] = {}
    for fold in FOLDS:
        candidates = [
            row
            for row in rows
            if row["fold"] == fold
            and int(row["iteration"]) <= cutoff
        ]
        if not candidates:
            raise RuntimeError(f"no candidate at cutoff {cutoff} for fold {fold}")
        chosen[fold] = min(candidates, key=lambda row: float(row["nqo_total_ms"]))
    pg_total = sum(float(row["pg_total_ms"]) for row in chosen.values())
    nqo_total = sum(float(row["nqo_total_ms"]) for row in chosen.values())
    return nqo_total / pg_total, chosen


def fold_best_so_far(
    rows: list[dict[str, object]], fold: str, cutoff: int
) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if row["fold"] == fold
        and int(row["iteration"]) <= cutoff
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["nqo_total_ms"]))


def print_checkpoint_tables(rows: list[dict[str, object]]) -> None:
    print("CHECKPOINT RESULTS")
    print(
        "Fold columns report each fold's best-so-far WS = PG / NQO; "
        "the final column reports the cross-fold best-so-far 1/WS."
    )
    for dataset, protocols in PROTOCOLS.items():
        for protocol in protocols:
            selected = group_rows(rows, dataset, protocol)
            iterations = sorted({int(row["iteration"]) for row in selected})
            print(f"\n{dataset} / {PROTOCOL_LABELS[protocol]}")
            print(
                f"{'Iteration':<18} {'Fold-a WS':>12} {'Fold-b WS':>12} "
                f"{'Fold-c WS':>12} {'Best-so-far 1/WS':>20}"
            )
            print("-" * 80)
            for iteration in iterations:
                fold_best = [
                    fold_best_so_far(selected, fold, iteration) for fold in FOLDS
                ]
                fold_values = [
                    f"{float(row['ws']):.6f}" if row is not None else "—"
                    for row in fold_best
                ]
                try:
                    best, _ = best_so_far(selected, iteration)
                    best_text = f"{best:.6f}"
                except RuntimeError:
                    best_text = "—"
                label = iteration_label(dataset, protocol, iteration)
                print(
                    f"{label:<18} {fold_values[0]:>12} {fold_values[1]:>12} "
                    f"{fold_values[2]:>12} {best_text:>20}"
                )


def best_at(
    rows: list[dict[str, object]], iteration: int
) -> float | None:
    try:
        value, _ = best_so_far(rows, iteration)
    except RuntimeError:
        return None
    return value


def format_best(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def print_plot_tables(
    rows: list[dict[str, object]], time_rows: list[dict[str, object]]
) -> None:
    print("\n\nPLOT DATA: BEST-SO-FAR 1/WS")
    for dataset, protocols in PROTOCOLS.items():
        print(f"\n{dataset}")
        if dataset == "TPCH":
            selected = group_rows(rows, dataset, "random")
            source_table = sorted(
                (
                    row
                    for row in time_rows
                    if row["dataset"] == dataset and row["protocol"] == "random"
                ),
                key=lambda row: int(row["iteration"]),
            )
            print(
                f"{'Training iteration':<18} {'Elapsed Time (min)':>18} "
                f"{'Best-so-far 1/WS':>20}"
            )
            print("-" * 58)
            for source in source_table:
                iteration = int(source["iteration"])
                print(
                    f"{iteration:<18} {float(source['elapsed_min']):>18.3f}  "
                    f"{format_best(best_at(selected, iteration))}"
                )
            continue

        print(
            f"{'Training iteration':<18} {'Base elapsed (min)':>18} "
            f"{'Base best-so-far 1/WS':>24} {'LOO elapsed (min)':>18} "
            f"{'LOO best-so-far 1/WS':>24} {'Random elapsed (min)':>21} "
            f"{'Random best-so-far 1/WS':>26}"
        )
        print("-" * 158)
        grouped = {
            protocol: group_rows(rows, dataset, protocol)
            for protocol in protocols
        }
        time_by_key = {
            (str(row["protocol"]), int(row["iteration"])): row
            for row in time_rows
            if row["dataset"] == dataset
        }
        iterations = sorted(
            int(row["iteration"])
            for row in time_rows
            if row["dataset"] == dataset and row["protocol"] == "base_query"
        )
        for iteration in iterations:
            values: list[str] = []
            for protocol in ("base_query", "leave_one_out", "random"):
                elapsed = float(time_by_key[(protocol, iteration)]["elapsed_min"])
                values.append(f"{elapsed:.3f}")
                values.append(format_best(best_at(grouped[protocol], iteration)))
            label = str(iteration)
            print(
                f"{label:<18} {values[0]:>18} {values[1]:>24} "
                f"{values[2]:>18} {values[3]:>24} "
                f"{values[4]:>21} {values[5]:>26}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--time-input", type=Path, default=DEFAULT_TIME_INPUT)
    args = parser.parse_args()
    rows = load_rows(args.input)
    time_rows = load_time_rows(args.time_input)
    validate(rows)
    validate_time_rows(time_rows)
    print("NQO LEARNING TRACE")
    estimated = sum(int(row["iteration"]) < 0 for row in rows)
    print(f"rows={len(rows)} estimated_init={estimated} measured={len(rows) - estimated}")
    print(
        "Negative iterations are estimated init points with shrunken fold "
        "variation; iteration 0 onward is measured."
    )
    print_checkpoint_tables(rows)
    print_plot_tables(rows, time_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
