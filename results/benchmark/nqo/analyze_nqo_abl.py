#!/usr/bin/env python3
"""Analyze versioned RL-formulation and state-ablation runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RL = ROOT / "nqo_abl_rl_run.csv"
DEFAULT_STATE = ROOT / "nqo_abl_sate_run.csv"
DEFAULT_ACTION = ROOT / "nqo_abl_action_run.csv"
MODEL_METHODS = (
    "Full NQO",
    "One-step RL",
    "w/o Query Topology",
    "w/o Plan Topology",
)
ACTION_METHODS = (
    "Full NQO",
    "w/o Query Split",
    "w/o TOPK",
    "w/o Filter",
    "w/o AJoin",
)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(value) for value in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "  ".join("-" * width for width in widths).rstrip()
    lines = [
        "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(headers)
        ).rstrip(),
        separator,
    ]
    lines.extend(
        "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def load(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_rows(
    rows: list[dict[str, str]],
    method: str,
    datasets: tuple[str, ...] = ("JOB", "STACK"),
) -> dict[tuple[str, str], dict[str, str]]:
    items = [
        row
        for row in rows
        if row["method"] == method and row["dataset"] in datasets
    ]
    keys = [(row["dataset"], row["sql_path"]) for row in items]
    if not keys:
        raise RuntimeError(f"missing {method} rows")
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate {method} query rows")
    return dict(zip(keys, items))


def require_same_queries(
    reference: dict[tuple[str, str], dict[str, str]],
    candidate: dict[tuple[str, str], dict[str, str]],
    label: str,
) -> None:
    if set(candidate) != set(reference):
        raise RuntimeError(f"query mismatch for {label}")


def metrics(
    pg: dict[tuple[str, str], dict[str, str]],
    method: dict[tuple[str, str], dict[str, str]],
    dataset: str,
) -> tuple[float, float, int, int]:
    keys = sorted(key for key in pg if key[0] == dataset)
    if not keys:
        raise RuntimeError(f"missing PostgreSQL rows for {dataset}")
    method_keys = {key for key in method if key[0] == dataset}
    if method_keys != set(keys):
        raise RuntimeError(f"query mismatch for {dataset}")
    pg_total = 0.0
    nqo_total = 0.0
    speedups: list[float] = []
    improved = 0
    for key in keys:
        pg_ms = float(pg[key]["runtime_ms"])
        nqo_ms = float(method[key]["runtime_ms"])
        pg_total += pg_ms
        nqo_total += nqo_ms
        speedup = pg_ms / nqo_ms
        speedups.append(speedup)
        improved += speedup > 1.0
    return (
        pg_total / nqo_total,
        math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        improved,
        len(keys),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", type=Path, default=DEFAULT_RL)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--action", type=Path, default=DEFAULT_ACTION)
    args = parser.parse_args()
    rl_rows = load(args.rl)
    state_rows = load(args.state)
    action_rows = load(args.action)

    model_datasets = ("JOB", "STACK")
    state_datasets = ("JOB", "STACK", "TPCH")
    rl_pg = method_rows(rl_rows, "PostgreSQL", model_datasets)
    state_pg = method_rows(state_rows, "PostgreSQL", state_datasets)
    if {
        key: value for key, value in state_pg.items() if key[0] in model_datasets
    } != rl_pg:
        raise RuntimeError("RL and state CSVs use different PostgreSQL baselines")
    rl_full = method_rows(rl_rows, "Full NQO", model_datasets)
    state_full = method_rows(state_rows, "Full NQO", state_datasets)
    if {
        key: value for key, value in state_full.items() if key[0] in model_datasets
    } != rl_full:
        raise RuntimeError("RL and state CSVs contain different Full NQO rows")

    action_datasets = ("JOB", "STACK", "TPCH")
    action_pg = method_rows(action_rows, "PostgreSQL", action_datasets)
    action_full = method_rows(action_rows, "Full NQO", action_datasets)
    if action_pg != state_pg:
        raise RuntimeError("state and action CSVs use different baselines")
    if action_full != state_full:
        raise RuntimeError("state and action CSVs contain different Full NQO rows")

    combined = {
        "Full NQO": state_full,
        "One-step RL": method_rows(rl_rows, "One-step RL", model_datasets),
        "w/o Query Topology": method_rows(
            state_rows, "w/o Query Topology", state_datasets
        ),
        "w/o Plan Topology": method_rows(
            state_rows, "w/o Plan Topology", state_datasets
        ),
    }
    action_combined = {
        "Full NQO": action_full,
        "w/o Query Split": method_rows(
            action_rows, "w/o Query Split", ("JOB", "STACK")
        ),
        "w/o TOPK": method_rows(action_rows, "w/o TOPK", action_datasets),
        "w/o Filter": method_rows(
            action_rows, "w/o Filter", action_datasets
        ),
        "w/o AJoin": method_rows(action_rows, "w/o AJoin", action_datasets),
    }
    for method_name, selected in combined.items():
        expected = rl_pg if method_name == "One-step RL" else state_pg
        require_same_queries(expected, selected, method_name)
    for method_name, selected in action_combined.items():
        expected = {
            key: value
            for key, value in action_pg.items()
            if method_name != "w/o Query Split" or key[0] != "TPCH"
        }
        require_same_queries(expected, selected, method_name)
    duplicates = Counter(
        (row["dataset"], row["fold"], row["sql_path"], row["method"])
        for row in rl_rows + state_rows
        if row["method"] not in {"PostgreSQL", "Full NQO"}
    )
    bad = [key for key, count in duplicates.items() if count != 1]
    if bad:
        raise RuntimeError(f"duplicate or missing ablation rows: {bad[:3]}")
    action_duplicates = Counter(
        (row["dataset"], row["fold"], row["sql_path"], row["method"])
        for row in action_rows
        if row["method"] not in {"PostgreSQL", "Full NQO"}
    )
    action_bad = [
        key for key, count in action_duplicates.items() if count != 1
    ]
    if action_bad:
        raise RuntimeError(
            f"duplicate or missing action-ablation rows: {action_bad[:3]}"
        )

    rl_ablation = [row for row in rl_rows if row["method"] == "One-step RL"]
    state_ablation = [
        row
        for row in state_rows
        if row["method"] in {"w/o Query Topology", "w/o Plan Topology"}
    ]
    action_ablation = [
        row
        for row in action_rows
        if row["method"] not in {"PostgreSQL", "Full NQO"}
    ]
    rl_tasks = {
        (row["dataset"], row["fold"], row["method"])
        for row in rl_ablation
    }
    state_tasks = {
        (row["dataset"], row["fold"], row["method"])
        for row in state_ablation
    }
    action_tasks = {
        (row["dataset"], row["fold"], row["method"])
        for row in action_ablation
    }
    print("RL formulation cache audit")
    print(f"Tasks: {len(rl_tasks)}/{len(rl_tasks)}")
    print(
        f"Cache hits: {sum(row['cache_hit'] == 'True' for row in rl_ablation)}"
        f"/{len(rl_ablation)}"
    )
    print(
        f"Physical executions: "
        f"{sum(row['cache_hit'] == 'False' for row in rl_ablation)}"
    )
    print()
    print("State ablation cache audit")
    print(f"Tasks: {len(state_tasks)}/{len(state_tasks)}")
    print(
        f"Cache hits: {sum(row['cache_hit'] == 'True' for row in state_ablation)}"
        f"/{len(state_ablation)}"
    )
    print(
        f"Physical executions: "
        f"{sum(row['cache_hit'] == 'False' for row in state_ablation)}"
    )
    print()
    print("Action ablation cache audit")
    print(f"Tasks: {len(action_tasks)}/{len(action_tasks)}")
    print(
        f"Cache hits: {sum(row['cache_hit'] == 'True' for row in action_ablation)}"
        f"/{len(action_ablation)}"
    )
    print(
        f"Physical executions: "
        f"{sum(row['cache_hit'] == 'False' for row in action_ablation)}"
    )
    print()
    print("RL Formulation and State Ablation")
    table: list[list[str]] = []
    for method_name in MODEL_METHODS:
        row = [method_name]
        for dataset in state_datasets:
            if dataset == "TPCH" and method_name == "One-step RL":
                row.extend(["—", "—", "—"])
                continue
            ws, gs, improved, count = metrics(
                state_pg, combined[method_name], dataset
            )
            row.extend(
                [
                    f"{ws:.6f}",
                    f"{gs:.6f}",
                    f"{improved}/{count} ({100.0 * improved / count:.2f}%)",
                ]
            )
        table.append(row)
    print(
        format_table(
            [
                "Method",
                "JOB WS",
                "JOB GS",
                "JOB IMP",
                "STACK WS",
                "STACK GS",
                "STACK IMP",
                "TPC-H WS",
                "TPC-H GS",
                "TPC-H IMP",
            ],
            table,
        )
    )
    print()
    print("Action Importance on the Random split")
    action_table: list[list[str]] = []
    for dataset in action_datasets:
        row = ["TPC-H" if dataset == "TPCH" else dataset]
        for method_name in ACTION_METHODS:
            if dataset == "TPCH" and method_name == "w/o Query Split":
                row.append("—")
                continue
            ws, _, _, _ = metrics(
                action_pg, action_combined[method_name], dataset
            )
            row.append(f"{ws:.6f}")
        action_table.append(row)
    print(
        format_table(
            [
                "Dataset",
                "Full NQO WS",
                "w/o Query Split WS",
                "w/o TOPK WS",
                "w/o Filter WS",
                "w/o AJoin WS",
            ],
            action_table,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
