#!/usr/bin/env python3
"""Print NQO benchmark tables from nqo_runs.csv; never write derived data."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "nqo_runs.csv"
DATASET_NAMES = {"job": "JOB", "stack": "STACK", "tpch": "TPCH"}
DATASET_LABELS = {"JOB": "JOB", "STACK": "STACK", "TPCH": "TPC-H"}
PROTOCOLS = {
    "JOB": ("base_query", "leave_one_out", "random"),
    "STACK": ("base_query", "leave_one_out", "random"),
    "TPCH": ("random",),
}
PROTOCOL_LABELS = {
    "base_query": "Base-query",
    "leave_one_out": "Leave-one-out",
    "random": "Random",
}
RAW_FIELDS = (
    "dataset",
    "protocol",
    "fold",
    "sql_path",
    "method",
    "runtime_ms",
    "inference_ms",
    "status",
    "result_hash",
    "result_rows",
    "cache_id",
    "trajectory_hash",
    "checkpoint",
    "actions_json",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise RuntimeError(
                f"unexpected columns in {path.name}: {reader.fieldnames}"
            )
        rows = list(reader)
    keys = [
        (
            row["dataset"],
            row["protocol"],
            row["fold"],
            row["sql_path"],
            row["method"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("nqo_runs.csv contains duplicate result keys")
    return rows


def pg_map(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], float]:
    result = {}
    for row in rows:
        if row["method"] != "PostgreSQL":
            continue
        key = (row["dataset"], row["sql_path"])
        if key in result:
            raise RuntimeError(f"duplicate PostgreSQL baseline: {key}")
        result[key] = float(row["runtime_ms"])
    return result


def method_protocol_rows(
    rows: list[dict[str, str]], method: str, protocol: str
) -> list[dict[str, str]]:
    exact = [
        row
        for row in rows
        if row["method"] == method and row["protocol"] == protocol
    ]
    if exact:
        return exact
    return [
        row
        for row in rows
        if row["method"] == method and not row["protocol"]
    ]


def runtime(row: dict[str, str], without_inference: bool) -> float:
    measured = float(row["runtime_ms"])
    if without_inference and row["status"] == "ok":
        measured -= float(row["inference_ms"] or 0.0)
    if measured <= 0.0:
        raise ValueError(
            f"non-positive runtime for {row['method']}/{row['sql_path']}"
        )
    return measured


def metrics(
    rows: list[dict[str, str]],
    pg: dict[tuple[str, str], float],
    *,
    without_inference: bool,
) -> tuple[float, float, int, int]:
    pairs = []
    for row in rows:
        key = (row["dataset"], row["sql_path"])
        if key not in pg:
            raise KeyError(f"missing PostgreSQL baseline for {key}")
        pairs.append((pg[key], runtime(row, without_inference)))
    if not pairs:
        raise ValueError("cannot compute metrics for an empty result set")
    ws = sum(base for base, _ in pairs) / sum(value for _, value in pairs)
    gs = math.exp(
        statistics.fmean(math.log(base / value) for base, value in pairs)
    )
    improved = sum(value < base for base, value in pairs)
    return ws, gs, improved, len(pairs)


def metric_cell(values: tuple[float, float, int, int] | None) -> str:
    if values is None:
        return "—"
    ws, gs, improved, total = values
    return (
        f"{ws:.6f} / {gs:.6f} / "
        f"{improved}/{total} ({100.0 * improved / total:.2f}%)"
    )


def print_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    right_aligned: set[int] | None = None,
) -> None:
    right_aligned = right_aligned or set()
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: tuple[str, ...]) -> str:
        cells = []
        for index, value in enumerate(row):
            if index in right_aligned:
                cells.append(value.rjust(widths[index]))
            else:
                cells.append(value.ljust(widths[index]))
        return "  ".join(cells).rstrip()

    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def print_overall(
    dataset: str,
    rows: list[dict[str, str]],
    pg: dict[tuple[str, str], float],
) -> None:
    protocols = PROTOCOLS[dataset]
    methods = sorted(
        {row["method"] for row in rows if row["method"] != "PostgreSQL"},
        key=lambda name: (name != "NQO", name.lower()),
    )
    print("1. Overall performance\n")
    print(
        "Each result is WS / GS / Imp, using the paired measurements stored "
        "in nqo_runs.csv.\n"
    )
    table_rows = []
    for method in methods:
        cells = []
        has_inference = False
        selected = {}
        for protocol in protocols:
            current = method_protocol_rows(rows, method, protocol)
            selected[protocol] = current
            has_inference |= any(float(row["inference_ms"] or 0.0) > 0.0 for row in current)
            cells.append(
                metric_cell(
                    metrics(current, pg, without_inference=False)
                    if current
                    else None
                )
            )
        table_rows.append((method, *cells))
        if has_inference:
            cells = [
                metric_cell(
                    metrics(selected[protocol], pg, without_inference=True)
                    if selected[protocol]
                    else None
                )
                for protocol in protocols
            ]
            table_rows.append((f"{method} w/o inference time", *cells))
    print_table(
        ("Method", *(PROTOCOL_LABELS[p] for p in protocols)),
        table_rows,
        right_aligned=set(range(1, len(protocols) + 1)),
    )
    print()


ACTION_KEYS = (
    "high_stop",
    "high_split",
    "select_alpha_0",
    "select_alpha_0_5",
    "select_alpha_1",
    "search_default",
    "search_top5",
    "search_top10",
    "low_none",
    "low_lip_selective",
    "low_lip_full",
    "low_aja_conservative",
    "low_aja_aggressive",
    "low_lip_selective_aja_conservative",
)


def count_actions(payload: str) -> dict[str, int]:
    counts = {key: 0 for key in ACTION_KEYS}
    if not payload:
        return counts
    low_labels = {
        ("none", "none"): "low_none",
        ("none", "selective"): "low_lip_selective",
        ("none", "full"): "low_lip_full",
        ("conservative", "none"): "low_aja_conservative",
        ("aggressive", "none"): "low_aja_aggressive",
        (
            "conservative",
            "selective",
        ): "low_lip_selective_aja_conservative",
    }
    actions = json.loads(payload)
    if not isinstance(actions, list):
        raise ValueError("actions_json must contain a list")
    for item in actions:
        phase = str(item["phase"])
        action = dict(item["action"])
        if phase == "high":
            key = f"high_{action.get('high_action')}"
        elif phase == "select":
            alpha = action.get("schedule_alpha")
            if alpha is None:
                continue
            if abs(float(alpha)) < 1e-9:
                key = "select_alpha_0"
            elif abs(float(alpha) - 0.5) < 1e-9:
                key = "select_alpha_0_5"
            elif abs(float(alpha) - 1.0) < 1e-9:
                key = "select_alpha_1"
            else:
                raise ValueError(f"unsupported Select action: {action}")
        elif phase == "search":
            strategy = str(action.get("search_strategy") or "default")
            search_k = int(action.get("search_k") or 1)
            if strategy == "default":
                key = "search_default"
            elif strategy == "topk" and search_k in {5, 10}:
                key = f"search_top{search_k}"
            else:
                raise ValueError(f"unsupported Search action: {action}")
        elif phase == "low":
            pair = (
                str(action.get("execution_action") or "none"),
                str(action.get("lip_action") or "none"),
            )
            key = low_labels.get(pair, "")
        else:
            raise ValueError(f"unsupported action phase: {phase}")
        if key not in counts:
            raise ValueError(f"unsupported action: phase={phase}, action={action}")
        counts[key] += 1
    return counts


DISPLAY_ACTIONS = (
    ("High", "Stop", "high_stop"),
    ("High", "Split", "high_split"),
    ("Select", "α=0.0", "select_alpha_0"),
    ("Select", "α=0.5", "select_alpha_0_5"),
    ("Select", "α=1.0", "select_alpha_1"),
    ("Search", "Default", "search_default"),
    ("Search", "TOPK (top5)", "search_top5"),
    ("Low", "None", "low_none"),
    ("Low", "Filter (LIP)", "low_lip_selective"),
    ("Low", "Ajoin (AJA)", "low_aja_conservative"),
    (
        "Low",
        "Filter + Ajoin",
        "low_lip_selective_aja_conservative",
    ),
)


def print_action_frequency(dataset: str, rows: list[dict[str, str]]) -> None:
    protocols = PROTOCOLS[dataset]
    by_protocol = {}
    denominators = {}
    for protocol in protocols:
        counts = {key: 0 for key in ACTION_KEYS}
        for row in method_protocol_rows(rows, "NQO", protocol):
            for key, value in count_actions(row["actions_json"]).items():
                counts[key] += value
        by_protocol[protocol] = counts
        denominators[protocol] = {
            head: sum(
                counts[key]
                for current_head, _, key in DISPLAY_ACTIONS
                if current_head == head
            )
            for head in ("High", "Select", "Search", "Low")
        }

    print("2. Action Frequence\n")
    print(
        "Counts cover the complete multi-round trajectory. Frequencies are "
        "normalized independently within each head.\n"
    )
    for protocol in protocols:
        values = denominators[protocol]
        print(
            f"{PROTOCOL_LABELS[protocol]} denominators: "
            + ", ".join(f"{head}={values[head]}" for head in values)
        )
    print()
    if dataset == "TPCH":
        protocol = "random"
        table_rows = []
        for head, label, key in DISPLAY_ACTIONS:
            if head == "Select":
                continue
            denominator = denominators[protocol][head]
            count = by_protocol[protocol][key]
            frequency = count / denominator if denominator else 0.0
            table_rows.append((head, label, str(count), f"{frequency:.4f}"))
        print_table(
            ("Head", "Action", "Count", "Frequency"),
            table_rows,
            right_aligned={2, 3},
        )
        print()
        return

    table_rows = []
    for head, label, key in DISPLAY_ACTIONS:
        frequencies = []
        cells = []
        for protocol in protocols:
            denominator = denominators[protocol][head]
            value = by_protocol[protocol][key] / denominator if denominator else 0.0
            frequencies.append(value)
            cells.append(f"{value:.4f}")
        mean = statistics.fmean(frequencies) if frequencies else 0.0
        table_rows.append((head, label, *cells, f"{mean:.4f}"))
    print_table(
        (
            "Head",
            "Action",
            *(PROTOCOL_LABELS[p] for p in protocols),
            "Mean frequency",
        ),
        table_rows,
        right_aligned=set(range(2, len(protocols) + 3)),
    )
    print()


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else float("nan")


def print_per_query(
    dataset: str,
    rows: list[dict[str, str]],
    pg: dict[tuple[str, str], float],
) -> None:
    protocols = PROTOCOLS[dataset]
    protocol_values: dict[str, dict[str, tuple[float, float, float]]] = {}
    for protocol in protocols:
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in method_protocol_rows(rows, "NQO", protocol):
            baseline = pg[(dataset, row["sql_path"])]
            query_id = Path(row["sql_path"]).stem
            grouped[query_id].append((baseline, runtime(row, False)))
        protocol_values[protocol] = {
            query_id: (
                statistics.fmean(base for base, _ in values),
                statistics.fmean(value for _, value in values),
                statistics.fmean(base - value for base, value in values),
            )
            for query_id, values in grouped.items()
        }

    print("3. Per-Query Performance\n")
    if dataset == "TPCH":
        print(
            "Δt(q) = tPG(q) - tNQO(q), in seconds; positive values mean NQO "
            "is faster.\n"
        )
    else:
        print(
            "Δt(q) = tPG(q) - tNQO(q), in seconds; positive values mean NQO "
            "is faster. Duplicate query IDs are averaged inside a protocol "
            "before the cross-protocol mean and min–max range are computed.\n"
        )
    query_ids = sorted(
        set().union(*(set(values) for values in protocol_values.values())),
        key=natural_key,
    )
    if dataset == "TPCH":
        table_rows = []
        for query_id in query_ids:
            baseline, measured, delta = protocol_values["random"][query_id]
            table_rows.append(
                (
                    f"Q{query_id}",
                    f"{baseline / 1000.0:.6f}",
                    f"{measured / 1000.0:.6f}",
                    f"{delta / 1000.0:+.6f}",
                )
            )
        print_table(
            ("Query", "PG (s)", "NQO (s)", "Δt (s)"),
            table_rows,
            right_aligned={0, 1, 2, 3},
        )
        print()
        return

    correlations = []
    for left, right in itertools.combinations(protocols, 2):
        common = sorted(
            set(protocol_values[left]) & set(protocol_values[right]),
            key=natural_key,
        )
        correlations.append(
            pearson(
                [protocol_values[left][query_id][2] for query_id in common],
                [protocol_values[right][query_id][2] for query_id in common],
            )
        )
    finite = [value for value in correlations if math.isfinite(value)]
    if finite:
        print(
            f"Pairwise protocol Δt Pearson correlation: "
            f"{min(finite):.4f}–{max(finite):.4f}.\n"
        )

    table_rows = []
    for query_id in query_ids:
        values = [
            protocol_values[protocol][query_id]
            for protocol in protocols
            if query_id in protocol_values[protocol]
        ]
        mean_pg = statistics.fmean(value[0] for value in values) / 1000.0
        mean_nqo = statistics.fmean(value[1] for value in values) / 1000.0
        deltas = [value[2] / 1000.0 for value in values]
        table_rows.append(
            (
                query_id,
                f"{mean_pg:.6f}",
                f"{mean_nqo:.6f}",
                f"{statistics.fmean(deltas):+.6f}",
                f"{min(deltas):+.6f}",
                f"{max(deltas):+.6f}",
            )
        )
    print_table(
        (
            "Query",
            "Mean PG (s)",
            "Mean NQO (s)",
            "Mean Δt (s)",
            "Min Δt (s)",
            "Max Δt (s)",
        ),
        table_rows,
        right_aligned=set(range(6)),
    )
    print()


def print_dataset(
    dataset: str,
    all_rows: list[dict[str, str]],
    pg: dict[tuple[str, str], float],
) -> bool:
    rows = [row for row in all_rows if row["dataset"] == dataset]
    method_rows = [row for row in rows if row["method"] != "PostgreSQL"]
    print("=" * 88)
    print(DATASET_LABELS[dataset])
    print("=" * 88)
    print()
    if not method_rows:
        print(
            f"No NQO result rows are currently available for "
            f"{DATASET_LABELS[dataset]}.\n"
        )
        return False
    print_overall(dataset, rows, pg)
    nqo_rows = [row for row in method_rows if row["method"] == "NQO"]
    if nqo_rows:
        print_action_frequency(dataset, rows)
        print_per_query(dataset, rows, pg)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print Overall, Action Frequency, and per-query NQO tables"
    )
    parser.add_argument("dataset", choices=("job", "stack", "tpch", "all"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    rows = read_rows(args.input)
    pg = pg_map(rows)
    selected = (
        tuple(DATASET_NAMES.values())
        if args.dataset == "all"
        else (DATASET_NAMES[args.dataset],)
    )
    available = [print_dataset(dataset, rows, pg) for dataset in selected]
    return 0 if any(available) else 2


if __name__ == "__main__":
    raise SystemExit(main())
