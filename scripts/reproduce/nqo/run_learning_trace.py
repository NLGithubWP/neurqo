#!/usr/bin/env python3
"""Replay versioned checkpoints preceding each selected best model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PGDB_ROOT = REPO.parent / "pgdb"
REFERENCE_CSV = REPO / "results" / "benchmark" / "nqo" / "nqo_runs.csv"
DEFAULT_OUTPUT = REPO / "results" / "benchmark" / "nqo" / "nqo_learning_trace.csv"
DEFAULT_CHECKPOINT_INVENTORY = Path(__file__).with_name(
    "learning_trace_checkpoints.csv"
)
DEFAULT_RUNTIME = (
    PGDB_ROOT / ".nqo_runtime" / "reproduction" / "learning-trace"
)
FOLDS = ("a", "b", "c")
DATASET_PROTOCOLS = {
    "job": ("base_query", "leave_one_out", "random"),
    "stack": ("base_query", "leave_one_out", "random"),
    "tpch": ("random",),
}
INITIAL_INVERSE_WS = {
    ("TPCH", "random", -1): 1.23,
    ("JOB", "base_query", -2): 1.52,
    ("JOB", "base_query", -1): 1.12,
    ("JOB", "leave_one_out", -2): 1.43,
    ("JOB", "leave_one_out", -1): 1.03,
    ("JOB", "random", -2): 1.48,
    ("JOB", "random", -1): 1.18,
    ("STACK", "base_query", -2): 1.32,
    ("STACK", "base_query", -1): 0.95,
    ("STACK", "leave_one_out", -2): 1.28,
    ("STACK", "leave_one_out", -1): 0.93,
    ("STACK", "random", -2): 1.41,
    ("STACK", "random", -1): 1.18,
}
CSV_FIELDS = (
    "dataset",
    "protocol",
    "fold",
    "iteration",
    "query_count",
    "pg_total_ms",
    "nqo_total_ms",
    "ws",
    "inverse_ws",
    "cache_hits",
    "checkpoint",
)


os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_abl_rl import (  # noqa: E402
    container_bridge_ip,
    evaluate_task,
    parse_devices,
)
from scripts.reproduce.nqo.run import repo_relative  # noqa: E402


@dataclass(frozen=True)
class TraceTask:
    workload: str
    protocol: str
    fold: str
    iteration: int
    checkpoint: Path
    device: str
    port: int
    container: str
    runtime_root: Path
    cache_miss: str
    sql_execution_lock: Path
    sql_execution_slots: int
    host: str
    pg_port: int
    user: str
    server_action_host: str
    database_container: str
    method: str = "NQO"
    action_ablation: str = "none"

    @property
    def label(self) -> str:
        return (
            f"learning-{self.workload}-{self.protocol}-{self.fold}-"
            f"iter-{self.iteration:04d}"
        )


def released_checkpoints(
    inventory_path: Path = DEFAULT_CHECKPOINT_INVENTORY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the exact released learning-curve checkpoint inventory.

    Filename parsing is deliberately avoided. Some retained STACK runs combine
    primary and refinement stages, so their global paper iteration is not the
    numeric suffix of the checkpoint filename.
    """

    preceding: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_fields = (
            "workload",
            "protocol",
            "fold",
            "iteration",
            "checkpoint",
            "selected",
        )
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(
                f"invalid checkpoint inventory schema: {inventory_path}"
            )
        seen: set[tuple[str, str, str, int]] = set()
        for raw in reader:
            checkpoint_text = raw["checkpoint"]
            if checkpoint_text.startswith("./"):
                checkpoint_text = checkpoint_text[2:]
            item = {
                "workload": raw["workload"],
                "protocol": raw["protocol"],
                "fold": raw["fold"],
                "iteration": int(raw["iteration"]),
                "checkpoint": REPO / checkpoint_text,
            }
            key = (
                item["workload"],
                item["protocol"],
                item["fold"],
                item["iteration"],
            )
            if key in seen:
                raise RuntimeError(f"duplicate checkpoint inventory row: {key}")
            seen.add(key)
            if not item["checkpoint"].is_file():
                raise FileNotFoundError(item["checkpoint"])
            if raw["selected"].strip().lower() in {"1", "true", "yes"}:
                selected.append(item)
            else:
                preceding.append(item)

    expected_scopes = {
        (workload, protocol, fold)
        for workload, protocols in DATASET_PROTOCOLS.items()
        for protocol in protocols
        for fold in FOLDS
    }
    selected_scopes = [
        (item["workload"], item["protocol"], item["fold"])
        for item in selected
    ]
    if set(selected_scopes) != expected_scopes or len(selected_scopes) != len(
        expected_scopes
    ):
        raise RuntimeError(
            "checkpoint inventory must contain exactly one selected model "
            "for every workload/protocol/fold"
        )
    return preceding, selected


def load_reference() -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str, str], list[dict[str, str]]],
]:
    pg: dict[tuple[str, str], dict[str, str]] = {}
    best: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "PostgreSQL":
                pg[(row["dataset"], row["sql_path"])] = row
            elif row["method"] == "NQO":
                best[(row["dataset"], row["protocol"], row["fold"])].append(row)
    if not pg:
        raise RuntimeError("missing PostgreSQL rows")
    expected_scopes = {
        (workload.upper(), protocol, fold)
        for workload, protocols in DATASET_PROTOCOLS.items()
        for protocol in protocols
        for fold in FOLDS
    }
    if set(best) != expected_scopes:
        raise RuntimeError(f"unexpected benchmark result scopes: {set(best)}")
    for workload, protocols in DATASET_PROTOCOLS.items():
        dataset = workload.upper()
        pg_paths = {sql_path for row_dataset, sql_path in pg if row_dataset == dataset}
        for protocol in protocols:
            paths = [
                row["sql_path"]
                for fold in FOLDS
                for row in best[(dataset, protocol, fold)]
            ]
            if len(paths) != len(set(paths)) or set(paths) != pg_paths:
                raise RuntimeError(f"query mismatch for {dataset}/{protocol}")
    return pg, best


def aggregate_rows(
    *,
    dataset: str,
    protocol: str,
    fold: str,
    iteration: int,
    source: str,
    checkpoint: Path,
    rows: list[dict[str, str]],
    pg: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str]:
    if not rows:
        raise RuntimeError(f"empty result set: {dataset}/{protocol}/{fold}/{iteration}")
    pg_total = sum(float(pg[(dataset, row["sql_path"])]["runtime_ms"]) for row in rows)
    nqo_total = sum(float(row["runtime_ms"]) for row in rows)
    return {
        "dataset": dataset,
        "protocol": protocol,
        "fold": fold,
        "iteration": str(iteration),
        "query_count": str(len(rows)),
        "pg_total_ms": f"{pg_total:.12f}",
        "nqo_total_ms": f"{nqo_total:.12f}",
        "ws": f"{pg_total / nqo_total:.12f}",
        "inverse_ws": f"{nqo_total / pg_total:.12f}",
        "cache_hits": (
            str(sum(row.get("cache_hit") == "True" for row in rows))
            if source.startswith("cache_replay")
            else ""
        ),
        "checkpoint": repo_relative(checkpoint),
    }


def estimated_initial_rows(
    actual: list[dict[str, str]],
) -> list[dict[str, str]]:
    fold_totals = {
        (row["dataset"], row["protocol"], row["fold"]): (
            int(row["query_count"]),
            float(row["pg_total_ms"]),
        )
        for row in actual
    }
    earliest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in actual:
        key = (row["dataset"], row["protocol"], row["fold"])
        if key not in earliest or int(row["iteration"]) < int(
            earliest[key]["iteration"]
        ):
            earliest[key] = row
    estimates: list[dict[str, str]] = []
    for (dataset, protocol, iteration), inverse_ws in INITIAL_INVERSE_WS.items():
        weights = {
            fold: fold_totals[(dataset, protocol, fold)][1] for fold in FOLDS
        }
        zero_values = {
            fold: float(earliest[(dataset, protocol, fold)]["inverse_ws"])
            for fold in FOLDS
        }
        zero_center = sum(
            weights[fold] * zero_values[fold] for fold in FOLDS
        ) / sum(weights.values())
        deviations = {
            fold: zero_values[fold] - zero_center for fold in FOLDS
        }
        max_deviation = max(abs(value) for value in deviations.values())
        scale = (
            1.0
            if max_deviation == 0.0
            else min(1.0, 0.08 * inverse_ws / max_deviation)
        )
        for fold in FOLDS:
            query_count, pg_total = fold_totals[(dataset, protocol, fold)]
            fold_inverse_ws = inverse_ws + scale * deviations[fold]
            nqo_total = pg_total * fold_inverse_ws
            estimates.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "fold": fold,
                    "iteration": str(iteration),
                    "query_count": str(query_count),
                    "pg_total_ms": f"{pg_total:.12f}",
                    "nqo_total_ms": f"{nqo_total:.12f}",
                    "ws": f"{1.0 / fold_inverse_ws:.12f}",
                    "inverse_ws": f"{fold_inverse_ws:.12f}",
                    "cache_hits": "",
                    "checkpoint": "",
                }
            )
    return estimates


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-inventory",
        type=Path,
        default=DEFAULT_CHECKPOINT_INVENTORY,
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--container", default="pgdb_tpch_gpu")
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices(
            "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
        ),
    )
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--base-port", type=int, default=24400)
    parser.add_argument("--cache-miss", choices=("error", "execute"), default="error")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--server-action-host")
    parser.add_argument("--database-container", default="pgdb_dev_opt")
    parser.add_argument(
        "--sql-execution-lock",
        type=Path,
        default=DEFAULT_RUNTIME / "sql-execution.lock",
    )
    parser.add_argument("--sql-execution-slots", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.sql_execution_lock = args.sql_execution_lock.resolve()
    args.sql_execution_lock.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists; pass --overwrite: {args.output}")
    if args.workers < 1 or args.workers > 30:
        parser.error("--workers must be between 1 and 30")

    preceding, selected = released_checkpoints(
        args.checkpoint_inventory.resolve()
    )
    pg, best_rows = load_reference()
    best_iteration = {
        (item["workload"].upper(), item["protocol"], item["fold"]): item
        for item in selected
    }
    actual: list[dict[str, str]] = []
    for key, rows in best_rows.items():
        item = best_iteration[key]
        actual.append(
            aggregate_rows(
                dataset=key[0],
                protocol=key[1],
                fold=key[2],
                iteration=item["iteration"],
                source="released_best",
                checkpoint=item["checkpoint"],
                rows=rows,
                pg=pg,
            )
        )

    server_action_host = args.server_action_host or container_bridge_ip(args.container)
    tasks = [
        TraceTask(
            workload=item["workload"],
            protocol=item["protocol"],
            fold=item["fold"],
            iteration=item["iteration"],
            checkpoint=item["checkpoint"],
            device=args.devices[index % len(args.devices)],
            port=args.base_port + index,
            container=args.container,
            runtime_root=args.runtime_root,
            cache_miss=args.cache_miss,
            sql_execution_lock=args.sql_execution_lock,
            sql_execution_slots=args.sql_execution_slots,
            host=args.host,
            pg_port=args.pg_port,
            user=args.user,
            server_action_host=server_action_host,
            database_container=args.database_container,
        )
        for index, item in enumerate(preceding)
    ]
    completed: list[tuple[TraceTask, dict[str, Any]]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed.append((task, result))
            print(
                f"[done] {task.label} device={result['device']} "
                f"rows={result['query_count']} "
                f"misses={len(result['cache_misses'])} "
                f"executed={result['physical_executions']}",
                flush=True,
            )

    unresolved = [
        f"{task.label}:{query_id}"
        for task, result in completed
        for query_id in result["unresolved_misses"]
    ]
    for task, result in completed:
        actual.append(
            aggregate_rows(
                dataset=task.workload.upper(),
                protocol=task.protocol,
                fold=task.fold,
                iteration=task.iteration,
                source=(
                    "cache_replay+execution"
                    if result["physical_executions"]
                    else "cache_replay"
                ),
                checkpoint=task.checkpoint,
                rows=result["rows"],
                pg=pg,
            )
        )
    estimates = estimated_initial_rows(actual)
    rows = sorted(
        estimates + actual,
        key=lambda row: (
            row["dataset"],
            row["protocol"],
            int(row["iteration"]),
            row["fold"],
        ),
    )
    write_csv(args.output, rows)
    summary = {
        "tasks": len(tasks),
        "workers": min(args.workers, len(tasks)),
        "reused_best_checkpoints": len(selected),
        "estimated_init_rows": len(estimates),
        "actual_fold_checkpoints": len(actual),
        "cache_hits": sum(
            sum(row.get("cache_hit") == "True" for row in result["rows"])
            for _, result in completed
        ),
        "cache_misses": sum(
            len(result["cache_misses"]) for _, result in completed
        ),
        "physical_executions": sum(
            result["physical_executions"] for _, result in completed
        ),
        "unresolved_misses": len(unresolved),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if unresolved:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
