#!/usr/bin/env python3
"""Collect AutoSteer candidates sequentially, three times, without a timeout."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from scripts.reproduce.autosteer import run_autosteer_tpch as autosteer
from benchmarking.workloads import (
    connect,
    query_sql,
    workload_query_ids,
)


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_FIELDS = [
    "query_id",
    "config",
    "disabled_count",
    "plan_hash",
    "plan_json",
]
EXECUTION_FIELDS = [
    "query_id",
    "config",
    "repetition",
    "runtime_s",
    "error",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, fields: Sequence[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def load_candidates(path: Path) -> Dict[str, Dict[Tuple[str, ...], dict]]:
    result: Dict[str, Dict[Tuple[str, ...], dict]] = {}
    for row in read_rows(path):
        config = () if row["config"] == "None" else tuple(row["config"].split(","))
        result.setdefault(row["query_id"], {})[config] = {
            "plan_hash": row["plan_hash"],
            "plan": json.loads(row["plan_json"]),
        }
    return result


def load_executions(
    path: Path,
) -> Dict[Tuple[str, Tuple[str, ...]], Dict[int, float]]:
    result: Dict[Tuple[str, Tuple[str, ...]], Dict[int, float]] = {}
    for row in read_rows(path):
        config = () if row["config"] == "None" else tuple(row["config"].split(","))
        if row["error"]:
            raise RuntimeError(
                "Cannot resume failed execution for {} config={}: {}".format(
                    row["query_id"], row["config"], row["error"]
                )
            )
        result.setdefault((row["query_id"], config), {})[
            int(row["repetition"])
        ] = float(row["runtime_s"])
    return result


def execute_missing_repetitions(
    cursor,
    sql: str,
    knobs: Sequence[str],
    query_id: str,
    config: Tuple[str, ...],
    repetitions: int,
    execution_path: Path,
    completed: Dict[int, float],
) -> List[float]:
    autosteer.set_optimizer_config(cursor, knobs, config)
    cursor.execute("SET statement_timeout = 0")
    config_name = autosteer.config_text(config)
    for repetition in range(1, repetitions + 1):
        if repetition in completed:
            continue
        started = time.perf_counter()
        try:
            cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
            cursor.fetchone()
            runtime_s = time.perf_counter() - started
        except Exception as exc:
            cursor.connection.rollback()
            append_row(
                execution_path,
                EXECUTION_FIELDS,
                {
                    "query_id": query_id,
                    "config": config_name,
                    "repetition": repetition,
                    "runtime_s": "",
                    "error": str(exc).replace("\n", " ")[:1000],
                },
            )
            raise
        completed[repetition] = runtime_s
        append_row(
            execution_path,
            EXECUTION_FIELDS,
            {
                "query_id": query_id,
                "config": config_name,
                "repetition": repetition,
                "runtime_s": runtime_s,
                "error": "",
            },
        )
        print(
            "[{} AutoSteer collect {}] config={} run={}/{} {:.6f}s".format(
                autosteer.WORKLOAD,
                query_id,
                config_name,
                repetition,
                repetitions,
                runtime_s,
            ),
            flush=True,
        )
    return [completed[index] for index in range(1, repetitions + 1)]


def collect_query(
    cursor,
    query_id: str,
    sql: str,
    knobs: Sequence[str],
    repetitions: int,
    candidate_path: Path,
    execution_path: Path,
    candidates: Dict[Tuple[str, ...], dict],
    executions: Dict[Tuple[str, Tuple[str, ...]], Dict[int, float]],
) -> dict:
    effective, dependencies, _ = autosteer.find_query_span(cursor, sql, knobs)
    plan_hashes = {
        record["plan_hash"]: config for config, record in candidates.items()
    }
    blacklist: List[frozenset] = []

    def measure(config: Iterable[str]) -> Optional[float]:
        config = autosteer.canonical_config(config)
        record = candidates.get(config)
        if record is None:
            plan = autosteer.explain_plan(cursor, sql, knobs, config)
            plan_hash = autosteer.stable_plan_hash(plan)
            if plan_hash in plan_hashes:
                return None
            record = {"plan_hash": plan_hash, "plan": plan}
            candidates[config] = record
            plan_hashes[plan_hash] = config
            append_row(
                candidate_path,
                CANDIDATE_FIELDS,
                {
                    "query_id": query_id,
                    "config": autosteer.config_text(config),
                    "disabled_count": len(config),
                    "plan_hash": plan_hash,
                    "plan_json": json.dumps(plan, sort_keys=True),
                },
            )
        completed = executions.setdefault((query_id, config), {})
        runtimes = execute_missing_repetitions(
            cursor,
            sql,
            knobs,
            query_id,
            config,
            repetitions,
            execution_path,
            completed,
        )
        return runtimes[-1]

    baseline_time = measure(())
    if baseline_time is None:
        raise RuntimeError("Default plan unexpectedly duplicated for {}".format(query_id))

    singletons = [
        (knob,)
        for knob in effective
        if autosteer.dependencies_satisfied((knob,), dependencies)
    ]
    stage_results: Dict[int, List[Tuple[str, ...]]] = {1: []}
    labels: Dict[Tuple[str, ...], float] = {(): baseline_time}
    for config in singletons:
        runtime_s = measure(config)
        if runtime_s is None:
            continue
        labels[config] = runtime_s
        stage_results[1].append(config)
        if runtime_s > baseline_time:
            blacklist.append(frozenset(config))

    good_previous = [
        config for config in stage_results[1] if labels[config] < baseline_time
    ]
    combination_singletons = sorted(
        set(good_previous).union((knob,) for knob in dependencies)
    )
    for depth in range(2, autosteer.MAX_DP_DEPTH + 1):
        stage = autosteer.combine_stage(
            combination_singletons,
            good_previous,
            blacklist,
            dependencies,
        )
        measured_stage = []
        for config in stage:
            runtime_s = measure(config)
            if runtime_s is None:
                continue
            labels[config] = runtime_s
            measured_stage.append(config)
            if runtime_s > baseline_time:
                blacklist.append(frozenset(config))
        stage_results[depth] = measured_stage
        good_previous = [
            config for config in measured_stage if labels[config] < baseline_time
        ]
        if not good_previous:
            break

    return {
        "query_id": query_id,
        "effective_knobs": effective,
        "dependencies": {
            knob: sorted(values) for knob, values in sorted(dependencies.items())
        },
        "candidate_configs": len(candidates),
        "stage_counts": {
            str(depth): len(configs) for depth, configs in stage_results.items()
        },
        "label_repetition": repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("JOB", "STACK", "TPCH"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    autosteer.WORKLOAD = args.workload
    autosteer.AUTOSTEER_ROOT = (
        ROOT
        / "thrid_party"
        / "genjoin"
        / "OtherMethods_Repository"
        / "autosteer"
    )
    autosteer.KNOB_PATH = autosteer.AUTOSTEER_ROOT / "knobs" / "postgres.txt"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidates.csv"
    execution_path = output_dir / "executions.csv"
    candidates = load_candidates(candidate_path)
    executions = load_executions(execution_path)
    starting_attempts = sum(len(values) for values in executions.values())

    connection = connect(
        args.workload,
        host=args.host,
        port=args.port,
        user=args.user,
    )
    cursor = connection.cursor()
    knobs = autosteer.load_knobs()
    audits = {}
    started = time.perf_counter()
    try:
        query_ids = workload_query_ids(args.workload)
        for index, query_id in enumerate(query_ids, 1):
            print(
                "[{} AutoSteer {}/{}] query-span and DP for Q{}".format(
                    args.workload, index, len(query_ids), query_id
                ),
                flush=True,
            )
            query_candidates = candidates.setdefault(query_id, {})
            audits[query_id] = collect_query(
                cursor,
                query_id,
                query_sql(args.workload, query_id),
                knobs,
                args.repetitions,
                candidate_path,
                execution_path,
                query_candidates,
                executions,
            )
    finally:
        cursor.close()
        connection.close()

    wall_s = time.perf_counter() - started
    execution_attempts = sum(len(values) for values in executions.values())
    audit = {
        "timing": {
            "wall_s": wall_s,
            "queries": len(audits),
            "candidate_configs": sum(len(values) for values in candidates.values()),
            "execution_attempts": execution_attempts,
            "execution_attempts_this_run": execution_attempts - starting_attempts,
            "candidate_repetitions": args.repetitions,
            "label_repetition": args.repetitions,
            "statement_timeout": 0,
            "sequential_sql_execution": True,
        },
        "queries": audits,
    }
    (output_dir / "collection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
