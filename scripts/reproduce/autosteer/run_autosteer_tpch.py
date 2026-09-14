#!/usr/bin/env python3
"""Collect, train, and evaluate AutoSteer on the revision TPC-H splits.

The released AutoSteer driver hard-codes JOB/STACK paths, opens concurrent
EXPLAIN connections, and derives random splits internally. This runner keeps
AutoSteer's PostgreSQL query-span exploration, depth-3 dynamic-programming
hint search, and Bao/TCNN predictor while applying the revision harness:

* all PostgreSQL work is sequential;
* candidate configurations are measured once;
* timeouts are min(360 seconds, 5 * PostgreSQL);
* the repository's exact random_a/random_b/random_c folds are used;
* only training-query runtimes are passed to each fold's model; and
* each selected test configuration is executed once for reported metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import psycopg2

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    MEASUREMENT_COLUMNS,
    CsvResultStore,
    connect,
    dynamic_timeout_s,
    execute_once,
    load_postgres_times,
    one_result_to_row,
    query_path,
    query_sql,
    split_folds,
    sql_sha256,
    workload_query_ids,
)


ROOT = Path(__file__).resolve().parents[3]
AUTOSTEER_ROOT = (
    ROOT
    / "thrid_party"
    / "genjoin"
    / "OtherMethods_Repository"
    / "autosteer"
)
KNOB_PATH = AUTOSTEER_ROOT / "knobs" / "postgres.txt"
MAX_DP_DEPTH = 3
WORKLOAD = "TPCH"

COLLECTION_FIELDS = [
    "query_id",
    "config",
    "disabled_count",
    "runtime_s",
    "charged_runtime_s",
    "timeout_s",
    "error",
    "plan_hash",
    "plan_json",
]

RESULT_FIELDS = [
    "result_key",
    "workload",
    "protocol",
    "fold",
    "query_id",
    "sql_path",
    "sql_sha256",
    "disabled_rules",
    "inference_s",
    "pg_measured_s",
] + MEASUREMENT_COLUMNS


def load_knobs() -> List[str]:
    return [line.strip() for line in KNOB_PATH.read_text().splitlines() if line.strip()]


def canonical_config(config: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(config)))


def config_text(config: Iterable[str]) -> str:
    return ",".join(canonical_config(config)) or "None"


def set_optimizer_config(cursor, knobs: Sequence[str], disabled: Iterable[str]) -> None:
    disabled = set(disabled)
    cursor.execute("RESET ALL")
    cursor.execute("SET search_path TO public")
    for knob in knobs:
        cursor.execute("SET {} TO {}".format(knob, "off" if knob in disabled else "on"))


def explain_plan(cursor, sql: str, knobs: Sequence[str], disabled: Iterable[str]) -> dict:
    set_optimizer_config(cursor, knobs, disabled)
    cursor.execute("EXPLAIN (FORMAT JSON) " + sql)
    return cursor.fetchone()[0][0]["Plan"]


def stable_plan_hash(plan: dict) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_plan(plan: dict) -> dict:
    """Make TPC-H auxiliary plans consumable by AutoSteer's JOB-era encoder."""
    node = copy.deepcopy(plan)
    if node.get("Node Type") == "CTE Scan":
        node["Node Type"] = "Seq Scan"
        node["Relation Name"] = node.get("CTE Name", node.get("Alias", "cte"))
    if "Plans" in node:
        children = [
            normalize_plan(child)
            for child in node["Plans"]
            if child.get("Parent Relationship") not in ("InitPlan", "SubPlan")
        ]
        if children:
            node["Plans"] = children
        else:
            node.pop("Plans", None)
    return node


def find_query_span(
    cursor,
    sql: str,
    knobs: Sequence[str],
) -> Tuple[List[str], Dict[str, Set[str]], dict]:
    """Mirror AutoSteer's iterative query-span approximation sequentially."""
    default_plan = explain_plan(cursor, sql, knobs, ())
    default_hash = stable_plan_hash(default_plan)
    effective: List[str] = []
    dependencies: Dict[str, Set[str]] = {}
    remaining: List[str] = []
    queue: List[Tuple[str, Tuple[str, ...], dict]] = []

    for knob in knobs:
        try:
            plan = explain_plan(cursor, sql, knobs, (knob,))
        except Exception:
            cursor.connection.rollback()
            continue
        if stable_plan_hash(plan) == default_hash:
            remaining.append(knob)
            continue
        effective.append(knob)
        queue.append((knob, (knob,), plan))

    while queue:
        _, base_config, base_plan = queue.pop(0)
        base_hash = stable_plan_hash(base_plan)
        for knob in list(remaining):
            candidate = canonical_config((*base_config, knob))
            try:
                plan = explain_plan(cursor, sql, knobs, candidate)
            except Exception:
                cursor.connection.rollback()
                continue
            if stable_plan_hash(plan) == base_hash:
                continue
            dependencies[knob] = set(base_config)
            effective.append(knob)
            queue.append((knob, candidate, plan))
            remaining.remove(knob)

    return sorted(set(effective)), dependencies, default_plan


def dependencies_satisfied(
    config: Iterable[str],
    dependencies: Dict[str, Set[str]],
) -> bool:
    config = set(config)
    return all(dependencies.get(knob, set()).issubset(config) for knob in config)


@dataclass
class CandidateResult:
    config: Tuple[str, ...]
    runtime_s: Optional[float]
    charged_runtime_s: float
    timeout_s: float
    error: Optional[str]
    plan_hash: str
    plan: dict


def execute_candidate(
    cursor,
    sql: str,
    knobs: Sequence[str],
    config: Iterable[str],
    timeout_s: float,
    plan: dict,
) -> CandidateResult:
    config = canonical_config(config)
    set_optimizer_config(cursor, knobs, config)
    timeout_ms = max(1, int(math.ceil(timeout_s * 1000.0)))
    cursor.execute("SET statement_timeout = {}".format(timeout_ms))
    started = time.perf_counter()
    try:
        cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        cursor.fetchone()
        runtime_s = time.perf_counter() - started
        return CandidateResult(
            config=config,
            runtime_s=runtime_s,
            charged_runtime_s=runtime_s,
            timeout_s=timeout_s,
            error=None,
            plan_hash=stable_plan_hash(plan),
            plan=plan,
        )
    except Exception as exc:
        cursor.connection.rollback()
        return CandidateResult(
            config=config,
            runtime_s=None,
            charged_runtime_s=timeout_s,
            timeout_s=timeout_s,
            error=str(exc).replace("\n", " ")[:1000],
            plan_hash=stable_plan_hash(plan),
            plan=plan,
        )


def candidate_to_row(query_id: str, result: CandidateResult) -> Dict[str, object]:
    return {
        "query_id": query_id,
        "config": config_text(result.config),
        "disabled_count": len(result.config),
        "runtime_s": "" if result.runtime_s is None else result.runtime_s,
        "charged_runtime_s": result.charged_runtime_s,
        "timeout_s": result.timeout_s,
        "error": result.error or "",
        "plan_hash": result.plan_hash,
        "plan_json": json.dumps(result.plan, sort_keys=True),
    }


def load_collection(path: Path) -> Dict[str, Dict[Tuple[str, ...], CandidateResult]]:
    by_query: Dict[str, Dict[Tuple[str, ...], CandidateResult]] = {}
    if not path.is_file():
        return by_query
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            config = () if row["config"] == "None" else tuple(row["config"].split(","))
            by_query.setdefault(row["query_id"], {})[config] = CandidateResult(
                config=config,
                runtime_s=float(row["runtime_s"]) if row["runtime_s"] else None,
                charged_runtime_s=float(row["charged_runtime_s"]),
                timeout_s=float(row["timeout_s"]),
                error=row["error"] or None,
                plan_hash=row["plan_hash"],
                plan=json.loads(row["plan_json"]),
            )
    return by_query


def append_collection(path: Path, query_id: str, result: CandidateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLLECTION_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(candidate_to_row(query_id, result))
        handle.flush()


def combine_stage(
    singletons: Sequence[Tuple[str, ...]],
    previous: Sequence[Tuple[str, ...]],
    blacklist: Sequence[frozenset],
    dependencies: Dict[str, Set[str]],
) -> List[Tuple[str, ...]]:
    result = set()
    for singleton in singletons:
        knob = singleton[0]
        for config in previous:
            if knob in config:
                continue
            candidate = canonical_config((*config, knob))
            if any(bad.issubset(candidate) for bad in blacklist):
                continue
            if dependencies_satisfied(candidate, dependencies):
                result.add(candidate)
    return sorted(result)


def collect_query(
    cursor,
    query_id: str,
    sql: str,
    pg_time_s: float,
    knobs: Sequence[str],
    collection_path: Path,
    existing: Dict[Tuple[str, ...], CandidateResult],
) -> Tuple[Dict[Tuple[str, ...], CandidateResult], dict]:
    timeout_s = dynamic_timeout_s(pg_time_s)
    effective, dependencies, _ = find_query_span(cursor, sql, knobs)
    blacklist: List[frozenset] = []
    candidate_hashes: Dict[str, Tuple[str, ...]] = {
        result.plan_hash: config for config, result in existing.items()
    }

    def measure(config: Tuple[str, ...]) -> Optional[CandidateResult]:
        if config in existing:
            return existing[config]
        plan = explain_plan(cursor, sql, knobs, config)
        plan_hash = stable_plan_hash(plan)
        if plan_hash in candidate_hashes:
            return None
        result = execute_candidate(cursor, sql, knobs, config, timeout_s, plan)
        existing[config] = result
        candidate_hashes[result.plan_hash] = config
        append_collection(collection_path, query_id, result)
        print(
            "[{} AutoSteer collect {}] config={} {:.6f}s{}".format(
                WORKLOAD,
                query_id,
                config_text(config),
                result.charged_runtime_s,
                " timeout/error" if result.error else "",
            ),
            flush=True,
        )
        return result

    baseline = measure(())
    if baseline is None:
        baseline = existing[()]
    baseline_time = baseline.charged_runtime_s

    singletons = [
        (knob,)
        for knob in effective
        if dependencies_satisfied((knob,), dependencies)
    ]
    stage_results: Dict[int, List[Tuple[str, ...]]] = {1: []}
    for config in singletons:
        result = measure(config)
        if result is None:
            continue
        stage_results[1].append(config)
        if result.charged_runtime_s > baseline_time:
            blacklist.append(frozenset(config))

    good_previous = [
        config
        for config in stage_results[1]
        if existing[config].charged_runtime_s < baseline_time
    ]
    # Dependency-only knobs cannot be measured alone, but they can become
    # valid combination seeds once their prerequisites are present.
    combination_singletons = sorted(
        set(good_previous).union((knob,) for knob in dependencies)
    )
    for depth in range(2, MAX_DP_DEPTH + 1):
        stage = combine_stage(
            combination_singletons,
            good_previous,
            blacklist,
            dependencies,
        )
        measured_stage = []
        for config in stage:
            result = measure(config)
            if result is None:
                continue
            measured_stage.append(config)
            if result.charged_runtime_s > baseline_time:
                blacklist.append(frozenset(config))
        stage_results[depth] = measured_stage
        good_previous = [
            config
            for config in measured_stage
            if existing[config].charged_runtime_s < baseline_time
        ]
        if not good_previous:
            break

    audit = {
        "query_id": query_id,
        "effective_knobs": effective,
        "dependencies": {
            knob: sorted(values) for knob, values in sorted(dependencies.items())
        },
        "measured_configs": len(existing),
        "stage_counts": {
            str(depth): len(configs) for depth, configs in stage_results.items()
        },
        "timeout_s": timeout_s,
    }
    return existing, audit


def collect_all(args, output_dir: Path, pg_times: Dict[str, float]) -> None:
    collection_path = output_dir / "collection.csv"
    existing_all = load_collection(collection_path)
    starting_executions = sum(len(values) for values in existing_all.values())
    audit_path = output_dir / "collection_audit.json"
    previous_wall_s = 0.0
    previous_sessions = 0
    if audit_path.is_file():
        previous_audit = json.loads(audit_path.read_text())
        previous_timing = previous_audit.get("timing", {})
        previous_wall_s = float(previous_timing.get("wall_s", 0.0))
        previous_sessions = int(previous_timing.get("collection_sessions", 1))
    audits = {}
    connection = connect(
        WORKLOAD,
        host=args.host,
        port=args.port,
        user=args.user,
    )
    cursor = connection.cursor()
    knobs = load_knobs()
    started = time.perf_counter()
    try:
        query_ids = workload_query_ids(WORKLOAD)
        for index, query_id in enumerate(query_ids, 1):
            print(
                "[{} AutoSteer {}/{}] query-span and DP for Q{}".format(
                    WORKLOAD, index, len(query_ids), query_id
                ),
                flush=True,
            )
            existing, audit = collect_query(
                cursor,
                query_id,
                query_sql(WORKLOAD, query_id),
                pg_times[query_id],
                knobs,
                collection_path,
                existing_all.setdefault(query_id, {}),
            )
            existing_all[query_id] = existing
            audits[query_id] = audit
    finally:
        cursor.close()
        connection.close()
    wall_s_this_run = time.perf_counter() - started
    candidate_executions = sum(len(values) for values in existing_all.values())
    timing = {
        "wall_s": previous_wall_s + wall_s_this_run,
        "wall_s_this_run": wall_s_this_run,
        "collection_sessions": previous_sessions + 1,
        "queries": len(audits),
        "candidate_executions": candidate_executions,
        "candidate_executions_this_run": candidate_executions - starting_executions,
        "sequential_sql_execution": True,
        "candidate_repetitions": 1,
    }
    audit_path.write_text(
        json.dumps({"timing": timing, "queries": audits}, indent=2, sort_keys=True)
        + "\n"
    )


def import_autosteer_model():
    sys.path.insert(0, str(AUTOSTEER_ROOT))
    from inference.model import BaoRegressionModel
    from inference.preprocessing.preprocess_postgres_plans import (
        PostgresPlanPreprocessor,
    )

    return BaoRegressionModel, PostgresPlanPreprocessor


def prepare_model_data(
    records: Dict[str, Dict[Tuple[str, ...], CandidateResult]],
    query_ids: Sequence[str],
) -> Tuple[List[dict], List[float]]:
    plans = []
    runtimes = []
    for query_id in query_ids:
        for result in records[query_id].values():
            plans.append(normalize_plan(result.plan))
            runtimes.append(result.charged_runtime_s * 1_000_000.0)
    return plans, runtimes


def train_fold(
    fold: str,
    spec: dict,
    records: Dict[str, Dict[Tuple[str, ...], CandidateResult]],
    model_dir: Path,
    seed: int,
):
    import torch

    torch.manual_seed(seed)
    BaoRegressionModel, PostgresPlanPreprocessor = import_autosteer_model()
    x_train, y_train = prepare_model_data(records, spec["train"])
    # AutoSteer's validation loss is diagnostic only. Reuse the training fold so
    # no test-query runtime is exposed and no training candidate is discarded.
    x_fit = x_train
    y_fit = y_train
    x_validation = x_train
    y_validation = y_train

    model = BaoRegressionModel(PostgresPlanPreprocessor())
    started = time.perf_counter()
    model.fit(x_fit, y_fit, x_validation, y_validation)
    training_s = time.perf_counter() - started
    fold_model_dir = model_dir / fold
    model.save(str(fold_model_dir))
    return model, {
        "fold": fold,
        "training_query_ids": list(spec["train"]),
        "test_query_ids": list(spec["test"]),
        "training_candidates": len(x_train),
        "fit_candidates": len(x_fit),
        "validation_candidates": len(x_validation),
        "training_s": training_s,
        "test_runtimes_used_for_training": False,
        "online_test_updates": False,
    }


def choose_test_configs(model, spec: dict, records) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, float]]:
    predictions = {}
    inference_times = {}
    for query_id in spec["test"]:
        configs = sorted(records[query_id])
        plans = [normalize_plan(records[query_id][config].plan) for config in configs]
        started = time.perf_counter()
        estimates = model.predict(plans).reshape(-1)
        inference_times[query_id] = time.perf_counter() - started
        predictions[query_id] = configs[int(np.argmin(estimates))]
    return predictions, inference_times


def execute_predictions(
    args,
    output_dir: Path,
    pg_times: Dict[str, float],
    predictions_by_fold,
    inference_by_fold,
) -> None:
    result_path = output_dir / "random_results.csv"
    store = CsvResultStore(result_path, RESULT_FIELDS)
    connection = connect(
        WORKLOAD,
        host=args.host,
        port=args.port,
        user=args.user,
    )
    cursor = connection.cursor()
    knobs = load_knobs()
    try:
        total = sum(len(values) for values in predictions_by_fold.values())
        index = 0
        for fold, predictions in predictions_by_fold.items():
            for query_id, config in predictions.items():
                index += 1
                sql = query_sql(WORKLOAD, query_id)
                digest = sql_sha256(sql)
                timeout_s = dynamic_timeout_s(pg_times[query_id])
                result_key = "TPCH:autosteer-test:random:{}:{}".format(
                    fold, query_id
                )
                previous = store.get(result_key)
                if (
                    previous
                    and previous["sql_sha256"] == digest
                    and previous["disabled_rules"] == config_text(config)
                    and abs(float(previous["timeout_s"]) - timeout_s) < 1e-9
                ):
                    print(
                        "[TPCH AutoSteer test {}/{}] {} Q{} resume {}s".format(
                            index, total, fold, query_id, previous["measured_s"]
                        ),
                        flush=True,
                    )
                    continue

                set_optimizer_config(cursor, knobs, config)
                result = execute_once(cursor, sql, timeout_s=timeout_s)
                row = {
                    "result_key": result_key,
                    "workload": WORKLOAD,
                    "protocol": "random",
                    "fold": fold,
                    "query_id": query_id,
                    "sql_path": str(query_path(WORKLOAD, query_id).relative_to(ROOT)),
                    "sql_sha256": digest,
                    "disabled_rules": config_text(config),
                    "inference_s": inference_by_fold[fold][query_id],
                    "pg_measured_s": pg_times[query_id],
                }
                row.update(one_result_to_row(result))
                store.append(row)
                print(
                    "[TPCH AutoSteer test {}/{}] {} Q{} config={} {:.6f}s{}".format(
                        index,
                        total,
                        fold,
                        query_id,
                        config_text(config),
                        result.charged_runtime_s,
                        " timeout/error" if result.error else "",
                    ),
                    flush=True,
                )
    finally:
        cursor.close()
        connection.close()


def summarize(output_dir: Path, pg_times: Dict[str, float]) -> dict:
    with (output_dir / "random_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["fold"], row["query_id"]): row for row in rows}
    expected = []
    for fold, spec in split_folds(WORKLOAD, "random").items():
        expected.extend((fold, query_id) for query_id in spec["test"])
    if set(by_key) != set(expected):
        raise RuntimeError("AutoSteer TPC-H result coverage is incomplete")

    seen = set()
    pg_values = []
    method_values = []
    errors = 0
    for key in expected:
        query_id = key[1]
        row = by_key[key]
        errors += bool(row["run1_error"])
        if query_id in seen:
            continue
        seen.add(query_id)
        pg_values.append(pg_times[query_id])
        method_values.append(float(row["run1_charged_s"]))
    speedups = [
        pg_time / method_time
        for pg_time, method_time in zip(pg_values, method_values)
    ]
    summary = {
        "method": "AutoSteer",
        "workload": WORKLOAD,
        "protocol": "random",
        "test_records": len(expected),
        "unique_queries": len(seen),
        "pg_total_s": sum(pg_values),
        "method_total_s": sum(method_values),
        "WS": sum(pg_values) / sum(method_values),
        "GS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "Imp_pct": 100.0
        * sum(
            method_time < pg_time
            for pg_time, method_time in zip(pg_values, method_values)
        )
        / len(method_values),
        "timeouts_or_errors": errors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def train_and_test(args, output_dir: Path, pg_times: Dict[str, float]) -> dict:
    records = load_collection(output_dir / "collection.csv")
    missing = [
        query_id
        for query_id in workload_query_ids(WORKLOAD)
        if not records.get(query_id)
    ]
    if missing:
        raise RuntimeError("AutoSteer collection missing {}".format(missing))

    model_dir = output_dir / "models"
    predictions_by_fold = {}
    inference_by_fold = {}
    audit_by_fold = {}
    for index, (fold, spec) in enumerate(
        split_folds(WORKLOAD, "random").items()
    ):
        model, audit = train_fold(
            fold,
            spec,
            records,
            model_dir,
            args.seed + index,
        )
        predictions, inference_times = choose_test_configs(model, spec, records)
        predictions_by_fold[fold] = predictions
        inference_by_fold[fold] = inference_times
        audit_by_fold[fold] = audit
        print(
            "[TPCH AutoSteer train] {} candidates={} time={:.3f}s".format(
                fold, audit["training_candidates"], audit["training_s"]
            ),
            flush=True,
        )

    (output_dir / "random_predictions.json").write_text(
        json.dumps(
            {
                "predictions": {
                    fold: {
                        query_id: config_text(config)
                        for query_id, config in values.items()
                    }
                    for fold, values in predictions_by_fold.items()
                },
                "inference_s": inference_by_fold,
                "fold_audit": audit_by_fold,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    execute_predictions(
        args,
        output_dir,
        pg_times,
        predictions_by_fold,
        inference_by_fold,
    )
    return summarize(output_dir, pg_times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("collect", "train-test", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_root.resolve() / "tpch" / "autosteer"
    output_dir.mkdir(parents=True, exist_ok=True)
    pg_times = load_postgres_times(
        args.output_root.resolve() / "tpch" / "postgres.csv",
        time_column="run1_charged_s",
    )
    if args.phase in ("collect", "all"):
        collect_all(args, output_dir, pg_times)
    if args.phase in ("train-test", "all"):
        summary = train_and_test(args, output_dir, pg_times)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
