#!/usr/bin/env python3
"""Resolve missing GenJoin/HybridQO TPC-H prediction runtimes sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
TPCH_DIR = ROOT / "results" / "revision" / "tpch"
CORRECTED_DIR = TPCH_DIR / "corrected_splits"

sys.path.insert(0, str(SCRIPT_DIR))

from benchmarking.workloads import (  # noqa: E402
    CsvResultStore,
    connect,
    dynamic_timeout_s,
    execute_once,
    load_postgres_times,
    query_sql,
    reset_session,
)


EXECUTION_FIELDS = (
    "result_key",
    "method",
    "fold",
    "query_id",
    "run_id",
    "hint",
    "timeout_s",
    "runtime_s",
    "charged_runtime_s",
    "error",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_hint(value: str) -> str:
    return " ".join(value.split())


def collection_runtime(row: dict) -> float:
    if row["error"]:
        return float(row["charged_s"])
    return float(row["wall_s"])


def resolve_fastgres(pg_times: dict[str, float]) -> None:
    from scripts.reproduce.fastgres.measure_fastgres_labels import set_fastgres_hint

    output_dir = CORRECTED_DIR / "fastgres"
    predictions = read_csv(output_dir / "selected_experience.csv")
    missing = [row for row in predictions if row["run1_error"]]
    store = CsvResultStore(output_dir / "retest_executions.csv", EXECUTION_FIELDS)
    connection = connect("TPCH")
    cursor = connection.cursor()
    try:
        for index, row in enumerate(missing, start=1):
            result_key = "fastgres:{}:{}".format(row["fold"], row["query_id"])
            if store.get(result_key) is not None:
                print(f"FASTgres retest {index}/{len(missing)} resume {result_key}")
                continue
            reset_session(cursor)
            set_fastgres_hint(cursor, int(row["predicted_hint"]))
            timeout_s = dynamic_timeout_s(pg_times[row["query_id"]])
            result = execute_once(
                cursor, query_sql("TPCH", row["query_id"]), timeout_s=timeout_s
            )
            store.append(
                {
                    "result_key": result_key,
                    "method": "FASTgres",
                    "fold": row["fold"],
                    "query_id": row["query_id"],
                    "run_id": "",
                    "hint": row["predicted_hint"],
                    "timeout_s": timeout_s,
                    "runtime_s": "" if result.runtime_s is None else result.runtime_s,
                    "charged_runtime_s": result.charged_runtime_s,
                    "error": result.error or "",
                }
            )
            print(
                "FASTgres retest {}/{} {} {:.6f}s{}".format(
                    index,
                    len(missing),
                    result_key,
                    result.charged_runtime_s,
                    " timeout/error" if result.error else "",
                ),
                flush=True,
            )
    finally:
        cursor.close()
        connection.close()

    executions = {
        row["result_key"]: row for row in read_csv(output_dir / "retest_executions.csv")
    }
    resolved = []
    for row in predictions:
        result = dict(row)
        if row["run1_error"]:
            execution = executions[
                "fastgres:{}:{}".format(row["fold"], row["query_id"])
            ]
            result["runtime_source"] = "fresh_standard_timeout_retest"
            result["resolved_runtime_s"] = float(execution["charged_runtime_s"])
            result["runtime_error"] = execution["error"]
        else:
            result["runtime_source"] = "exact_successful_label_experience"
            result["resolved_runtime_s"] = float(row["run1_charged_s"])
            result["runtime_error"] = ""
        resolved.append(result)
    write_csv(output_dir / "resolved_predictions.csv", resolved)


def resolve_tonic(pg_times: dict[str, float]) -> None:
    from scripts.reproduce.tonic.tonic_common import (
        configure_tonic_session,
        hinted_sql,
        load_skeletons,
    )

    output_dir = CORRECTED_DIR / "tonic"
    predictions = read_csv(output_dir / "selected_experience.csv")
    missing = [row for row in predictions if row["run1_error"]]
    skeletons = load_skeletons(TPCH_DIR / "tonic" / "skeletons.json")
    store = CsvResultStore(output_dir / "retest_executions.csv", EXECUTION_FIELDS)
    connection = connect("TPCH")
    cursor = connection.cursor()
    try:
        for index, row in enumerate(missing, start=1):
            result_key = "tonic:{}:{}".format(row["fold"], row["query_id"])
            if store.get(result_key) is not None:
                print(f"TONIC retest {index}/{len(missing)} resume {result_key}")
                continue
            configure_tonic_session(cursor)
            timeout_s = dynamic_timeout_s(pg_times[row["query_id"]])
            result = execute_once(
                cursor,
                hinted_sql(
                    "TPCH",
                    skeletons[row["query_id"]],
                    row["predicted_assignment"],
                ),
                timeout_s=timeout_s,
            )
            store.append(
                {
                    "result_key": result_key,
                    "method": "TONIC",
                    "fold": row["fold"],
                    "query_id": row["query_id"],
                    "run_id": "",
                    "hint": row["predicted_assignment"],
                    "timeout_s": timeout_s,
                    "runtime_s": "" if result.runtime_s is None else result.runtime_s,
                    "charged_runtime_s": result.charged_runtime_s,
                    "error": result.error or "",
                }
            )
            print(
                "TONIC retest {}/{} {} {:.6f}s{}".format(
                    index,
                    len(missing),
                    result_key,
                    result.charged_runtime_s,
                    " timeout/error" if result.error else "",
                ),
                flush=True,
            )
    finally:
        cursor.close()
        connection.close()

    executions = {
        row["result_key"]: row for row in read_csv(output_dir / "retest_executions.csv")
    }
    resolved = []
    for row in predictions:
        result = dict(row)
        if row["run1_error"]:
            execution = executions[
                "tonic:{}:{}".format(row["fold"], row["query_id"])
            ]
            result["runtime_source"] = "fresh_standard_timeout_retest"
            result["resolved_runtime_s"] = float(execution["charged_runtime_s"])
            result["runtime_error"] = execution["error"]
        else:
            result["runtime_source"] = "exact_successful_feedback_experience"
            result["resolved_runtime_s"] = float(row["run1_charged_s"])
            result["runtime_error"] = ""
        resolved.append(result)
    write_csv(output_dir / "resolved_predictions.csv", resolved)


def resolve_genjoin(pg_times: dict[str, float]) -> None:
    from scripts.reproduce.genjoin import run_genjoin_tpch as genjoin

    output_dir = CORRECTED_DIR / "genjoin"
    predictions = read_csv(output_dir / "predictions_with_experience.csv")
    collection = read_csv(TPCH_DIR / "genjoin" / "collection.csv")
    by_query: dict[str, list[dict]] = {}
    by_hint: dict[tuple[str, str], list[dict]] = {}
    for row in collection:
        by_query.setdefault(row["query_id"], []).append(row)
        by_hint.setdefault(
            (row["query_id"], canonical_hint(row["hints"])), []
        ).append(row)

    store = CsvResultStore(output_dir / "missing_executions.csv", EXECUTION_FIELDS)
    missing = [
        row
        for row in predictions
        if row["supported"] == "True" and row["experience_match"] == "False"
    ]
    connection = connect("TPCH")
    cursor = connection.cursor()
    try:
        for index, row in enumerate(missing, start=1):
            result_key = "genjoin:{}:{}:{}".format(
                row["fold"], row["query_id"], row["run_id"]
            )
            if store.get(result_key) is not None:
                print(f"GenJoin missing {index}/{len(missing)} resume {result_key}")
                continue
            reset_session(cursor, load_pg_hint_plan=True)
            timeout_s = dynamic_timeout_s(pg_times[row["query_id"]])
            result = execute_once(
                cursor,
                genjoin.hinted_sql(
                    genjoin.clean_sql(row["query_id"]), row["predicted_hints"]
                ),
                timeout_s=timeout_s,
            )
            store.append(
                {
                    "result_key": result_key,
                    "method": "GenJoin",
                    "fold": row["fold"],
                    "query_id": row["query_id"],
                    "run_id": row["run_id"],
                    "hint": row["predicted_hints"],
                    "timeout_s": timeout_s,
                    "runtime_s": "" if result.runtime_s is None else result.runtime_s,
                    "charged_runtime_s": result.charged_runtime_s,
                    "error": result.error or "",
                }
            )
            print(
                "GenJoin missing {}/{} {} {:.6f}s{}".format(
                    index,
                    len(missing),
                    result_key,
                    result.charged_runtime_s,
                    " timeout/error" if result.error else "",
                ),
                flush=True,
            )
    finally:
        cursor.close()
        connection.close()

    executions = {
        row["result_key"]: row for row in read_csv(output_dir / "missing_executions.csv")
    }
    resolved = []
    for row in predictions:
        query_id = row["query_id"]
        result = dict(row)
        result["runtime_error"] = ""
        if row["supported"] == "False":
            result["runtime_source"] = "postgres_fallback"
            result["resolved_runtime_s"] = pg_times[query_id]
            result["matched_candidate_count"] = 0
        else:
            predicted = np.asarray(
                json.loads(row["predicted_plan_encoding"]), dtype=float
            )
            exact = [
                candidate
                for candidate in by_query[query_id]
                if np.allclose(
                    np.asarray(json.loads(candidate["plan_encoding"]), dtype=float),
                    predicted,
                    rtol=0,
                    atol=1e-7,
                )
            ]
            hint_matches = by_hint.get(
                (query_id, canonical_hint(row["predicted_hints"])), []
            )
            if exact:
                values = [collection_runtime(candidate) for candidate in exact]
                result["runtime_source"] = "exact_plan_encoding_experience"
                result["resolved_runtime_s"] = sum(values) / len(values)
                result["matched_candidate_count"] = len(values)
            elif hint_matches:
                values = [collection_runtime(candidate) for candidate in hint_matches]
                result["runtime_source"] = "exact_hint_experience_mean"
                result["resolved_runtime_s"] = sum(values) / len(values)
                result["matched_candidate_count"] = len(values)
            else:
                result_key = "genjoin:{}:{}:{}".format(
                    row["fold"], query_id, row["run_id"]
                )
                execution = executions[result_key]
                result["runtime_source"] = "fresh_missing_test_execution"
                result["resolved_runtime_s"] = float(
                    execution["charged_runtime_s"]
                )
                result["matched_candidate_count"] = 0
                result["runtime_error"] = execution["error"]
        resolved.append(result)
    write_csv(output_dir / "resolved_predictions.csv", resolved)


def execute_hybrid_hint(cursor, sql: str, timeout_s: float) -> tuple[float, str]:
    reset_session(cursor, load_pg_hint_plan=True)
    if "Leading" in sql:
        cursor.execute("SET geqo TO off")
    else:
        cursor.execute("SET geqo TO on")
        cursor.execute("SET geqo_threshold = 12")
    result = execute_once(cursor, sql, timeout_s=timeout_s)
    return result.charged_runtime_s, result.error or ""


def resolve_hybridqo(pg_times: dict[str, float]) -> None:
    output_dir = CORRECTED_DIR / "hybridqo"
    predictions = []
    for fold in ("random_a", "random_b", "random_c"):
        predictions.extend(read_csv(output_dir / f"{fold}_results.csv"))
    missing = [row for row in predictions if row["chosen_plan"] == "hint"]
    store = CsvResultStore(output_dir / "missing_executions.csv", EXECUTION_FIELDS)
    connection = connect("TPCH")
    cursor = connection.cursor()
    try:
        for index, row in enumerate(missing, start=1):
            result_key = "hybridqo:{}:{}".format(row["fold"], row["query_id"])
            if store.get(result_key) is not None:
                print(f"HybridQO missing {index}/{len(missing)} resume {result_key}")
                continue
            timeout_s = dynamic_timeout_s(pg_times[row["query_id"]])
            runtime_s, error = execute_hybrid_hint(
                cursor,
                row["hint"] + query_sql("TPCH", row["query_id"]),
                timeout_s,
            )
            store.append(
                {
                    "result_key": result_key,
                    "method": "HybridQO",
                    "fold": row["fold"],
                    "query_id": row["query_id"],
                    "run_id": "",
                    "hint": row["hint"],
                    "timeout_s": timeout_s,
                    "runtime_s": "" if error else runtime_s,
                    "charged_runtime_s": runtime_s,
                    "error": error,
                }
            )
            print(
                "HybridQO missing {}/{} {} {:.6f}s{}".format(
                    index,
                    len(missing),
                    result_key,
                    runtime_s,
                    " timeout/error" if error else "",
                ),
                flush=True,
            )
    finally:
        cursor.close()
        connection.close()

    executions = {
        row["result_key"]: row for row in read_csv(output_dir / "missing_executions.csv")
    }
    resolved = []
    for row in predictions:
        result = dict(row)
        if row["chosen_plan"] == "hint":
            execution = executions[
                "hybridqo:{}:{}".format(row["fold"], row["query_id"])
            ]
            result["runtime_source"] = "fresh_missing_test_execution"
            result["resolved_runtime_s"] = float(execution["charged_runtime_s"])
            result["runtime_error"] = execution["error"]
        else:
            result["runtime_source"] = (
                "postgres_fallback"
                if row["chosen_plan"] == "PG_fallback"
                else "postgres_selected"
            )
            result["resolved_runtime_s"] = pg_times[row["query_id"]]
            result["runtime_error"] = ""
        resolved.append(result)
    write_csv(output_dir / "resolved_predictions.csv", resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("fastgres", "tonic", "genjoin", "hybridqo", "all"),
        default="all",
    )
    args = parser.parse_args()
    pg_times = load_postgres_times(
        TPCH_DIR / "postgres.csv", time_column="run1_charged_s"
    )
    if args.method in ("fastgres", "all"):
        resolve_fastgres(pg_times)
    if args.method in ("tonic", "all"):
        resolve_tonic(pg_times)
    if args.method in ("genjoin", "all"):
        resolve_genjoin(pg_times)
    if args.method in ("hybridqo", "all"):
        resolve_hybridqo(pg_times)


if __name__ == "__main__":
    main()
