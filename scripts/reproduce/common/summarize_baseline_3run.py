#!/usr/bin/env python3
"""Validate and summarize FASTgres/TONIC reproduction measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    SPLIT_PROTOCOLS,
    dynamic_timeout_s,
    load_postgres_times,
    split_folds,
    workload_query_ids,
)
from scripts.reproduce.tonic.tonic_common import load_skeletons


METHODS = {
    "FASTgres": "fastgres",
    "TONIC": "tonic",
}

SUMMARY_FIELDS = [
    "method",
    "workload",
    "protocol",
    "status",
    "test_records",
    "unique_queries",
    "pg_total_s",
    "method_total_s",
    "WS",
    "GS",
    "Imp_pct",
    "timeouts_or_errors",
]


def read_rows(path):
    if not path.is_file():
        raise RuntimeError("missing result file {}".format(path))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_three_runs(row, context):
    for run_idx in range(1, 4):
        runtime = row["run{}_s".format(run_idx)]
        error = row["run{}_error".format(run_idx)]
        charged = row["run{}_charged_s".format(run_idx)]
        if bool(runtime) == bool(error):
            raise RuntimeError(
                "{} run {} must contain exactly one of runtime/error".format(
                    context,
                    run_idx,
                )
            )
        if not charged:
            raise RuntimeError(
                "{} run {} is missing charged runtime".format(
                    context,
                    run_idx,
                )
            )
    if float(row["measured_s"]) != float(row["run3_charged_s"]):
        raise RuntimeError("{} does not use the third run".format(context))


def validate_fastgres_label_run(row, context):
    runtime = row["run1_s"]
    error = row["run1_error"]
    charged = row["run1_charged_s"]
    if bool(runtime) == bool(error):
        raise RuntimeError(
            "{} first run must contain exactly one of runtime/error".format(context)
        )
    if not charged:
        raise RuntimeError("{} first run is missing charged runtime".format(context))

    has_later_runs = any(
        row["run{}_s".format(idx)]
        or row["run{}_charged_s".format(idx)]
        or row["run{}_error".format(idx)]
        for idx in (2, 3)
    )
    if has_later_runs:
        validate_three_runs(row, context)
    elif float(row["measured_s"]) != float(charged):
        raise RuntimeError("{} does not use its single run".format(context))


def validate_postgres(output_root, workload):
    path = output_root / workload.lower() / "postgres.csv"
    rows = read_rows(path)
    expected = set(workload_query_ids(workload))
    found = {row["query_id"] for row in rows}
    if found != expected or len(rows) != len(expected):
        raise RuntimeError(
            "{} PostgreSQL coverage mismatch: expected {}, got {}".format(
                workload,
                len(expected),
                len(rows),
            )
        )
    for row in rows:
        context = "{} PostgreSQL {}".format(workload, row["query_id"])
        validate_three_runs(row, context)
        if row["measured_error"]:
            raise RuntimeError("{} failed".format(context))
        if row["timeout_s"]:
            raise RuntimeError("{} unexpectedly has a timeout".format(context))


def validate_dynamic_timeout(row, pg_times, context):
    expected = dynamic_timeout_s(pg_times[row["query_id"]])
    observed = float(row["timeout_s"])
    tolerance = max(1e-9, abs(expected) * 1e-9)
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            "{} timeout is {}, expected {}".format(context, observed, expected)
        )


def validate_fastgres_labels(output_root, workload, pg_times):
    path = output_root / workload.lower() / "fastgres" / "labels.csv"
    rows = read_rows(path)
    by_query = defaultdict(dict)
    for row in rows:
        context = "{} FASTgres label {} hint {}".format(
            workload,
            row["query_id"],
            row["hint"],
        )
        validate_fastgres_label_run(row, context)
        hint = int(row["hint"])
        if hint in by_query[row["query_id"]]:
            raise RuntimeError("{} is duplicated".format(context))
        by_query[row["query_id"]][hint] = row
    expected_hints = set(range(64))
    incomplete = [
        query_id
        for query_id in workload_query_ids(workload)
        if set(by_query[query_id]) != expected_hints
    ]
    if incomplete:
        raise RuntimeError(
            "{} FASTgres labels are incomplete for {}".format(
                workload,
                incomplete[:5],
            )
        )

    for query_id in workload_query_ids(workload):
        best_time = pg_times[query_id]
        for hint in range(63, -1, -1):
            row = by_query[query_id][hint]
            context = "{} FASTgres label {} hint {}".format(
                workload,
                query_id,
                hint,
            )
            if row["run2_charged_s"]:
                validate_dynamic_timeout(row, pg_times, context)
            else:
                observed = float(row["timeout_s"])
                tolerance = max(1e-9, abs(best_time) * 1e-9)
                if abs(observed - best_time) > tolerance:
                    raise RuntimeError(
                        "{} adaptive timeout is {}, expected {}".format(
                            context,
                            observed,
                            best_time,
                        )
                    )
            runtime = float(row["run1_charged_s"])
            if not row["run1_error"] and runtime < best_time:
                best_time = runtime


def validate_tonic_feedback(output_root, workload, pg_times):
    tonic_dir = output_root / workload.lower() / "tonic"
    rows = read_rows(tonic_dir / "feedback.csv")
    skeletons = load_skeletons(tonic_dir / "skeletons.json")
    by_query = defaultdict(dict)
    for row in rows:
        context = "{} TONIC feedback {} {}".format(
            workload,
            row["query_id"],
            row["assignment"] or "NOJOIN",
        )
        validate_three_runs(row, context)
        validate_dynamic_timeout(row, pg_times, context)
        by_query[row["query_id"]][row["assignment"]] = float(row["measured_s"])

    incomplete = []
    for query_id in workload_query_ids(workload):
        runtimes = by_query[query_id]
        if not runtimes:
            incomplete.append(query_id)
            continue
        best = min(
            runtimes,
            key=lambda assignment: (runtimes[assignment], assignment),
        )
        required = set()
        for idx in range(skeletons[query_id].decision_count):
            required.add(best[:idx] + "H" + best[idx + 1 :])
            required.add(best[:idx] + "N" + best[idx + 1 :])
        if not required.issubset(runtimes):
            incomplete.append(query_id)
    if incomplete:
        raise RuntimeError(
            "{} TONIC feedback is incomplete for {}".format(
                workload,
                incomplete[:5],
            )
        )


def validate_fold_audit(path, workload, protocol):
    content = json.loads(path.read_text())
    audits = content["fold_audit"]
    for fold, spec in split_folds(workload, protocol).items():
        audit = audits[fold]
        if audit["online_test_updates"]:
            raise RuntimeError("{} unexpectedly updates on test".format(fold))
        if set(audit["training_feedback_queries"]) != set(spec["train"]):
            raise RuntimeError("{} training feedback mismatch".format(fold))
        if set(audit["training_query_ids"]) & set(audit["test_query_ids"]):
            raise RuntimeError("{} has train/test overlap".format(fold))


def summarize_method(
    output_root,
    method,
    directory,
    workload,
    protocol,
    timeout_pg_times,
    metric_pg_times,
    run_index,
):
    result_path = (
        output_root
        / workload.lower()
        / directory
        / "{}_results.csv".format(protocol)
    )
    rows = read_rows(result_path)
    by_key = {}
    for row in rows:
        key = (row["fold"], row["query_id"])
        if key in by_key:
            raise RuntimeError(
                "{} {} has duplicate result {}".format(
                    workload,
                    method,
                    key,
                )
            )
        context = "{} {} {} {} {}".format(
            workload,
            method,
            protocol,
            row["fold"],
            row["query_id"],
        )
        validate_three_runs(row, context)
        validate_dynamic_timeout(row, timeout_pg_times, context)
        by_key[key] = row

    expected_keys = []
    for fold, spec in split_folds(workload, protocol).items():
        expected_keys.extend((fold, query_id) for query_id in spec["test"])
    if set(by_key) != set(expected_keys) or len(by_key) != len(expected_keys):
        missing = [key for key in expected_keys if key not in by_key]
        extra = [key for key in by_key if key not in set(expected_keys)]
        raise RuntimeError(
            "{} {} {} coverage mismatch; missing={}, extra={}".format(
                workload,
                method,
                protocol,
                missing[:5],
                extra[:5],
            )
        )

    pg_values = []
    method_values = []
    query_ids = []
    errors = 0
    seen_query_ids = set()
    for key in expected_keys:
        row = by_key[key]
        query_id = key[1]
        errors += bool(row["run{}_error".format(run_index)])
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        query_ids.append(query_id)
        pg_values.append(metric_pg_times[query_id])
        method_values.append(
            float(row["run{}_charged_s".format(run_index)])
        )

    speedups = [
        pg_time / method_time
        for pg_time, method_time in zip(pg_values, method_values)
    ]
    return {
        "method": method,
        "workload": workload,
        "protocol": protocol,
        "status": "measured",
        "test_records": len(expected_keys),
        "unique_queries": len(set(query_ids)),
        "pg_total_s": sum(pg_values),
        "method_total_s": sum(method_values),
        "WS": sum(pg_values) / sum(method_values),
        "GS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "Imp_pct": 100.0
        * sum(
            method_time < pg_time
            for pg_time, method_time in zip(pg_values, method_values)
        )
        / len(query_ids),
        "timeouts_or_errors": errors,
    }


def format_markdown(rows):
    lines = [
        "| Method | Workload | Split | WS | GS | Imp (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "not_applicable":
            values = ("N/A", "N/A", "N/A")
        else:
            values = (
                "{:.3f}".format(row["WS"]),
                "{:.3f}".format(row["GS"]),
                "{:.1f}".format(row["Imp_pct"]),
            )
        lines.append(
            "| {method} | {workload} | {protocol} | {ws} | {gs} | {imp} |".format(
                method=row["method"],
                workload=row["workload"],
                protocol=row["protocol"],
                ws=values[0],
                gs=values[1],
                imp=values[2],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()

    measured_rows = []
    for workload in ("JOB", "STACK", "TPCH"):
        validate_postgres(output_root, workload)
        pg_times = load_postgres_times(
            output_root / workload.lower() / "postgres.csv"
        )
        pg_run1_times = {
            row["query_id"]: float(row["run1_charged_s"])
            for row in read_rows(
                output_root / workload.lower() / "postgres.csv"
            )
        }
        validate_fastgres_labels(output_root, workload, pg_times)
        validate_tonic_feedback(output_root, workload, pg_times)

        for protocol in SPLIT_PROTOCOLS[workload]:
            validate_fold_audit(
                output_root
                / workload.lower()
                / "tonic"
                / "{}_predictions.json".format(protocol),
                workload,
                protocol,
            )
            for method, directory in METHODS.items():
                run_index = 1 if method == "FASTgres" else 3
                measured_rows.append(
                    summarize_method(
                        output_root,
                        method,
                        directory,
                        workload,
                        protocol,
                        pg_times,
                        pg_run1_times if run_index == 1 else pg_times,
                        run_index,
                    )
                )

    report_rows = list(measured_rows)
    tpch_random = {
        row["method"]: row
        for row in measured_rows
        if row["workload"] == "TPCH" and row["protocol"] == "random"
    }
    for method in METHODS:
        random_row = tpch_random[method]
        base_row = dict(random_row)
        base_row["protocol"] = "base_query"
        base_row["status"] = "equivalent_to_random"
        report_rows.append(base_row)
        report_rows.append(
            {
                "method": method,
                "workload": "TPCH",
                "protocol": "leave_one_out",
                "status": "not_applicable",
                "test_records": "",
                "unique_queries": "",
                "pg_total_s": "",
                "method_total_s": "",
                "WS": "",
                "GS": "",
                "Imp_pct": "",
                "timeouts_or_errors": "",
            }
        )

    workload_order = {"JOB": 0, "STACK": 1, "TPCH": 2}
    protocol_order = {"base_query": 0, "leave_one_out": 1, "random": 2}
    method_order = {method: idx for idx, method in enumerate(METHODS)}
    report_rows.sort(
        key=lambda row: (
            workload_order[row["workload"]],
            protocol_order[row["protocol"]],
            method_order[row["method"]],
        )
    )

    csv_path = output_root / "fastgres_tonic_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)
    markdown_path = output_root / "fastgres_tonic_summary.md"
    markdown_path.write_text(format_markdown(report_rows))
    print(markdown_path.read_text(), end="")
    print("CSV: {}".format(csv_path))


if __name__ == "__main__":
    main()
