#!/usr/bin/env python3
"""Validate and summarize one-run TONIC reproduction measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


FIELDS = [
    "method",
    "workload",
    "protocol",
    "test_records",
    "unique_queries",
    "pg_total_s",
    "method_total_s",
    "WS",
    "GS",
    "Imp_pct",
    "timeouts_or_errors",
]


def rows(path: Path):
    if not path.is_file():
        raise RuntimeError("missing {}".format(path))
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_one_run(row, context):
    if bool(row["run1_s"]) == bool(row["run1_error"]):
        raise RuntimeError(
            "{} must contain exactly one runtime or error".format(context)
        )
    if not row["run1_charged_s"]:
        raise RuntimeError("{} is missing charged runtime".format(context))
    if float(row["measured_s"]) != float(row["run1_charged_s"]):
        raise RuntimeError("{} does not report run 1".format(context))
    for index in (2, 3):
        if (
            row["run{}_s".format(index)]
            or row["run{}_charged_s".format(index)]
            or row["run{}_error".format(index)]
        ):
            raise RuntimeError("{} unexpectedly has later runs".format(context))


def validate_feedback(output_root: Path, workload: str):
    tonic_dir = output_root / workload.lower() / "tonic"
    skeletons = load_skeletons(tonic_dir / "skeletons.json")
    by_query = {query_id: {} for query_id in workload_query_ids(workload)}
    for row in rows(tonic_dir / "feedback.csv"):
        context = "{} TONIC feedback {} {}".format(
            workload, row["query_id"], row["assignment"] or "NOJOIN"
        )
        validate_one_run(row, context)
        timeout = float(row["timeout_s"])
        if not 0.1 <= timeout <= 10.0:
            raise RuntimeError("{} has invalid timeout {}".format(context, timeout))
        by_query[row["query_id"]][row["assignment"]] = row

    for query_id, skeleton in skeletons.items():
        feedback = by_query[query_id]
        if not feedback:
            raise RuntimeError("{} has no feedback".format(query_id))
        successful = {
            assignment: float(row["run1_charged_s"])
            for assignment, row in feedback.items()
            if not row["run1_error"]
        }
        center = successful or {
            assignment: float(row["run1_charged_s"])
            for assignment, row in feedback.items()
        }
        best = min(center, key=lambda assignment: (center[assignment], assignment))
        required = set()
        for index in range(skeleton.decision_count):
            required.add(best[:index] + "H" + best[index + 1 :])
            required.add(best[:index] + "N" + best[index + 1 :])
        if not required.issubset(feedback):
            raise RuntimeError(
                "{} is missing required one-bit feedback".format(query_id)
            )


def validate_audit(
    path: Path,
    workload: str,
    protocol: str,
    feedback_source: str,
):
    content = json.loads(path.read_text())
    for fold, spec in split_folds(workload, protocol).items():
        audit = content["fold_audit"][fold]
        if audit["online_test_updates"]:
            raise RuntimeError("{} enables online test updates".format(fold))
        if audit["feedback_source"] != feedback_source:
            raise RuntimeError(
                "{} uses feedback source {}, expected {}".format(
                    fold,
                    audit["feedback_source"],
                    feedback_source,
                )
            )
        if set(audit["training_feedback_queries"]) != set(spec["train"]):
            raise RuntimeError("{} loaded non-training feedback".format(fold))
        if set(audit["training_query_ids"]) & set(audit["test_query_ids"]):
            raise RuntimeError("{} has train/test overlap".format(fold))


def summarize_protocol(
    output_root: Path,
    workload: str,
    protocol: str,
    pg_times,
    result_directory: str,
    executed_timeout_factor: float,
    metric_timeout_factor: float,
):
    tonic_dir = output_root / workload.lower() / result_directory
    result_rows = rows(tonic_dir / "{}_results.csv".format(protocol))
    by_key = {}
    for row in result_rows:
        key = (row["fold"], row["query_id"])
        if key in by_key:
            raise RuntimeError("duplicate TONIC result {}".format(key))
        context = "{} TONIC {} {} {}".format(
            workload, protocol, row["fold"], row["query_id"]
        )
        validate_one_run(row, context)
        expected_timeout = dynamic_timeout_s(
            pg_times[row["query_id"]],
            executed_timeout_factor,
        )
        if abs(float(row["timeout_s"]) - expected_timeout) > 1e-9:
            raise RuntimeError("{} has wrong final timeout".format(context))
        by_key[key] = row

    expected = []
    for fold, spec in split_folds(workload, protocol).items():
        expected.extend((fold, query_id) for query_id in spec["test"])
    if set(by_key) != set(expected):
        raise RuntimeError("{} result coverage mismatch".format(protocol))

    seen = set()
    pg_values = []
    method_values = []
    errors = 0
    for key in expected:
        row = by_key[key]
        query_id = key[1]
        pg_time = pg_times[query_id]
        metric_timeout = dynamic_timeout_s(
            pg_time,
            metric_timeout_factor,
        )
        raw_method_time = float(row["run1_charged_s"])
        errors += bool(row["run1_error"]) or raw_method_time > metric_timeout
        if query_id in seen:
            continue
        seen.add(query_id)
        pg_values.append(pg_time)
        method_values.append(min(raw_method_time, metric_timeout))
    speedups = [
        pg_time / method_time
        for pg_time, method_time in zip(pg_values, method_values)
    ]
    return {
        "method": "TONIC",
        "workload": workload,
        "protocol": protocol,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=("JOB", "STACK", "TPCH"),
        required=True,
    )
    parser.add_argument(
        "--feedback-source",
        choices=("measured", "original_job"),
        default="measured",
    )
    parser.add_argument("--timeout-factor", type=float, default=5.0)
    parser.add_argument(
        "--metric-timeout-factor",
        type=float,
        help="Censor measured runtimes at this factor without re-execution",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    args = parser.parse_args()
    if args.feedback_source == "original_job" and args.workload != "JOB":
        parser.error("--feedback-source original_job requires --workload JOB")
    output_root = args.output_root.resolve()
    workload = args.workload
    metric_timeout_factor = (
        args.metric_timeout_factor
        if args.metric_timeout_factor is not None
        else args.timeout_factor
    )
    result_directory = (
        "tonic_original"
        if args.feedback_source == "original_job"
        else "tonic"
    )
    pg_times = load_postgres_times(
        output_root / workload.lower() / "postgres.csv",
        time_column="run1_charged_s",
    )
    if args.feedback_source == "measured":
        validate_feedback(output_root, workload)
    summaries = []
    for protocol in SPLIT_PROTOCOLS[workload]:
        validate_audit(
            output_root
            / workload.lower()
            / result_directory
            / "{}_predictions.json".format(protocol),
            workload,
            protocol,
            args.feedback_source,
        )
        summaries.append(
            summarize_protocol(
                output_root,
                workload,
                protocol,
                pg_times,
                result_directory,
                args.timeout_factor,
                metric_timeout_factor,
            )
        )

    summary_name = (
        "summary.csv"
        if metric_timeout_factor == args.timeout_factor
        else "summary_{:g}x.csv".format(metric_timeout_factor)
    )
    output_path = (
        output_root / workload.lower() / result_directory / summary_name
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(
            "{workload} TONIC {protocol}: WS={WS:.6f} GS={GS:.6f} "
            "Imp={Imp_pct:.3f}% errors={timeouts_or_errors}".format(**summary)
        )


if __name__ == "__main__":
    main()
