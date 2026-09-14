#!/usr/bin/env python3
"""Train and evaluate TONIC with fold-isolated feedback."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    MEASUREMENT_COLUMNS,
    SPLIT_PROTOCOLS,
    CsvResultStore,
    connect,
    dynamic_timeout_s,
    execute_once,
    load_postgres_times,
    natural_key,
    one_result_to_row,
    query_path,
    query_sql,
    split_folds,
    sql_sha256,
)
from scripts.reproduce.tonic.tonic_common import (
    QepsNode,
    assignment_hint,
    configure_tonic_session,
    hinted_sql,
    integrate_feedback,
    job_source_feedback,
    load_skeletons,
    predict_assignment,
)


RESULT_FIELDS = [
    "result_key",
    "workload",
    "protocol",
    "fold",
    "query_id",
    "sql_path",
    "sql_sha256",
    "predicted_assignment",
    "hint",
    "pg_measured_s",
] + MEASUREMENT_COLUMNS


def load_training_feedback(path: Path, training_query_ids):
    """Load measured runtimes only after confirming a row is in the train set."""
    training_query_ids = set(training_query_ids)
    result = {query_id: {} for query_id in training_query_ids}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = row["query_id"]
            if query_id not in training_query_ids:
                continue
            result[query_id][row["assignment"]] = float(row["measured_s"])

    missing = sorted(
        (query_id for query_id, runtimes in result.items() if not runtimes),
        key=natural_key,
    )
    if missing:
        raise RuntimeError(
            "TONIC training feedback is missing for {}".format(missing[:5])
        )
    return result


def load_original_job_training_feedback(training_query_ids, skeletons):
    return {
        query_id: job_source_feedback(skeletons[query_id])
        for query_id in training_query_ids
    }


def train_and_predict_fold(
    workload,
    fold_spec,
    skeletons,
    feedback_path,
    feedback_source,
):
    training_ids = list(fold_spec["train"])
    test_ids = list(fold_spec["test"])
    overlap = set(training_ids) & set(test_ids)
    if overlap:
        raise RuntimeError(
            "train/test overlap in TONIC fold: {}".format(sorted(overlap))
        )

    loading_started = time.perf_counter()
    if feedback_source == "original_job":
        training_feedback = load_original_job_training_feedback(
            training_ids,
            skeletons,
        )
    else:
        training_feedback = load_training_feedback(feedback_path, training_ids)
    feedback_loading_s = time.perf_counter() - loading_started
    root = QepsNode()
    started = time.perf_counter()
    for query_id in sorted(training_ids, key=natural_key):
        skeleton = skeletons[query_id]
        integrate_feedback(
            root,
            skeleton,
            training_feedback[query_id],
            active_join_count=len(skeleton.steps),
        )
    training_s = time.perf_counter() - started

    prediction_started = time.perf_counter()
    predictions = {
        query_id: predict_assignment(root, skeletons[query_id])
        for query_id in test_ids
    }
    prediction_s = time.perf_counter() - prediction_started
    audit = {
        "training_query_ids": training_ids,
        "test_query_ids": test_ids,
        "training_feedback_queries": sorted(
            training_feedback,
            key=natural_key,
        ),
        "feedback_loading_s": feedback_loading_s,
        "training_s": training_s,
        "prediction_s": prediction_s,
        "online_test_updates": False,
        "feedback_source": feedback_source,
    }
    return predictions, audit


def execute_protocol(
    args,
    workload,
    protocol,
    predictions_by_fold,
    skeletons,
    pg_times,
    tonic_dir,
):
    result_path = tonic_dir / "{}_results.csv".format(protocol)
    store = CsvResultStore(result_path, RESULT_FIELDS)
    connection = connect(
        workload,
        host=args.host,
        port=args.port,
        user=args.user,
    )
    cursor = connection.cursor()
    try:
        total = sum(
            len(predictions) for predictions in predictions_by_fold.values()
        )
        result_idx = 0
        for fold, predictions in predictions_by_fold.items():
            for query_id, assignment in predictions.items():
                result_idx += 1
                skeleton = skeletons[query_id]
                sql = query_sql(workload, query_id)
                digest = sql_sha256(sql)
                timeout_s = dynamic_timeout_s(
                    pg_times[query_id], args.timeout_factor
                )
                result_key = "{}:tonic-test:{}:{}:{}".format(
                    workload,
                    protocol,
                    fold,
                    query_id,
                )
                previous = store.get(result_key)
                if (
                    previous
                    and previous.get("sql_sha256") == digest
                    and previous.get("predicted_assignment") == assignment
                    and abs(float(previous["timeout_s"]) - timeout_s) < 1e-9
                ):
                    print(
                        "[{} TONIC {} {}/{}] {} {}: resume {}s".format(
                            workload,
                            protocol,
                            result_idx,
                            total,
                            fold,
                            query_id,
                            previous["measured_s"],
                        ),
                        flush=True,
                    )
                    continue

                configure_tonic_session(cursor)
                result = execute_once(
                    cursor,
                    hinted_sql(workload, skeleton, assignment),
                    timeout_s=timeout_s,
                )
                row = {
                    "result_key": result_key,
                    "workload": workload,
                    "protocol": protocol,
                    "fold": fold,
                    "query_id": query_id,
                    "sql_path": str(
                        query_path(workload, query_id).relative_to(Path.cwd())
                    ),
                    "sql_sha256": digest,
                    "predicted_assignment": assignment,
                    "hint": assignment_hint(skeleton, assignment),
                    "pg_measured_s": pg_times[query_id],
                }
                row.update(one_result_to_row(result))
                store.append(row)
                print(
                    "[{} TONIC {} {}/{}] {} {} assignment={}: "
                    "{:.6f}s{}".format(
                        workload,
                        protocol,
                        result_idx,
                        total,
                        fold,
                        query_id,
                        assignment or "NOJOIN",
                        result.charged_runtime_s,
                        " timeout/error" if result.error else "",
                    ),
                    flush=True,
                )
    finally:
        cursor.close()
        connection.close()


def run_workload(args, workload):
    workload_dir = args.output_root / workload.lower()
    measured_tonic_dir = workload_dir / "tonic"
    tonic_dir = (
        workload_dir / "tonic_original"
        if args.feedback_source == "original_job"
        else measured_tonic_dir
    )
    tonic_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = measured_tonic_dir / "feedback.csv"
    skeleton_path = measured_tonic_dir / "skeletons.json"
    if not skeleton_path.is_file() or (
        args.feedback_source == "measured" and not feedback_path.is_file()
    ):
        raise RuntimeError(
            "{} TONIC feedback/skeletons have not been generated".format(
                workload
            )
        )

    skeletons = load_skeletons(skeleton_path)
    pg_times = load_postgres_times(
        workload_dir / "postgres.csv",
        time_column=args.pg_time_column,
    )
    for protocol in SPLIT_PROTOCOLS[workload]:
        predictions_by_fold = {}
        audit_by_fold = {}
        for fold, spec in split_folds(workload, protocol).items():
            predictions, audit = train_and_predict_fold(
                workload,
                spec,
                skeletons,
                feedback_path,
                args.feedback_source,
            )
            predictions_by_fold[fold] = predictions
            audit_by_fold[fold] = audit
            print(
                "{} TONIC {} trained on {} and predicted {} queries for "
                "{} in {:.3f}s + {:.3f}s".format(
                    workload,
                    protocol,
                    len(audit["training_query_ids"]),
                    len(predictions),
                    fold,
                    audit["training_s"],
                    audit["prediction_s"],
                ),
                flush=True,
            )

        prediction_path = tonic_dir / "{}_predictions.json".format(protocol)
        prediction_path.write_text(
            json.dumps(
                {
                    "predictions": predictions_by_fold,
                    "fold_audit": audit_by_fold,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        execute_protocol(
            args,
            workload,
            protocol,
            predictions_by_fold,
            skeletons,
            pg_times,
            tonic_dir,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=["JOB", "STACK", "TPCH", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--timeout-factor", type=float, default=5.0)
    parser.add_argument(
        "--pg-time-column",
        default="run1_charged_s",
        help="PostgreSQL timing column used for timeouts and reported metrics",
    )
    parser.add_argument(
        "--feedback-source",
        choices=["measured", "original_job"],
        default="measured",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.feedback_source == "original_job" and args.workload != "JOB":
        parser.error("--feedback-source original_job requires --workload JOB")

    workloads = (
        ("JOB", "STACK", "TPCH")
        if args.workload == "all"
        else (args.workload,)
    )
    for workload in workloads:
        run_workload(args, workload)


if __name__ == "__main__":
    main()
