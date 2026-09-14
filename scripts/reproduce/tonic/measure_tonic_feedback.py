#!/usr/bin/env python3
"""Generate TONIC feedback with sequential one-run adaptive timeouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import random
from pathlib import Path

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    MEASUREMENT_COLUMNS,
    CsvResultStore,
    PhaseTiming,
    connect,
    execute_once,
    load_postgres_times,
    measurement_database_execution_s,
    one_result_to_row,
    query_path,
    query_sql,
    sql_sha256,
    workload_query_ids,
)
from scripts.reproduce.tonic.tonic_common import (
    assignment_hint,
    build_skeletons,
    configure_tonic_session,
    explain_json,
    hinted_sql,
    initial_assignments,
    load_skeletons,
    save_skeletons,
)


FIELDS = [
    "result_key",
    "workload",
    "query_id",
    "sql_path",
    "sql_sha256",
    "assignment",
    "join_count",
    "hint",
    "candidate_source",
    "pg_measured_s",
] + MEASUREMENT_COLUMNS


def existing_query_feedback(store, query_id):
    prefix = ":tonic-feedback:{}:".format(query_id)
    runtimes = {}
    successful = {}
    for key, row in store._completed.items():
        if prefix in key:
            assignment = row["assignment"]
            runtime = float(row["measured_s"])
            runtimes[assignment] = runtime
            if not row["run1_error"]:
                successful[assignment] = runtime
    return runtimes, successful


def feedback_timeout_s(
    pg_time,
    best_successful,
    timeout_factor,
    timeout_cap,
    timeout_floor,
):
    reference = min(pg_time, best_successful)
    return min(timeout_cap, max(timeout_floor, timeout_factor * reference))


def sampled_assignments(skeleton, limit, seed):
    decision_count = skeleton.decision_count
    total = 1 << decision_count
    if total <= limit:
        return [
            "".join(bits)
            for bits in itertools.product(("H", "N"), repeat=decision_count)
        ]

    mandatory = [
        "H" * decision_count,
        "N" * decision_count,
        skeleton.default_assignment or "H" * decision_count,
    ]
    assignments = []
    seen = set()
    for assignment in mandatory:
        if assignment not in seen:
            seen.add(assignment)
            assignments.append(assignment)

    digest = hashlib.sha256(
        "{}:{}".format(seed, skeleton.query_id).encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    while len(assignments) < limit:
        value = rng.getrandbits(decision_count)
        assignment = format(value, "0{}b".format(decision_count)).translate(
            str.maketrans("01", "HN")
        )
        if assignment in seen:
            continue
        seen.add(assignment)
        assignments.append(assignment)
    return assignments


def revision_assignments(workload, skeleton, sample_limit, seed):
    if (
        workload == "STACK"
        and skeleton.query_id.split("_", 1)[0] in {"q2", "q3"}
    ):
        return (
            sampled_assignments(skeleton, sample_limit, seed),
            "sampled_{}".format(sample_limit),
        )
    return initial_assignments(workload, skeleton), (
        "official_full" if workload == "JOB" else "exhaustive"
    )


def rank_sampled_assignments(cursor, workload, skeleton, assignments):
    """Run only cost-based EXPLAIN and try promising sampled hints first."""
    configure_tonic_session(cursor)
    ranked = []
    for assignment in assignments:
        try:
            plan = explain_json(
                cursor,
                hinted_sql(workload, skeleton, assignment),
            )
            cost = float(plan["Total Cost"])
        except Exception:
            cursor.connection.rollback()
            cost = float("inf")
        ranked.append((cost, assignment))
    return [assignment for _, assignment in sorted(ranked)]


def execute_assignment(
    args,
    workload,
    cursor,
    store,
    timing,
    skeleton,
    assignment,
    candidate_source,
    pg_time,
    best_successful,
):
    sql = query_sql(workload, skeleton.query_id)
    digest = sql_sha256(sql)
    timeout_s = feedback_timeout_s(
        pg_time,
        best_successful,
        args.timeout_factor,
        args.timeout_cap,
        args.timeout_floor,
    )
    result_key = "{}:tonic-feedback:{}:{}".format(
        workload, skeleton.query_id, assignment or "NOJOIN"
    )
    previous = store.get(result_key)
    if (
        previous
        and previous.get("sql_sha256") == digest
        and abs(float(previous["timeout_s"]) - timeout_s) < 1e-9
    ):
        timing.record(resumed=True)
        return (
            float(previous["measured_s"]),
            not bool(previous["run1_error"]),
            True,
        )

    configure_tonic_session(cursor)
    result = execute_once(
        cursor,
        hinted_sql(workload, skeleton, assignment),
        timeout_s=timeout_s,
    )
    row = {
        "result_key": result_key,
        "workload": workload,
        "query_id": skeleton.query_id,
        "sql_path": str(query_path(workload, skeleton.query_id).relative_to(Path.cwd())),
        "sql_sha256": digest,
        "assignment": assignment,
        "join_count": skeleton.decision_count,
        "hint": assignment_hint(skeleton, assignment),
        "candidate_source": candidate_source,
        "pg_measured_s": pg_time,
    }
    row.update(one_result_to_row(result))
    store.append(row)
    timing.record(
        resumed=False,
        database_execution_s=result.charged_runtime_s,
    )
    return result.charged_runtime_s, result.error is None, False


def measure_workload(args, workload):
    workload = workload.upper()
    workload_dir = args.output_root / workload.lower()
    pg_times = load_postgres_times(
        workload_dir / "postgres.csv",
        time_column=args.pg_time_column,
    )
    feedback_path = workload_dir / "tonic" / "feedback.csv"
    skeleton_path = workload_dir / "tonic" / "skeletons.json"
    store = CsvResultStore(feedback_path, FIELDS)
    timing = PhaseTiming(
        workload_dir / "tonic" / "feedback_collection_timing.json",
        method="TONIC",
        workload=workload,
        phase="feedback_collection",
        protocol="one_run_adaptive_2x_best_floor_0.1s_cap_10s",
        preexisting_records=len(store._completed),
        database_execution_s_baseline=sum(
            measurement_database_execution_s(row)
            for row in store._completed.values()
        ),
    )

    connection = None
    cursor = None
    status = "failed"
    try:
        connection = connect(
            workload, host=args.host, port=args.port, user=args.user
        )
        cursor = connection.cursor()
        if skeleton_path.is_file():
            skeletons = load_skeletons(skeleton_path)
        else:
            skeletons = build_skeletons(cursor, workload)
            save_skeletons(skeleton_path, skeletons)
        if args.skeletons_only:
            status = "completed"
            print(
                "{} TONIC skeletons: {} queries at {}".format(
                    workload, len(skeletons), skeleton_path
                ),
                flush=True,
            )
            return
        query_ids = workload_query_ids(workload)
        if args.query_limit is not None:
            query_ids = query_ids[: args.query_limit]

        for query_idx, query_id in enumerate(query_ids, start=1):
            skeleton = skeletons[query_id]
            assignments, candidate_source = revision_assignments(
                workload,
                skeleton,
                args.stack_sample_limit,
                args.seed,
            )
            if candidate_source.startswith("sampled_"):
                print(
                    "[{} TONIC rank q={}/{}] {} cost-ranking {} candidates".format(
                        workload,
                        query_idx,
                        len(query_ids),
                        query_id,
                        len(assignments),
                    ),
                    flush=True,
                )
                assignments = rank_sampled_assignments(
                    cursor,
                    workload,
                    skeleton,
                    assignments,
                )
            if args.candidate_limit is not None:
                assignments = assignments[: args.candidate_limit]
            # Replay the deterministic candidate order when resuming so each
            # stored row is checked against the timeout it originally received.
            best_successful = pg_times[query_id]
            for candidate_idx, assignment in enumerate(assignments, start=1):
                runtime, success, resumed = execute_assignment(
                    args,
                    workload,
                    cursor,
                    store,
                    timing,
                    skeleton,
                    assignment,
                    candidate_source,
                    pg_times[query_id],
                    best_successful,
                )
                if success:
                    best_successful = min(best_successful, runtime)
                print(
                    "[{} TONIC feedback q={}/{} c={}/{}] {} {}: {}{:.6f}s".format(
                        workload,
                        query_idx,
                        len(query_ids),
                        candidate_idx,
                        len(assignments),
                        query_id,
                        assignment or "NOJOIN",
                        "resume " if resumed else "",
                        runtime,
                    ),
                    flush=True,
                )

            # Complete the one-operator alternatives required by QEP-S. Repeat
            # if a newly measured alternative becomes the local best.
            if args.candidate_limit is None:
                augment_round = 0
                while True:
                    runtimes, successful = existing_query_feedback(
                        store, query_id
                    )
                    center_runtimes = successful or runtimes
                    best = min(
                        center_runtimes,
                        key=lambda assignment: (
                            center_runtimes[assignment],
                            assignment,
                        ),
                    )
                    if successful:
                        best_successful = successful[best]
                    else:
                        best_successful = pg_times[query_id]
                    required = set()
                    for idx in range(skeleton.decision_count):
                        required.add(best[:idx] + "H" + best[idx + 1 :])
                        required.add(best[:idx] + "N" + best[idx + 1 :])
                    missing = sorted(required - set(runtimes))
                    if not missing:
                        break
                    augment_round += 1
                    for extra_idx, assignment in enumerate(missing, start=1):
                        runtime, success, resumed = execute_assignment(
                            args,
                            workload,
                            cursor,
                            store,
                            timing,
                            skeleton,
                            assignment,
                            "required_one_flip_r{}".format(augment_round),
                            pg_times[query_id],
                            best_successful,
                        )
                        if success:
                            best_successful = min(best_successful, runtime)
                        print(
                            "[{} TONIC augment r={} q={}/{} c={}/{}] "
                            "{} {}: {}{:.6f}s".format(
                                workload,
                                augment_round,
                                query_idx,
                                len(query_ids),
                                extra_idx,
                                len(missing),
                                query_id,
                                assignment or "NOJOIN",
                                "resume " if resumed else "",
                                runtime,
                            ),
                            flush=True,
                        )
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
        timing.finish(status)


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
    parser.add_argument("--timeout-factor", type=float, default=2.0)
    parser.add_argument("--timeout-cap", type=float, default=10.0)
    parser.add_argument("--timeout-floor", type=float, default=0.1)
    parser.add_argument("--pg-time-column", default="run1_charged_s")
    parser.add_argument("--stack-sample-limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument(
        "--skeletons-only",
        action="store_true",
        help="extract query skeletons without executing feedback candidates",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    workloads = ("JOB", "STACK", "TPCH") if args.workload == "all" else (args.workload,)
    for workload in workloads:
        measure_workload(args, workload)


if __name__ == "__main__":
    main()
