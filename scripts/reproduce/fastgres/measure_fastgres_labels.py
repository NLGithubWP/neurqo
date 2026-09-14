#!/usr/bin/env python3
"""Generate complete 64-hint FASTgres labels with upstream-style pruning."""

from __future__ import annotations

import argparse
import csv
import json
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
    reset_session,
    sql_sha256,
    workload_query_ids,
)


HINT_GUCS = [
    "enable_hashjoin",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_indexscan",
    "enable_seqscan",
    "enable_indexonlyscan",
]

FIELDS = [
    "result_key",
    "workload",
    "query_id",
    "sql_path",
    "sql_sha256",
    "hint",
    "hint_bits",
    "pg_measured_s",
] + MEASUREMENT_COLUMNS


def hint_bits(hint: int):
    return [int(value) for value in bin(hint)[2:].zfill(len(HINT_GUCS))]


def set_fastgres_hint(cursor, hint: int) -> None:
    bits = hint_bits(hint)
    for guc, enabled in zip(HINT_GUCS, bits):
        cursor.execute("SET {} = {}".format(guc, "true" if enabled else "false"))


def label_runtime(row) -> float:
    value = row.get("run1_charged_s") or row.get("measured_s")
    if not value:
        raise RuntimeError("FASTgres label row has no charged first run")
    return float(value)


def label_succeeded(row) -> bool:
    return not (row.get("run1_error") or row.get("measured_error"))


def build_archive(labels_path: Path, archive_path: Path) -> int:
    by_query = {}
    with labels_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_query.setdefault(row["query_id"], {})[int(row["hint"])] = row

    archive = {}
    for query_id, rows in by_query.items():
        if len(rows) != 64:
            continue
        pg_time = float(rows[63]["pg_measured_s"])
        best_hint = 63
        best_time = pg_time
        values = {}
        for hint in range(63, -1, -1):
            row = rows[hint]
            runtime = label_runtime(row)
            values[hint] = runtime
            if label_succeeded(row) and runtime < best_time:
                best_hint = hint
                best_time = runtime
        archive["{}.sql".format(query_id)] = {
            **{str(hint): runtime for hint, runtime in sorted(values.items())},
            "opt": best_hint,
        }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n")
    temporary.replace(archive_path)
    return len(archive)


def measure_workload(args, workload: str) -> None:
    workload = workload.upper()
    workload_dir = args.output_root / workload.lower()
    pg_path = workload_dir / "postgres.csv"
    pg_times = load_postgres_times(pg_path)
    labels_path = workload_dir / "fastgres" / "labels.csv"
    archive_path = workload_dir / "fastgres" / "archive.json"
    store = CsvResultStore(labels_path, FIELDS)
    timing = PhaseTiming(
        workload_dir / "fastgres" / "label_collection_timing.json",
        method="FASTgres",
        workload=workload,
        phase="label_collection",
        protocol="full_64_single_run_best_so_far",
        preexisting_records=len(store._completed),
        database_execution_s_baseline=sum(
            measurement_database_execution_s(row)
            for row in store._completed.values()
        ),
    )

    query_ids = workload_query_ids(workload)
    if args.query_limit is not None:
        query_ids = query_ids[: args.query_limit]
    hints = list(range(63, -1, -1))
    if args.hint_limit is not None:
        hints = hints[: args.hint_limit]

    missing_pg = sorted(set(query_ids) - set(pg_times))
    if missing_pg:
        raise RuntimeError(
            "{} PostgreSQL results are missing, starting with {}".format(
                workload, missing_pg[:5]
            )
        )

    connection = None
    cursor = None
    status = "failed"
    try:
        connection = connect(
            workload,
            host=args.host,
            port=args.port,
            user=args.user,
        )
        cursor = connection.cursor()
        total = len(query_ids) * len(hints)
        completed = 0
        for query_idx, query_id in enumerate(query_ids, start=1):
            sql = query_sql(workload, query_id)
            digest = sql_sha256(sql)
            pg_time = pg_times[query_id]
            best_time = pg_time
            best_hint = 63
            query_added = 0
            query_resumed = 0
            for hint_idx, hint in enumerate(hints, start=1):
                completed += 1
                result_key = "{}:fastgres-label:{}:{}".format(
                    workload, query_id, hint
                )
                previous = store.get(result_key)
                if (
                    previous
                    and previous.get("sql_sha256") == digest
                    and (previous.get("run1_charged_s") or previous.get("measured_s"))
                ):
                    runtime = label_runtime(previous)
                    if label_succeeded(previous) and runtime < best_time:
                        best_time = runtime
                        best_hint = hint
                    timing.record(resumed=True)
                    query_resumed += 1
                    if args.show_resumes:
                        print(
                            "[{} {}/{} q={}/{} h={}/{}] {} hint={}: "
                            "resume {}s best={} {:.6f}s".format(
                                workload,
                                completed,
                                total,
                                query_idx,
                                len(query_ids),
                                hint_idx,
                                len(hints),
                                query_id,
                                hint,
                                runtime,
                                best_hint,
                                best_time,
                            ),
                            flush=True,
                        )
                    continue

                timeout_s = best_time
                reset_session(cursor)
                set_fastgres_hint(cursor, hint)
                result = execute_once(cursor, sql, timeout_s=timeout_s)
                row = {
                    "result_key": result_key,
                    "workload": workload,
                    "query_id": query_id,
                    "sql_path": str(query_path(workload, query_id).relative_to(Path.cwd())),
                    "sql_sha256": digest,
                    "hint": hint,
                    "hint_bits": "".join(str(value) for value in hint_bits(hint)),
                    "pg_measured_s": pg_time,
                }
                row.update(one_result_to_row(result))
                store.append(row)
                timing.record(
                    resumed=False,
                    database_execution_s=result.charged_runtime_s,
                )
                query_added += 1
                if not result.error and result.charged_runtime_s < best_time:
                    best_time = result.charged_runtime_s
                    best_hint = hint
                print(
                    "[{} {}/{} q={}/{} h={}/{}] {} hint={}: "
                    "measured={:.6f}s timeout={:.6f}s best={} {:.6f}s{}".format(
                        workload,
                        completed,
                        total,
                        query_idx,
                        len(query_ids),
                        hint_idx,
                        len(hints),
                        query_id,
                        hint,
                        result.charged_runtime_s,
                        timeout_s,
                        best_hint,
                        best_time,
                        " timeout/error" if result.error else "",
                    ),
                    flush=True,
                )
            if query_added:
                build_archive(labels_path, archive_path)
            if query_resumed and not args.show_resumes:
                print(
                    "[{} q={}/{}] {}: resume={} new={} best={} {:.6f}s".format(
                        workload,
                        query_idx,
                        len(query_ids),
                        query_id,
                        query_resumed,
                        query_added,
                        best_hint,
                        best_time,
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

    completed_queries = build_archive(labels_path, archive_path)
    print(
        "{} FASTgres archive: {}/{} complete queries at {}".format(
            workload, completed_queries, len(workload_query_ids(workload)), archive_path
        )
    )


def main() -> None:
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
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--hint-limit", type=int)
    parser.add_argument("--show-resumes", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    workloads = ("JOB", "STACK", "TPCH") if args.workload == "all" else (args.workload,)
    for workload in workloads:
        measure_workload(args, workload)


if __name__ == "__main__":
    main()
