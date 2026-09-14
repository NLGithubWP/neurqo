#!/usr/bin/env python3
"""Measure one PostgreSQL execution per query for baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    MEASUREMENT_COLUMNS,
    CsvResultStore,
    connect,
    execute_once,
    one_result_to_row,
    query_path,
    query_sql,
    reset_session,
    sql_sha256,
    workload_query_ids,
)


FIELDS = [
    "result_key",
    "workload",
    "query_id",
    "sql_path",
    "sql_sha256",
] + MEASUREMENT_COLUMNS


def measure_workload(args, workload: str) -> None:
    workload = workload.upper()
    out_path = args.output_root / workload.lower() / "postgres.csv"
    store = CsvResultStore(out_path, FIELDS)
    query_ids = workload_query_ids(workload)
    if args.limit is not None:
        query_ids = query_ids[: args.limit]

    connection = connect(
        workload,
        host=args.host,
        port=args.port,
        user=args.user,
    )
    cursor = connection.cursor()
    try:
        for idx, query_id in enumerate(query_ids, start=1):
            sql = query_sql(workload, query_id)
            digest = sql_sha256(sql)
            result_key = "{}:postgres:{}".format(workload, query_id)
            previous = store.get(result_key)
            if (
                previous
                and previous.get("sql_sha256") == digest
                and not previous.get("measured_error")
            ):
                print(
                    "[{} {}/{}] {}: resume {:.6f}s".format(
                        workload,
                        idx,
                        len(query_ids),
                        query_id,
                        float(previous["measured_s"]),
                    ),
                    flush=True,
                )
                continue

            reset_session(cursor)
            result = execute_once(cursor, sql, timeout_s=None)
            if result.error:
                raise RuntimeError(
                    "PostgreSQL failed for {} {}: {}".format(
                        workload, query_id, result.error
                    )
                )
            row = {
                "result_key": result_key,
                "workload": workload,
                "query_id": query_id,
                "sql_path": str(query_path(workload, query_id).relative_to(Path.cwd())),
                "sql_sha256": digest,
            }
            row.update(one_result_to_row(result))
            store.append(row)
            print(
                "[{} {}/{}] {}: measured={:.6f}s".format(
                    workload,
                    idx,
                    len(query_ids),
                    query_id,
                    result.charged_runtime_s,
                ),
                flush=True,
            )
    finally:
        cursor.close()
        connection.close()


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
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    workloads = ("JOB", "STACK", "TPCH") if args.workload == "all" else (args.workload,)
    for workload in workloads:
        measure_workload(args, workload)


if __name__ == "__main__":
    main()
