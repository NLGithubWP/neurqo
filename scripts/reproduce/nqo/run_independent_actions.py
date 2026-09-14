#!/usr/bin/env python3
"""Collect and export the standalone-Action measurements used by the paper."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PGDB_ROOT = REPO.parent / "pgdb"
RUNNER = Path(__file__).with_name("run.py")
DEFAULT_OUTPUT = (
    REPO / "results" / "benchmark" / "nqo" / "nqo_independent_action_runs.csv"
)
DEFAULT_RUNTIME = (
    PGDB_ROOT / ".nqo_runtime" / "reproduction" / "independent-actions"
)
DATASET_SIZES = {"JOB": 113, "STACK": 112}
PROFILE_METHODS = {
    "aja_conservative": "AJA (Cons.)",
    "lip_selective": "LIP (Sel.)",
    "query_split": "QuerySplit",
    "top5": "TOP-5 (DP)",
}
CSV_FIELDS = (
    "dataset",
    "query_id",
    "sql_path",
    "profile",
    "repetition",
    "status",
    "first_runtime_ms",
    "charged_runtime_ms",
    "timeout_limit_ms",
    "result_hash",
    "result_rows",
    "materializations",
    "materialized_rows",
    "materialized_bytes",
    "cache_id",
)


def decode_payload(encoding: str, payload: bytes) -> Any:
    raw = bytes(payload)
    if encoding == "zlib":
        raw = zlib.decompress(raw)
    elif encoding not in {"json", "plain", "raw"}:
        raise ValueError(f"unsupported cache encoding: {encoding}")
    return json.loads(raw.decode("utf-8"))


def run_method(
    args: argparse.Namespace,
    *,
    dataset: str,
    method: str,
    results_path: Path,
    cache_path: Path,
    port: int,
) -> None:
    runtime_dir = args.runtime_root / dataset.lower() / method
    command = [
        sys.executable,
        str(RUNNER),
        "--dataset",
        dataset.lower(),
        "--method",
        method,
        "--output",
        str(results_path),
        "--runtime-dir",
        str(runtime_dir),
        "--container",
        args.container,
        "--server-port",
        str(port),
        "--host",
        args.host,
        "--pg-port",
        str(args.pg_port),
        "--user",
        args.user,
        "--sql-execution-lock",
        str(args.sql_execution_lock),
        "--sql-execution-slots",
        str(args.sql_execution_slots),
    ]
    if method != "postgres":
        command.extend(
            [
                "--cache",
                str(cache_path),
                "--cache-miss",
                "execute",
            ]
        )
    subprocess.run(command, cwd=REPO, check=True)


def cache_records(path: Path) -> dict[str, sqlite3.Row]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return {
            str(row["cache_id"]): row
            for row in connection.execute("SELECT * FROM replay_cache")
        }
    finally:
        connection.close()


def export_rows(runtime_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_to_profile = {value: key for key, value in PROFILE_METHODS.items()}
    for dataset, expected_queries in DATASET_SIZES.items():
        dataset_root = runtime_root / dataset.lower()
        results_path = dataset_root / "runs.csv"
        cache_path = dataset_root / "experience.sql"
        if not results_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(
                f"incomplete independent-Action run under {dataset_root}"
            )
        records = cache_records(cache_path)
        with results_path.open(newline="", encoding="utf-8") as handle:
            selected = [
                row
                for row in csv.DictReader(handle)
                if row["dataset"] == dataset and row["method"] in method_to_profile
            ]
        counts = {
            profile: sum(row["method"] == method for row in selected)
            for profile, method in PROFILE_METHODS.items()
        }
        if any(value != expected_queries for value in counts.values()):
            raise RuntimeError(
                f"incomplete independent-Action rows for {dataset}: {counts}"
            )
        for source in selected:
            cache_id = source["cache_id"]
            if not cache_id or cache_id not in records:
                raise RuntimeError(
                    f"missing execution cache entry for {dataset}/{source['sql_path']}"
                )
            record = records[cache_id]
            events = decode_payload(
                str(record["db_events_encoding"]), record["db_events_payload"]
            )
            if not isinstance(events, list):
                raise TypeError(f"non-list DB events for cache entry {cache_id}")
            split_events = [
                event
                for event in events
                if isinstance(event, dict) and event.get("phase") == "split"
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "query_id": Path(source["sql_path"]).stem,
                    "sql_path": source["sql_path"],
                    "profile": method_to_profile[source["method"]],
                    "repetition": 0,
                    "status": record["status"],
                    "first_runtime_ms": record["first_runtime_ms"],
                    "charged_runtime_ms": record["charged_runtime_ms"],
                    "timeout_limit_ms": record["timeout_limit_ms"],
                    "result_hash": record["result_hash"] or "",
                    "result_rows": (
                        "" if record["result_rows"] is None else record["result_rows"]
                    ),
                    "materializations": len(split_events),
                    "materialized_rows": sum(
                        int((event.get("materialized") or {}).get("rows") or 0)
                        for event in split_events
                    ),
                    "materialized_bytes": sum(
                        int((event.get("materialized") or {}).get("bytes") or 0)
                        for event in split_events
                    ),
                    "cache_id": cache_id,
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["dataset"], row["profile"], row["query_id"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    parser = argparse.ArgumentParser(
        description=(
            "Collect JOB/STACK standalone Actions into a private runtime "
            "buffer and export the paper's compact CSV"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--container", default="pgdb_dev_opt")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--base-port", type=int, default=18500)
    parser.add_argument("--sql-execution-slots", type=int, default=1)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.sql_execution_lock = args.runtime_root / "sql-execution.lock"
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists; pass --overwrite: {args.output}")
    if args.sql_execution_slots < 1:
        parser.error("--sql-execution-slots must be positive")
    try:
        args.runtime_root.relative_to(PGDB_ROOT.resolve())
    except ValueError:
        parser.error(f"runtime root must be inside {PGDB_ROOT}")
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    if not args.export_only:
        port = args.base_port
        for dataset in DATASET_SIZES:
            dataset_root = args.runtime_root / dataset.lower()
            dataset_root.mkdir(parents=True, exist_ok=True)
            results_path = dataset_root / "runs.csv"
            cache_path = dataset_root / "experience.sql"
            run_method(
                args,
                dataset=dataset,
                method="postgres",
                results_path=results_path,
                cache_path=cache_path,
                port=port,
            )
            for method in PROFILE_METHODS:
                port += 1
                run_method(
                    args,
                    dataset=dataset,
                    method=method,
                    results_path=results_path,
                    cache_path=cache_path,
                    port=port,
                )

    rows = export_rows(args.runtime_root)
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
