#!/usr/bin/env python3
"""Run the fixed-depth query-decomposition microbenchmark."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PGDB_ROOT = ROOT.parent / "pgdb"
ACTION_RUNNER_MODULE = "benchmarking.action_runner"
NQO_RUNS = ROOT / "results" / "benchmark" / "nqo" / "nqo_runs.csv"
OUTPUT_CSV = ROOT / "results" / "benchmark" / "nqo" / "nqo_decomposition_depth.csv"
TEMP_ROOT = ROOT / "results" / "benchmark" / "nqo" / "temp_decomposition_depth"
RAW_ROOT = TEMP_ROOT / "raw"
BASELINE_ROOT = TEMP_ROOT / "baselines"
LOG_ROOT = ROOT / ".local" / "logs" / "decomposition_depth"
LOCK_PATH = PGDB_ROOT / ".nqo_runtime" / "reproduction" / "locks" / "decomposition-depth.lock"
VERSION = "v3-20260806"

# Each workload includes a deep-decomposition win, a second structural regime,
# and a query for which full decomposition is not beneficial.
QUERY_SPECS = {
    "JOB": {
        "29a": 6,
        "22c": 4,
        "33a": 6,
    },
    "STACK": {
        "q3_q3-099": 7,
        "q2_q2-012": 7,
        "q13_935e2051": 5,
    },
}

CSV_FIELDS = (
    "dataset",
    "query_id",
    "sql_path",
    "method",
    "requested_split_rounds",
    "actual_split_rounds",
    "executed_units",
    "runtime_ms",
    "charged_runtime_ms",
    "speedup",
    "split_total_ms",
    "final_total_ms",
    "policy_ms",
    "planning_ms",
    "execution_ms",
    "materialized_rows",
    "materialized_bytes",
    "status",
    "result_hash",
    "cache_hit",
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarking.workloads import query_sql  # noqa: E402
from experience.store import ExperienceStore, content_hash  # noqa: E402
from optimization.actions import load_jsonl  # noqa: E402


@dataclass(frozen=True)
class Task:
    workload: str
    depth: int
    query_ids: tuple[str, ...]
    port: int

    @property
    def experiment_id(self) -> str:
        return f"decomposition-depth-{self.workload.lower()}-d{self.depth}-{VERSION}"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_pg_rows() -> dict[tuple[str, str], dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    with NQO_RUNS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workload = row["dataset"]
            if workload not in QUERY_SPECS or row["method"] != "PostgreSQL":
                continue
            query_id = Path(row["sql_path"]).stem
            if query_id in QUERY_SPECS[workload]:
                selected[(workload, query_id)] = row
    expected = {
        (workload, query_id)
        for workload, queries in QUERY_SPECS.items()
        for query_id in queries
    }
    missing = sorted(expected - set(selected))
    if missing:
        raise RuntimeError(f"missing benchmark PostgreSQL rows: {missing}")
    return selected


def build_baselines(
    pg_rows: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for workload, query_depths in QUERY_SPECS.items():
        payload = {}
        for query_id in query_depths:
            row = pg_rows[(workload, query_id)]
            runtime_ms = float(row["runtime_ms"])
            payload[query_id] = {
                "query_id": query_id,
                "median_charged_ms": runtime_ms,
                "official_charged_ms": runtime_ms,
                "official_client_wall_ms": runtime_ms,
                "official_repetition": 0,
                "result_hash": row["result_hash"] or None,
                "result_rows": int(row["result_rows"] or 0),
                "status": row["status"],
            }
        path = BASELINE_ROOT / f"{workload.lower()}.json"
        atomic_json(path, payload)
        paths[workload] = path
    return paths


def seed_cache(workload: str) -> Path:
    path = (
        PGDB_ROOT
        / ".nqo_runtime"
        / "online"
        / "experience"
        / f"decomposition-depth-{workload.lower()}-{VERSION}_raw_light.sql"
    )
    if path.is_file():
        return path
    source = ROOT / "results" / "buffers" / f"{workload.lower()}_light.sql"
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExperienceStore(source, read_only=True) as source_store:
        with ExperienceStore(path) as output_store:
            for item in source_store.iter_executions():
                if not item["trajectory"]:
                    continue
                output_store.append_execution(
                    cache_id=item["cache_id"],
                    created_at_ms=item["created_at_ms"],
                    query_id=item["query_id"],
                    sql_hash=item["sql_hash"],
                    action_config_hash=item["action_config_hash"],
                    trajectory=item["trajectory"],
                    db_events=item["db_events"],
                    status=item["status"],
                    first_runtime_ms=item["first_runtime_ms"],
                    charged_runtime_ms=item["charged_runtime_ms"],
                    timeout_limit_ms=item["timeout_limit_ms"],
                    result_hash=item["result_hash"],
                    result_rows=item["result_rows"],
                    source_episode_id=item["source_episode_id"],
                )
    return path


def build_tasks() -> list[Task]:
    tasks = []
    port = 24600
    for workload, query_depths in QUERY_SPECS.items():
        for depth in range(max(query_depths.values()) + 1):
            query_ids = tuple(
                query_id
                for query_id, max_depth in query_depths.items()
                if depth <= max_depth
            )
            tasks.append(Task(workload, depth, query_ids, port))
            port += 1
    return tasks


def run_task(
    task: Task,
    *,
    baseline: Path,
    experience_db: Path,
    sql_slots: int,
) -> Path:
    command = [
        sys.executable,
        "-m",
        ACTION_RUNNER_MODULE,
        "--workload",
        task.workload,
        "--profiles",
        "query_split",
        "--role",
        "all",
        "--warmups",
        "0",
        "--measurements",
        "1",
        "--experiment-id",
        task.experiment_id,
        "--output-root",
        str(RAW_ROOT),
        "--experience-db",
        str(experience_db),
        "--baseline-json",
        str(baseline),
        "--container",
        "pgdb_dev_opt",
        "--sql-execution-lock",
        str(LOCK_PATH),
        "--sql-execution-slots",
        str(sql_slots),
        "--ai-port",
        str(task.port),
        "--max-rounds",
        "16",
        "--dec-rounds",
        str(task.depth),
        "--sched-alpha",
        "0.5",
        "--execution-cache",
        "read-write",
        "--cache-match-mode",
        "statewise",
        "--correctness",
        "strict",
    ]
    for query_id in task.query_ids:
        command.extend(["--query-id", query_id])
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{task.experiment_id}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return log_path


def cache_events(
    store: ExperienceStore,
    workload: str,
    query_id: str,
    cache_id: str,
) -> list[dict[str, Any]]:
    sql_hash = content_hash(query_sql(workload, query_id))
    for candidate in store.trajectory_cache_candidates(
        sql_hash=sql_hash,
    ):
        if candidate["cache_id"] == cache_id:
            return list(candidate["db_events"])
    raise RuntimeError(f"missing cache payload {cache_id}")


def event_metrics(events: list[dict[str, Any]]) -> dict[str, float | int]:
    split_events = [event for event in events if event.get("phase") == "split"]
    final_events = [event for event in events if event.get("phase") == "final"]

    def timing(events_: list[dict[str, Any]], key: str) -> float:
        return sum(float((event.get("timing_ms") or {}).get(key) or 0.0) for event in events_)

    return {
        "actual_split_rounds": len(split_events),
        "executed_units": len(split_events) + len(final_events),
        "split_total_ms": timing(split_events, "total"),
        "final_total_ms": timing(final_events, "total"),
        "policy_ms": timing(events, "policy"),
        "planning_ms": timing(events, "planning"),
        "execution_ms": timing(events, "execution"),
        "materialized_rows": sum(
            int((event.get("materialized") or {}).get("rows") or 0)
            for event in split_events
        ),
        "materialized_bytes": sum(
            int((event.get("materialized") or {}).get("bytes") or 0)
            for event in split_events
        ),
    }


def collect_rows(
    tasks: list[Task],
    pg_rows: dict[tuple[str, str], dict[str, str]],
    experience_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (workload, query_id), row in sorted(pg_rows.items()):
        runtime_ms = float(row["runtime_ms"])
        rows.append(
            {
                "dataset": workload,
                "query_id": query_id,
                "sql_path": row["sql_path"],
                "method": "PostgreSQL",
                "requested_split_rounds": "",
                "actual_split_rounds": 0,
                "executed_units": 1,
                "runtime_ms": runtime_ms,
                "charged_runtime_ms": runtime_ms,
                "speedup": 1.0,
                "split_total_ms": 0.0,
                "final_total_ms": runtime_ms,
                "policy_ms": 0.0,
                "planning_ms": "",
                "execution_ms": "",
                "materialized_rows": 0,
                "materialized_bytes": 0,
                "status": row["status"],
                "result_hash": row["result_hash"],
                "cache_hit": "",
            }
        )

    stores = {
        workload: ExperienceStore(path, read_only=True)
        for workload, path in experience_paths.items()
    }
    try:
        for task in tasks:
            episode_path = RAW_ROOT / task.workload.lower() / task.experiment_id / "episodes.csv"
            with episode_path.open(newline="", encoding="utf-8") as handle:
                episodes = list(csv.DictReader(handle))
            for episode in episodes:
                query_id = episode["query_id"]
                cache_id = episode["cache_source"]
                if cache_id:
                    events = cache_events(stores[task.workload], task.workload, query_id, cache_id)
                else:
                    trace = (
                        PGDB_ROOT
                        / ".nqo_runtime"
                        / "online"
                        / task.experiment_id
                        / f"query_split.{query_id}.0.db.jsonl"
                    )
                    events = load_jsonl(trace)
                measured = event_metrics(events)
                pg_runtime = float(pg_rows[(task.workload, query_id)]["runtime_ms"])
                runtime_ms = float(episode["client_wall_ms"])
                charged_runtime_ms = float(episode["charged_wall_ms"])
                rows.append(
                    {
                        "dataset": task.workload,
                        "query_id": query_id,
                        "sql_path": pg_rows[(task.workload, query_id)]["sql_path"],
                        "method": "Fixed-depth",
                        "requested_split_rounds": task.depth,
                        "runtime_ms": runtime_ms,
                        "charged_runtime_ms": charged_runtime_ms,
                        "speedup": pg_runtime / charged_runtime_ms,
                        "status": episode["status"],
                        "result_hash": episode["result_hash"],
                        "cache_hit": episode["cache_hit"],
                        **measured,
                    }
                )
    finally:
        for store in stores.values():
            store.close()
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            row["query_id"],
            row["method"] != "PostgreSQL",
            -1 if row["requested_split_rounds"] == "" else int(row["requested_split_rounds"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sql-slots", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.sql_slots < 1:
        parser.error("worker and SQL-slot counts must be positive")

    pg_rows = load_pg_rows()
    baselines = build_baselines(pg_rows)
    experience_paths = {
        workload: seed_cache(workload) for workload in QUERY_SPECS
    }
    tasks = build_tasks()
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {
            pool.submit(
                run_task,
                task,
                baseline=baselines[task.workload],
                experience_db=experience_paths[task.workload],
                sql_slots=args.sql_slots,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            log_path = future.result()
            print(
                f"[done] {task.workload} depth={task.depth} "
                f"queries={len(task.query_ids)} log={log_path}",
                flush=True,
            )

    rows = collect_rows(tasks, pg_rows, experience_paths)
    write_csv(OUTPUT_CSV, rows)
    print(f"rows={len(rows)} output={OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
