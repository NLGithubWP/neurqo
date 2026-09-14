#!/usr/bin/env python3
"""Run versioned checkpoints for transferability experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PGDB_ROOT = REPO.parent / "pgdb"
DEFAULT_OUTPUT = (
    REPO
    / "results"
    / "benchmark"
    / "nqo"
    / "nqo_transfer_run.csv"
)
DEFAULT_RUNTIME = (
    PGDB_ROOT
    / ".nqo_runtime"
    / "reproduction"
    / "zero-shot-transfer"
)
MIXED_MODEL_ROOT = REPO / "results" / "models" / "mixed"
PG_SOURCE = REPO / "results" / "benchmark" / "nqo" / "nqo_runs.csv"
WORKLOADS = ("job", "stack", "tpch")
FOLDS = ("a", "b", "c")
ZERO_SHOT_CHECKPOINTS = {
    ("job", "stack", "b"): ("best", "0016"),
    ("job", "stack", "c"): ("best", "0000"),
}
TRANSFER_METHODS = {
    "zero-shot": "NQO zero-shot",
    "mixed": "NQO mixed",
}
CSV_FIELDS = (
    "target_dataset",
    "source_dataset",
    "fold",
    "sql_path",
    "method",
    "runtime_ms",
    "inference_ms",
    "status",
    "result_hash",
    "result_rows",
    "cache_hit",
    "cache_id",
    "trajectory_hash",
    "checkpoint",
    "actions_json",
)


os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(REPO))

from scripts.reproduce.nqo.run import (  # noqa: E402
    DATASETS,
    PGDB_ROOT as NQO_PGDB_ROOT,
    DockerLearnedPolicyServer,
    ExperienceStore,
    acquire_sql_execution_slot,
    actions_json,
    content_hash,
    execute_miss,
    load_query_sql,
    lookup_cached_execution,
    model_path,
    policy_inference_ms,
    query_relative_path,
    release_sql_execution_slot,
    repo_relative,
    semantic_policy_trajectory,
    semantic_trajectory_hash,
    split_folds,
    stage_catalog_snapshot,
    stage_model,
)


if NQO_PGDB_ROOT != PGDB_ROOT:
    raise RuntimeError("NQO runtime root mismatch")


@dataclass(frozen=True)
class Task:
    index: int
    experiment: str
    source: str
    target: str
    fold: str
    device: str
    port: int
    container: str
    runtime_root: Path
    cache_miss: str
    sql_execution_lock: Path
    sql_execution_slots: int
    host: str
    pg_port: int
    user: str
    server_action_host: str
    database_container: str
    checkpoint_label: str = "best"

    @property
    def label(self) -> str:
        suffix = (
            "" if self.checkpoint_label == "best" else f"-{self.checkpoint_label}"
        )
        return f"{self.source}-to-{self.target}-{self.fold}{suffix}"

    @property
    def method(self) -> str:
        return TRANSFER_METHODS[self.experiment]


def zero_shot_tasks(
    *,
    container: str,
    runtime_root: Path,
    base_port: int,
    devices: tuple[str, ...],
    cache_miss: str,
    sql_execution_lock: Path,
    sql_execution_slots: int,
    host: str,
    pg_port: int,
    user: str,
    server_action_host: str,
    database_container: str,
) -> list[Task]:
    tasks: list[Task] = []
    for source in WORKLOADS:
        for target in WORKLOADS:
            if source == target:
                continue
            for fold in FOLDS:
                checkpoint_labels = ZERO_SHOT_CHECKPOINTS.get(
                    (source, target, fold), ("best",)
                )
                for checkpoint_label in checkpoint_labels:
                    index = len(tasks)
                    tasks.append(
                        Task(
                            index=index,
                            experiment="zero-shot",
                            source=source,
                            target=target,
                            fold=fold,
                            device=devices[index % len(devices)],
                            port=base_port + index,
                            container=container,
                            runtime_root=runtime_root,
                            cache_miss=cache_miss,
                            sql_execution_lock=sql_execution_lock,
                            sql_execution_slots=sql_execution_slots,
                            host=host,
                            pg_port=pg_port,
                            user=user,
                            server_action_host=server_action_host,
                            database_container=database_container,
                            checkpoint_label=checkpoint_label,
                        )
                    )
    return tasks


def mixed_tasks(
    *,
    container: str,
    runtime_root: Path,
    base_port: int,
    devices: tuple[str, ...],
    cache_miss: str,
    sql_execution_lock: Path,
    sql_execution_slots: int,
    host: str,
    pg_port: int,
    user: str,
    server_action_host: str,
    database_container: str,
) -> list[Task]:
    tasks: list[Task] = []
    for target in WORKLOADS:
        for fold in FOLDS:
            index = len(tasks)
            tasks.append(
                Task(
                    index=index,
                    experiment="mixed",
                    source="mixed",
                    target=target,
                    fold=fold,
                    device=devices[index % len(devices)],
                    port=base_port + index,
                    container=container,
                    runtime_root=runtime_root,
                    cache_miss=cache_miss,
                    sql_execution_lock=sql_execution_lock,
                    sql_execution_slots=sql_execution_slots,
                    host=host,
                    pg_port=pg_port,
                    user=user,
                    server_action_host=server_action_host,
                    database_container=database_container,
                )
            )
    return tasks


def checkpoint_for_task(task: Task) -> Path:
    if task.experiment == "zero-shot":
        path = model_path(task.source, "random", task.fold)
        if task.checkpoint_label != "best":
            path = path.with_name(f"online-iter-{task.checkpoint_label}.pt")
        return path
    path = (
        MIXED_MODEL_ROOT
        / f"random_{task.fold}"
        / f"best-{task.target}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_postgres_rows() -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    if not PG_SOURCE.is_file():
        raise FileNotFoundError(PG_SOURCE)
    selected: list[dict[str, str]] = []
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    with PG_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "PostgreSQL":
                continue
            dataset = str(row["dataset"]).upper()
            sql_path = str(row["sql_path"])
            key = (dataset, sql_path)
            if key in indexed:
                raise RuntimeError(f"duplicate PostgreSQL row: {key}")
            indexed[key] = row
            selected.append(
                {
                    "target_dataset": dataset,
                    "source_dataset": "",
                    "fold": "",
                    "sql_path": sql_path,
                    "method": "PostgreSQL",
                    "runtime_ms": row["runtime_ms"],
                    "inference_ms": "0.000000000000",
                    "status": row["status"],
                    "result_hash": row["result_hash"],
                    "result_rows": row["result_rows"],
                    "cache_hit": "",
                    "cache_id": "",
                    "trajectory_hash": "",
                    "checkpoint": "",
                    "actions_json": "",
                }
            )
    return selected, indexed


def runtime_container_dir(runtime_dir: Path) -> str:
    relative = runtime_dir.resolve().relative_to(PGDB_ROOT.resolve())
    return f"/code/pgdb-dev/{relative.as_posix()}"


def container_bridge_ip(container: str) -> str:
    completed = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{(index .NetworkSettings.Networks \"bridge\").IPAddress}}",
            container,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    address = completed.stdout.strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", address):
        raise RuntimeError(f"cannot resolve bridge IP for {container}: {address!r}")
    return address


def verify_database_can_reach_server(task: Task, action_url: str) -> None:
    health_url = action_url.rsplit("/", 1)[0] + "/"
    completed = subprocess.run(
        [
            "docker",
            "exec",
            task.database_container,
            "curl",
            "-fsS",
            "--max-time",
            "5",
            health_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"database container cannot reach GPU policy server {health_url}: "
            f"{completed.stderr.strip()}"
        )


def evaluate_task(task: Task) -> dict[str, Any]:
    spec = DATASETS[task.target]
    checkpoint = checkpoint_for_task(task)
    runtime_dir = task.runtime_root / task.label
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "worker.log"
    rows: list[dict[str, str]] = []
    encountered_misses: list[str] = []
    unresolved_misses: list[str] = []
    physical_executions = 0
    _, pg_index = load_postgres_rows()

    with log_path.open("w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            staged, digest = stage_model(checkpoint, runtime_dir)
            container_dir = runtime_container_dir(runtime_dir)
            staged_container = f"{container_dir}/models/{staged.name}"
            _catalog_path, catalog_container_path = stage_catalog_snapshot(
                workload=spec.name,
                host=task.host,
                port=task.pg_port,
                user=task.user,
                database=None,
                runtime_dir=runtime_dir,
                runtime_container_dir=container_dir,
            )
            server_context = DockerLearnedPolicyServer(
                container=task.container,
                port=task.port,
                profile=spec.profile,
                runtime_host_dir=runtime_dir,
                runtime_container_dir=container_dir,
                run_label=f"{task.experiment}-{task.label}",
                model_container_path=staged_container,
                model_method="standardmdp_rl",
                model_hidden=128,
                workload=spec.name,
                catalog_container_path=catalog_container_path,
                model_device=task.device,
                nqo_src="/code/pgdb-dev/.nqo_runtime/nqo/src",
                inference_mode="deterministic",
                temperature=1.0,
                exploration_epsilon=0.0,
                coverage_counts_container_path=None,
                coverage_mix=0.0,
                coverage_power=0.5,
                stochastic_heads=None,
                sampling_seed=42,
                policy_version=f"released-{task.experiment}-{task.label}",
                action_ablation="none",
                listen_host="0.0.0.0",
                action_host=task.server_action_host,
            )
            fold_spec = split_folds(spec.name, "random")[f"random_{task.fold}"]
            query_ids = [str(value) for value in fold_spec["test"]]

            with ExperienceStore(
                spec.cache, writable=task.cache_miss == "execute"
            ) as store:
                store.set_action_config_hash(spec.action_config_hash)
                store.set_binding(
                    protocol="random",
                    fold=task.fold,
                    checkpoint_sha256=digest,
                )
                with server_context as server:
                    verify_database_can_reach_server(task, server.action_url)
                    policy_offset = 0
                    for query_id in query_ids:
                        sql_path = query_relative_path(spec.name, query_id)
                        sql = load_query_sql(spec.name, query_id)
                        sql_hash = content_hash(sql)
                        expected_hash = store.expected_result_hash(sql_hash)
                        cached, policy_offset = lookup_cached_execution(
                            store=store,
                            sql_hash=sql_hash,
                            expected_result_hash=expected_hash,
                            container=task.container,
                            server_url=server.action_url,
                            policy_log=server.policy_log_host,
                            policy_offset=policy_offset,
                            match_mode="prefix",
                        )
                        if cached is None:
                            encountered_misses.append(query_id)
                            if task.cache_miss == "error":
                                unresolved_misses.append(query_id)
                                rows.append(
                                    {
                                        "target_dataset": spec.name,
                                        "source_dataset": task.source.upper(),
                                        "fold": task.fold,
                                        "sql_path": sql_path,
                                        "method": task.method,
                                        "runtime_ms": "",
                                        "inference_ms": "",
                                        "status": "cache_miss",
                                        "result_hash": "",
                                        "result_rows": "",
                                        "cache_hit": "False",
                                        "cache_id": "",
                                        "trajectory_hash": "",
                                        "checkpoint": repo_relative(checkpoint),
                                        "actions_json": "",
                                    }
                                )
                                continue
                            lock_handle, _ = acquire_sql_execution_slot(
                                task.sql_execution_lock,
                                task.sql_execution_slots,
                            )
                            try:
                                (
                                    execution,
                                    policy_events,
                                    db_events,
                                    policy_offset,
                                    timeout_ms,
                                ) = execute_miss(
                                    args=SimpleNamespace(
                                        host=task.host,
                                        pg_port=task.pg_port,
                                        user=task.user,
                                        database=None,
                                    ),
                                    spec=spec,
                                    query_id=query_id,
                                    sql=sql,
                                    pg_first_ms=float(
                                        pg_index[(spec.name, sql_path)]["runtime_ms"]
                                    ),
                                    expected_result_hash=expected_hash,
                                    profile=spec.profile,
                                    server=server,
                                    policy_offset=policy_offset,
                                    runtime_dir=runtime_dir,
                                    runtime_container_dir=container_dir,
                                )
                            finally:
                                release_sql_execution_slot(lock_handle)
                            physical_executions += 1
                            trajectory = semantic_policy_trajectory(policy_events)
                            if not trajectory and query_id not in spec.fallback_queries:
                                raise RuntimeError(
                                    f"online cache miss produced no policy trajectory: "
                                    f"{spec.name}/{query_id}"
                                )
                            cache_id = ""
                            if execution["status"] in {"ok", "timeout"}:
                                cache_id, _ = store.append_execution(
                                    query_id=query_id,
                                    sql_hash=sql_hash,
                                    trajectory=trajectory,
                                    db_events=db_events,
                                    status=execution["status"],
                                    first_runtime_ms=execution["client_wall_ms"],
                                    charged_runtime_ms=execution["charged_wall_ms"],
                                    timeout_limit_ms=timeout_ms,
                                    action_config_hash=spec.action_config_hash,
                                    result_hash=execution.get("result_hash"),
                                    result_rows=execution.get("result_rows"),
                                )
                            else:
                                unresolved_misses.append(query_id)
                            cache_hit = False
                        else:
                            execution = cached["execution"]
                            policy_events = cached["policy_events"]
                            db_events = cached["db_events"]
                            cache_id = str(cached["cache_source"])
                            cache_hit = True
                        trajectory = semantic_policy_trajectory(policy_events)
                        if query_id in spec.fallback_queries:
                            if trajectory:
                                raise RuntimeError(
                                    f"fallback query produced Actions: "
                                    f"{spec.name}/{query_id}"
                                )
                            status = "fallback"
                            runtime_ms = ""
                            inference_ms = 0.0
                        else:
                            status = str(execution["status"])
                            runtime_ms = float(execution["charged_wall_ms"])
                            inference_ms = policy_inference_ms(
                                db_events, policy_events
                            )
                        rows.append(
                            {
                                "target_dataset": spec.name,
                                "source_dataset": task.source.upper(),
                                "fold": task.fold,
                                "sql_path": sql_path,
                                "method": task.method,
                                "runtime_ms": (
                                    "" if runtime_ms == "" else f"{runtime_ms:.12f}"
                                ),
                                "inference_ms": f"{inference_ms:.12f}",
                                "status": status,
                                "result_hash": execution.get("result_hash") or "",
                                "result_rows": (
                                    ""
                                    if execution.get("result_rows") is None
                                    else str(execution["result_rows"])
                                ),
                                "cache_hit": str(cache_hit),
                                "cache_id": cache_id,
                                "trajectory_hash": semantic_trajectory_hash(
                                    trajectory
                                ),
                                "checkpoint": repo_relative(checkpoint),
                                "actions_json": actions_json(trajectory),
                            }
                        )

    return {
        "task": task.label,
        "target": spec.name,
        "device": task.device,
        "port": task.port,
        "query_count": len(rows),
        "hits": len(rows) - len(encountered_misses),
        "cache_misses": encountered_misses,
        "physical_executions": physical_executions,
        "misses": unresolved_misses,
        "rows": rows,
        "log": repo_relative(log_path),
    }


def fill_fallback_runtimes(
    rows: list[dict[str, str]],
    pg_index: dict[tuple[str, str], dict[str, str]],
) -> None:
    for row in rows:
        if row["status"] != "fallback":
            continue
        key = (row["target_dataset"], row["sql_path"])
        pg = pg_index[key]
        row["runtime_ms"] = pg["runtime_ms"]
        row["result_hash"] = pg["result_hash"]
        row["result_rows"] = pg["result_rows"]


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_preserved_rows(
    path: Path,
    replaced_tasks: set[tuple[str, str, str, str, str]],
) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError(f"unexpected transfer CSV schema: {path}")
        return [
            row
            for row in reader
            if row["method"] != "PostgreSQL"
            and (
                row["method"],
                row["target_dataset"],
                row["source_dataset"],
                row["fold"],
                row["checkpoint"],
            )
            not in replaced_tasks
        ]


def parse_devices(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("--devices requires at least one device")
    for item in result:
        if item != "cpu" and not re.fullmatch(r"cuda:\d+", item):
            raise argparse.ArgumentTypeError(
                "devices must be cpu or cuda:<nonnegative ID>"
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transferability evaluation with versioned checkpoints"
    )
    parser.add_argument(
        "--experiment",
        choices=("zero-shot", "mixed", "all"),
        default="zero-shot",
    )
    parser.add_argument("--target", choices=WORKLOADS)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--container", default="pgdb_tpch_gpu")
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices(
            "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
        ),
    )
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--base-port", type=int, default=23100)
    parser.add_argument(
        "--cache-miss", choices=("error", "execute"), default="error"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--server-action-host")
    parser.add_argument("--database-container", default="pgdb_dev_opt")
    parser.add_argument(
        "--sql-execution-lock",
        type=Path,
        default=DEFAULT_RUNTIME / "sql-execution.lock",
    )
    parser.add_argument("--sql-execution-slots", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.runtime_root = args.runtime_root.resolve()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists; pass --overwrite: {args.output}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.sql_execution_slots < 1:
        parser.error("--sql-execution-slots must be positive")
    try:
        args.runtime_root.relative_to(PGDB_ROOT.resolve())
    except ValueError:
        parser.error(f"runtime root must be inside {PGDB_ROOT}")
    args.sql_execution_lock = args.sql_execution_lock.resolve()
    args.sql_execution_lock.parent.mkdir(parents=True, exist_ok=True)

    pg_rows, pg_index = load_postgres_rows()
    server_action_host = (
        args.server_action_host or container_bridge_ip(args.container)
    )
    task_args = {
        "container": args.container,
        "runtime_root": args.runtime_root,
        "devices": args.devices,
        "cache_miss": args.cache_miss,
        "sql_execution_lock": args.sql_execution_lock,
        "sql_execution_slots": args.sql_execution_slots,
        "host": args.host,
        "pg_port": args.pg_port,
        "user": args.user,
        "server_action_host": server_action_host,
        "database_container": args.database_container,
    }
    tasks: list[Task] = []
    if args.experiment in {"zero-shot", "all"}:
        tasks.extend(zero_shot_tasks(base_port=args.base_port, **task_args))
    if args.experiment in {"mixed", "all"}:
        mixed_base_port = args.base_port + (len(tasks) if tasks else 0)
        tasks.extend(mixed_tasks(base_port=mixed_base_port, **task_args))
    if args.target:
        tasks = [task for task in tasks if task.target == args.target]
    if args.fold:
        tasks = [task for task in tasks if task.fold == args.fold]
    if not tasks:
        parser.error("the experiment/target/fold selection produced no tasks")
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                print(f"[failed] {task.label}: {exc!r}", flush=True)
                raise
            completed.append(result)
            print(
                f"[done] {task.label} device={result['device']} "
                f"hits={result['hits']}/{result['query_count']} "
                f"cache_misses={len(result['cache_misses'])} "
                f"executed={result['physical_executions']} "
                f"unresolved={len(result['misses'])}",
                flush=True,
            )

    transfer_rows = [
        row
        for result in sorted(completed, key=lambda item: item["task"])
        for row in result["rows"]
    ]
    fill_fallback_runtimes(transfer_rows, pg_index)
    replaced_tasks = {
        (
            task.method,
            DATASETS[task.target].name,
            task.source.upper(),
            task.fold,
            repo_relative(checkpoint_for_task(task)),
        )
        for task in tasks
    }
    preserved_rows = load_preserved_rows(args.output, replaced_tasks)
    combined_transfer_rows = preserved_rows + transfer_rows
    pg_needed = {
        (row["target_dataset"], row["sql_path"])
        for row in combined_transfer_rows
    }
    selected_pg = [
        row
        for row in pg_rows
        if (row["target_dataset"], row["sql_path"]) in pg_needed
    ]
    rows = sorted(
        selected_pg + combined_transfer_rows,
        key=lambda row: (
            row["target_dataset"],
            row["method"],
            row["source_dataset"],
            row["fold"],
            row["checkpoint"],
            row["sql_path"],
        ),
    )
    write_csv_atomic(args.output, rows)

    misses = [
        f"{result['task']}:{query_id}"
        for result in completed
        for query_id in result["misses"]
    ]
    encountered_misses = sum(
        len(result["cache_misses"]) for result in completed
    )
    physical_executions = sum(
        int(result["physical_executions"]) for result in completed
    )
    summary = {
        "experiment": args.experiment,
        "tasks": len(completed),
        "workers": min(args.workers, len(tasks)),
        "devices": list(args.devices),
        "postgres_rows": len(selected_pg),
        "new_rows": len(transfer_rows),
        "preserved_rows": len(preserved_rows),
        "method_rows": dict(
            sorted(
                Counter(row["method"] for row in combined_transfer_rows).items()
            )
        ),
        "cache_hits": len(transfer_rows) - encountered_misses,
        "cache_misses": encountered_misses,
        "physical_executions": physical_executions,
        "unresolved_misses": len(misses),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if misses:
        print("Cache misses:")
        for item in misses:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
