#!/usr/bin/env python3
"""Replay versioned RL-formulation ablation checkpoints."""

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
MODEL_ROOT = REPO / "results" / "models" / "ablation"
LEGACY_ACTION_MODEL_DIR = {"no_dec": "no_split", "no_enum": "no_topk"}
REFERENCE_CSV = REPO / "results" / "benchmark" / "nqo" / "nqo_runs.csv"
OUTPUT_ROOT = REPO / "results" / "benchmark" / "nqo"
RUNTIME_ROOT = PGDB_ROOT / ".nqo_runtime" / "reproduction" / "ablation"
FOLDS = ("a", "b", "c")
CSV_FIELDS = (
    "dataset",
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
FAMILY_CONFIG = {
    "rl": {
        "variants": (("one_step", "One-step RL"),),
        "workloads": ("job", "stack"),
        "output": OUTPUT_ROOT / "nqo_abl_rl_run.csv",
        "runtime": RUNTIME_ROOT / "rl",
        "base_port": 24100,
        "workers": 6,
    },
    "state": {
        "variants": (
            ("no_query_topology", "w/o Query Topology"),
            ("no_plan_topology", "w/o Plan Topology"),
        ),
        "workloads": ("job", "stack", "tpch"),
        # Keep the requested release filename, including its spelling.
        "output": OUTPUT_ROOT / "nqo_abl_sate_run.csv",
        "runtime": RUNTIME_ROOT / "state",
        "base_port": 24200,
        "workers": 18,
    },
    "action": {
        "variants": (
            ("no_dec", "w/o Dec"),
            ("no_enum", "w/o Enum"),
            ("no_filter", "w/o Filter"),
            ("no_ajoin", "w/o AJoin"),
        ),
        "workloads": ("job", "stack", "tpch"),
        "output": OUTPUT_ROOT / "nqo_abl_action_run.csv",
        "runtime": RUNTIME_ROOT / "action",
        "base_port": 24300,
        "workers": 33,
    },
}


os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(REPO))

from scripts.reproduce.nqo.run import (  # noqa: E402
    DATASETS,
    DockerLearnedPolicyServer,
    ExperienceStore,
    acquire_sql_execution_slot,
    actions_json,
    content_hash,
    execute_miss,
    load_query_sql,
    lookup_cached_execution,
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
from optimization.decomposition_eligibility import (  # noqa: E402
    workload_supports_decomposition,
)


@dataclass(frozen=True)
class Task:
    family: str
    variant: str
    method: str
    workload: str
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

    @property
    def label(self) -> str:
        return f"{self.family}-{self.variant}-{self.workload}-{self.fold}"

    @property
    def checkpoint(self) -> Path:
        variant_dir = (
            LEGACY_ACTION_MODEL_DIR.get(self.variant, self.variant)
            if self.family == "action"
            else self.variant
        )
        return (
            MODEL_ROOT
            / self.family
            / variant_dir
            / self.workload
            / f"random_{self.fold}"
            / "best.pt"
        )

    @property
    def action_ablation(self) -> str:
        return self.variant if self.family == "action" else "none"


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    if any(
        item != "cpu" and re.fullmatch(r"cuda:\d+", item) is None
        for item in devices
    ):
        raise argparse.ArgumentTypeError("devices must be cpu or cuda:<id>")
    return devices


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
        capture_output=True,
        text=True,
    )
    address = completed.stdout.strip()
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", address) is None:
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
            f"database cannot reach {health_url}: {completed.stderr.strip()}"
        )


def runtime_container_dir(runtime_dir: Path) -> str:
    relative = runtime_dir.resolve().relative_to(PGDB_ROOT.resolve())
    return f"/code/pgdb-dev/{relative.as_posix()}"


def load_reference_rows(workloads: tuple[str, ...]) -> tuple[
    list[dict[str, str]], dict[tuple[str, str], dict[str, str]]
]:
    selected: list[dict[str, str]] = []
    pg_index: dict[tuple[str, str], dict[str, str]] = {}
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"].lower()
            if dataset not in workloads:
                continue
            if row["method"] == "PostgreSQL":
                pg_index[(row["dataset"], row["sql_path"])] = row
                method = "PostgreSQL"
            elif row["method"] == "NQO" and row["protocol"] == "random":
                method = "Full NQO"
            else:
                continue
            selected.append(
                {
                    "dataset": row["dataset"],
                    "fold": row["fold"],
                    "sql_path": row["sql_path"],
                    "method": method,
                    "runtime_ms": row["runtime_ms"],
                    "inference_ms": row["inference_ms"],
                    "status": row["status"],
                    "result_hash": row["result_hash"],
                    "result_rows": row["result_rows"],
                    "cache_hit": "",
                    "cache_id": row["cache_id"],
                    "trajectory_hash": row["trajectory_hash"],
                    "checkpoint": row["checkpoint"],
                    "actions_json": row["actions_json"],
                }
            )
    full = [row for row in selected if row["method"] == "Full NQO"]
    full_keys = [(row["dataset"], row["sql_path"]) for row in full]
    if not pg_index:
        raise RuntimeError("missing PostgreSQL rows")
    if len(full_keys) != len(set(full_keys)):
        raise RuntimeError("duplicate Full NQO query rows")
    if set(full_keys) != set(pg_index):
        raise RuntimeError("PostgreSQL and Full NQO query sets differ")
    return selected, pg_index


def evaluate_task(task: Task) -> dict[str, Any]:
    spec = DATASETS[task.workload]
    protocol = getattr(task, "protocol", "random")
    runtime_dir = task.runtime_root / task.label
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "worker.log"
    rows: list[dict[str, str]] = []
    encountered_misses: list[str] = []
    unresolved_misses: list[str] = []
    physical_executions = 0
    _, pg_index = load_reference_rows((task.workload,))

    with log_path.open("w", encoding="utf-8") as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            staged, digest = stage_model(task.checkpoint, runtime_dir)
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
                run_label=f"released-{task.label}",
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
                policy_version=f"released-{task.label}",
                action_ablation=task.action_ablation,
                listen_host="0.0.0.0",
                action_host=task.server_action_host,
                startup_timeout=60.0,
            )
            fold_spec = split_folds(spec.name, protocol)[
                f"{protocol}_{task.fold}"
            ]
            query_ids = [str(value) for value in fold_spec["test"]]

            with ExperienceStore(
                spec.cache, writable=task.cache_miss == "execute"
            ) as store:
                store.set_action_config_hash(spec.action_config_hash)
                store.set_binding(
                    protocol=protocol,
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
                            if (
                                not trajectory
                                and query_id not in spec.fallback_queries
                            ):
                                raise RuntimeError(
                                    f"online miss produced no policy trajectory: "
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
                            pg_row = pg_index[(spec.name, sql_path)]
                            runtime_ms = float(pg_row["runtime_ms"])
                            inference_ms = 0.0
                            status = "fallback"
                            result_hash = pg_row["result_hash"]
                            result_rows = pg_row["result_rows"]
                        else:
                            runtime_ms = float(execution["charged_wall_ms"])
                            inference_ms = policy_inference_ms(
                                db_events, policy_events
                            )
                            status = str(execution["status"])
                            result_hash = execution.get("result_hash") or ""
                            result_rows = (
                                ""
                                if execution.get("result_rows") is None
                                else str(execution["result_rows"])
                            )
                        rows.append(
                            {
                                "dataset": spec.name,
                                "fold": task.fold,
                                "sql_path": sql_path,
                                "method": task.method,
                                "runtime_ms": f"{runtime_ms:.12f}",
                                "inference_ms": f"{inference_ms:.12f}",
                                "status": status,
                                "result_hash": result_hash,
                                "result_rows": result_rows,
                                "cache_hit": str(cache_hit),
                                "cache_id": cache_id,
                                "trajectory_hash": semantic_trajectory_hash(trajectory),
                                "checkpoint": repo_relative(task.checkpoint),
                                "actions_json": actions_json(trajectory),
                            }
                        )

    return {
        "task": task.label,
        "device": task.device,
        "rows": rows,
        "query_count": len(rows),
        "cache_misses": encountered_misses,
        "physical_executions": physical_executions,
        "unresolved_misses": unresolved_misses,
        "log": repo_relative(log_path),
    }


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_tasks(args: argparse.Namespace, family: str) -> list[Task]:
    tasks: list[Task] = []
    config = FAMILY_CONFIG[family]
    server_action_host = args.server_action_host or container_bridge_ip(args.container)
    for variant, method in config["variants"]:
        for workload in args.workloads:
            if variant == "no_dec" and not workload_supports_decomposition(workload):
                continue
            for fold in FOLDS:
                index = len(tasks)
                tasks.append(
                    Task(
                        family=family,
                        variant=variant,
                        method=method,
                        workload=workload,
                        fold=fold,
                        device=args.devices[index % len(args.devices)],
                        port=args.base_port + index,
                        container=args.container,
                        runtime_root=args.runtime_root,
                        cache_miss=args.cache_miss,
                        sql_execution_lock=args.sql_execution_lock,
                        sql_execution_slots=args.sql_execution_slots,
                        host=args.host,
                        pg_port=args.pg_port,
                        user=args.user,
                        server_action_host=server_action_host,
                        database_container=args.database_container,
                    )
                )
    return tasks


def run_family(family: str) -> int:
    config = FAMILY_CONFIG[family]
    parser = argparse.ArgumentParser(
        description=f"Replay versioned {family} ablation checkpoints"
    )
    parser.add_argument("--output", type=Path, default=config["output"])
    parser.add_argument("--runtime-root", type=Path, default=config["runtime"])
    parser.add_argument("--container", default="pgdb_tpch_gpu")
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices(
            "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
        ),
    )
    parser.add_argument("--workers", type=int, default=config["workers"])
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=tuple(DATASETS),
        default=config["workloads"],
    )
    parser.add_argument("--base-port", type=int, default=config["base_port"])
    parser.add_argument("--cache-miss", choices=("error", "execute"), default="error")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--server-action-host")
    parser.add_argument("--database-container", default="pgdb_dev_opt")
    parser.add_argument(
        "--sql-execution-lock",
        type=Path,
        default=RUNTIME_ROOT / "sql-execution.lock",
    )
    parser.add_argument("--sql-execution-slots", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    args.output = args.output.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.sql_execution_lock = args.sql_execution_lock.resolve()
    args.sql_execution_lock.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.merge_existing:
        parser.error("--overwrite and --merge-existing are mutually exclusive")
    if args.output.exists() and not (args.overwrite or args.merge_existing):
        parser.error(f"output exists; pass --overwrite: {args.output}")
    if args.merge_existing and not args.output.exists():
        parser.error(f"cannot merge missing output: {args.output}")
    if args.workers < 1 or args.sql_execution_slots < 1:
        parser.error("worker and SQL-slot counts must be positive")
    try:
        args.runtime_root.relative_to(PGDB_ROOT.resolve())
    except ValueError:
        parser.error(f"runtime root must be inside {PGDB_ROOT}")

    reference_rows, _ = load_reference_rows(tuple(args.workloads))
    tasks = build_tasks(args, family)
    for task in tasks:
        if not task.checkpoint.is_file():
            raise FileNotFoundError(task.checkpoint)

    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed.append(result)
            print(
                f"[done] {task.label} device={result['device']} "
                f"rows={result['query_count']} "
                f"misses={len(result['cache_misses'])} "
                f"executed={result['physical_executions']}",
                flush=True,
            )

    ablation_rows = [
        row
        for result in sorted(completed, key=lambda item: item["task"])
        for row in result["rows"]
    ]
    existing_rows: list[dict[str, str]] = []
    if args.merge_existing:
        with args.output.open(newline="", encoding="utf-8") as handle:
            existing_rows = [
                row
                for row in csv.DictReader(handle)
                if row["dataset"].lower() not in args.workloads
            ]
    rows = sorted(
        existing_rows + reference_rows + ablation_rows,
        key=lambda row: (
            row["dataset"],
            row["method"],
            row["fold"],
            row["sql_path"],
        ),
    )
    write_csv_atomic(args.output, rows)
    unresolved = [
        f"{result['task']}:{query_id}"
        for result in completed
        for query_id in result["unresolved_misses"]
    ]
    summary = {
        "family": family,
        "tasks": len(tasks),
        "workers": min(args.workers, len(tasks)),
        "devices": list(args.devices),
        "reference_rows": len(reference_rows),
        "ablation_rows": len(ablation_rows),
        "method_rows": dict(Counter(row["method"] for row in rows)),
        "cache_hits": sum(row["cache_hit"] == "True" for row in ablation_rows),
        "cache_misses": sum(
            len(result["cache_misses"]) for result in completed
        ),
        "physical_executions": sum(
            result["physical_executions"] for result in completed
        ),
        "unresolved_misses": len(unresolved),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if unresolved:
        print("Unresolved cache misses:")
        for item in unresolved:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_family("rl"))
