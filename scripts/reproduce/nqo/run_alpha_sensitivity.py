#!/usr/bin/env python3
"""Evaluate fixed and learned Sched-head alpha policies on JOB Random."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PGDB_ROOT = REPO.parent / "pgdb"
DEFAULT_INPUT = REPO / "results" / "benchmark" / "nqo" / "nqo_runs.csv"
DEFAULT_OUTPUT = (
    REPO / "results" / "benchmark" / "nqo" / "nqo_alpha_sensitivity.csv"
)
DEFAULT_RUNTIME_DIR = (
    PGDB_ROOT / ".nqo_runtime" / "reproduction" / "alpha-sensitivity-job-random"
)
FIELDS = (
    "dataset",
    "protocol",
    "fold",
    "sql_path",
    "schedule_policy",
    "fixed_alpha",
    "runtime_ms",
    "inference_ms",
    "pg_runtime_ms",
    "status",
    "cache_hit",
    "result_hash",
    "result_rows",
    "cache_id",
    "trajectory_hash",
    "checkpoint",
    "actions_json",
)
FIXED_POLICIES = {
    "fixed_0": 0.0,
    "fixed_0_5": 0.5,
    "fixed_1": 1.0,
}

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.reproduce.nqo import run as nqo_runner  # noqa: E402
from benchmarking.action_runner import (  # noqa: E402
    acquire_sql_execution_slot,
    release_sql_execution_slot,
)
from benchmarking.execution_cache import lookup_cached_execution  # noqa: E402
from benchmarking.policy_server import DockerLearnedPolicyServer  # noqa: E402
from experience.store import (  # noqa: E402
    content_hash,
    semantic_trajectory_hash,
)
from optimization.actions import (  # noqa: E402
    semantic_policy_trajectory,
)


class ResultCsv:
    def __init__(self, path: Path, *, overwrite: bool) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.thread_lock = threading.Lock()
        if overwrite:
            path.unlink(missing_ok=True)
        self.rows: dict[tuple[str, str, str], dict[str, str]] = {}
        if path.is_file() and path.stat().st_size:
            self._load()

    @staticmethod
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["schedule_policy"]),
            str(row["fold"]),
            str(row["sql_path"]),
        )

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise RuntimeError(
                    f"{self.path} is missing columns: {sorted(missing)}"
                )
            for row in reader:
                key = self.key(row)
                if key in self.rows:
                    raise RuntimeError(f"duplicate alpha result: {key}")
                self.rows[key] = row

    def contains(self, policy: str, fold: str, sql_path: str) -> bool:
        return (policy, fold, sql_path) in self.rows

    def put(self, row: dict[str, Any]) -> bool:
        normalized = {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in FIELDS
        }
        key = self.key(normalized)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.thread_lock, self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if key in self.rows:
                return False
            write_header = not self.path.is_file() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=FIELDS, lineterminator="\n"
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(normalized)
                handle.flush()
                os.fsync(handle.fileno())
            self.rows[key] = normalized
        return True

    def sort(self) -> None:
        policy_order = {"fixed_0": 0, "fixed_0_5": 1, "fixed_1": 2, "learned": 3}
        ordered = sorted(
            self.rows.values(),
            key=lambda row: (
                policy_order.get(row["schedule_policy"], 99),
                row["fold"],
                row["sql_path"],
            ),
        )
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(ordered)
        temporary.replace(self.path)


def load_benchmark_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pg = {
        row["sql_path"]: float(row["runtime_ms"])
        for row in rows
        if row["dataset"] == "JOB" and row["method"] == "PostgreSQL"
    }
    learned = [
        row
        for row in rows
        if row["dataset"] == "JOB"
        and row["protocol"] == "random"
        and row["method"] == "NQO"
    ]
    if not pg or not learned:
        raise RuntimeError(f"missing JOB benchmark rows in {path}")
    return learned, pg


def add_learned_rows(
    output: ResultCsv,
    learned: list[dict[str, str]],
    pg: dict[str, float],
) -> None:
    for row in learned:
        output.put(
            {
                "dataset": "JOB",
                "protocol": "random",
                "fold": row["fold"],
                "sql_path": row["sql_path"],
                "schedule_policy": "learned",
                "fixed_alpha": "",
                "runtime_ms": row["runtime_ms"],
                "inference_ms": row["inference_ms"],
                "pg_runtime_ms": f"{pg[row['sql_path']]:.12f}",
                "status": row["status"],
                "cache_hit": "1",
                "result_hash": row["result_hash"],
                "result_rows": row["result_rows"],
                "cache_id": row["cache_id"],
                "trajectory_hash": row["trajectory_hash"],
                "checkpoint": row["checkpoint"],
                "actions_json": row["actions_json"],
            }
        )


def assert_fixed_alpha(trajectory: list[dict[str, Any]], alpha: float) -> None:
    for decision in trajectory:
        if decision["phase"] != "sched":
            continue
        actual = float(decision["action"]["sched_alpha"])
        if abs(actual - alpha) > 1e-9:
            raise RuntimeError(
                f"Sched override failed: expected {alpha}, got {actual}"
            )


def evaluate_fold(
    *,
    args: argparse.Namespace,
    output: ResultCsv,
    pg: dict[str, float],
    policy: str,
    alpha: float,
    fold: str,
    port: int,
) -> dict[str, int]:
    spec = nqo_runner.DATASETS["job"]
    fold_spec = nqo_runner.split_folds("JOB", "random")[f"random_{fold}"]
    query_ids = [str(value) for value in fold_spec["test"]]
    pending = [
        query_id
        for query_id in query_ids
        if not output.contains(
            policy, fold, nqo_runner.query_relative_path("JOB", query_id)
        )
    ]
    if not pending:
        return {"hits": 0, "misses": 0, "executed": 0, "skipped": len(query_ids)}

    checkpoint = nqo_runner.model_path("job", "random", fold)
    runtime_dir = args.runtime_dir / policy / fold
    runtime_dir.mkdir(parents=True, exist_ok=True)
    staged, digest = nqo_runner.stage_model(checkpoint, runtime_dir)
    runtime_relative = runtime_dir.resolve().relative_to(PGDB_ROOT)
    runtime_container_dir = f"/code/pgdb-dev/{runtime_relative.as_posix()}"
    staged_container = f"{runtime_container_dir}/models/{staged.name}"
    _catalog_path, catalog_container_path = nqo_runner.stage_catalog_snapshot(
        workload="JOB",
        host=args.host,
        port=args.pg_port,
        user=args.user,
        database=None,
        runtime_dir=runtime_dir,
        runtime_container_dir=runtime_container_dir,
    )
    counters = {"hits": 0, "misses": 0, "executed": 0, "skipped": 0}

    with nqo_runner.ExperienceStore(
        spec.cache, writable=args.cache_miss == "execute"
    ) as store:
        store.set_binding(
            protocol=f"alpha_sensitivity_{policy}",
            fold=fold,
            checkpoint_sha256=digest,
        )
        store.set_action_config_hash(spec.action_config_hash)
        server_context = DockerLearnedPolicyServer(
            container=args.container,
            port=port,
            profile=spec.profile,
            runtime_host_dir=runtime_dir,
            runtime_container_dir=runtime_container_dir,
            run_label=f"alpha-sensitivity-job-random-{policy}-{fold}",
            listen_host="0.0.0.0",
            action_host=args.action_host,
            model_container_path=staged_container,
            model_method="standardmdp_rl",
            model_hidden=128,
            workload="JOB",
            catalog_container_path=catalog_container_path,
            model_device=args.model_device,
            nqo_src="/code/pgdb-dev/.nqo_runtime/nqo/src",
            inference_mode="deterministic",
            temperature=1.0,
            exploration_epsilon=0.0,
            coverage_counts_container_path=None,
            coverage_mix=0.0,
            coverage_power=0.5,
            stochastic_heads=None,
            sampling_seed=42,
            policy_version=f"alpha-sensitivity-{policy}-{fold}",
            action_ablation="none",
            fixed_sched_alpha=alpha,
        )
        with server_context as server:
            policy_offset = 0
            for index, query_id in enumerate(pending, start=1):
                sql_path = nqo_runner.query_relative_path("JOB", query_id)
                sql = nqo_runner.load_query_sql("JOB", query_id)
                sql_hash = content_hash(sql)
                expected_hash = store.expected_result_hash(sql_hash)
                cached, policy_offset = lookup_cached_execution(
                    store=store,
                    sql_hash=sql_hash,
                    expected_result_hash=expected_hash,
                    container=args.container,
                    server_url=server.action_url,
                    policy_log=server.policy_log_host,
                    policy_offset=policy_offset,
                    match_mode="prefix",
                )
                cache_hit = cached is not None
                if cached is None:
                    counters["misses"] += 1
                    print(
                        f"[{policy}/{fold} {index}/{len(pending)}] "
                        f"{query_id}: cache miss",
                        flush=True,
                    )
                    if args.cache_miss == "error":
                        continue
                    sql_lock, slot = acquire_sql_execution_slot(
                        args.sql_execution_lock, args.sql_execution_slots
                    )
                    print(
                        f"[{policy}/{fold}] {query_id}: SQL slot "
                        f"{slot + 1}/{args.sql_execution_slots}",
                        flush=True,
                    )
                    try:
                        (
                            execution,
                            policy_events,
                            db_events,
                            policy_offset,
                            timeout_ms,
                        ) = nqo_runner.execute_miss(
                            args=args,
                            spec=spec,
                            query_id=query_id,
                            sql=sql,
                            pg_first_ms=pg[sql_path],
                            expected_result_hash=expected_hash,
                            profile=spec.profile,
                            server=server,
                            policy_offset=policy_offset,
                            runtime_dir=runtime_dir,
                            runtime_container_dir=runtime_container_dir,
                        )
                    finally:
                        release_sql_execution_slot(sql_lock)
                    trajectory = semantic_policy_trajectory(policy_events)
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
                    counters["executed"] += 1
                else:
                    counters["hits"] += 1
                    execution = cached["execution"]
                    policy_events = cached["policy_events"]
                    db_events = cached["db_events"]
                    cache_id = str(cached["cache_source"])
                    trajectory = semantic_policy_trajectory(policy_events)

                assert_fixed_alpha(trajectory, alpha)
                output.put(
                    {
                        "dataset": "JOB",
                        "protocol": "random",
                        "fold": fold,
                        "sql_path": sql_path,
                        "schedule_policy": policy,
                        "fixed_alpha": f"{alpha:.1f}",
                        "runtime_ms": f"{float(execution['charged_wall_ms']):.12f}",
                        "inference_ms": (
                            f"{nqo_runner.policy_inference_ms(db_events, policy_events):.12f}"
                        ),
                        "pg_runtime_ms": f"{pg[sql_path]:.12f}",
                        "status": execution["status"],
                        "cache_hit": "1" if cache_hit else "0",
                        "result_hash": execution.get("result_hash") or "",
                        "result_rows": (
                            ""
                            if execution.get("result_rows") is None
                            else execution["result_rows"]
                        ),
                        "cache_id": cache_id,
                        "trajectory_hash": semantic_trajectory_hash(trajectory),
                        "checkpoint": nqo_runner.repo_relative(checkpoint),
                        "actions_json": nqo_runner.actions_json(trajectory),
                    }
                )
                print(
                    f"[{policy}/{fold} {index}/{len(pending)}] {query_id}: "
                    f"{'hit' if cache_hit else 'executed'}, "
                    f"runtime_ms={float(execution['charged_wall_ms']):.3f}",
                    flush=True,
                )
    return counters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache-miss", choices=("error", "execute"), default="error"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--container", default="pgdb_tpch_gpu")
    parser.add_argument("--action-host")
    parser.add_argument("--server-port-base", type=int, default=18420)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument(
        "--sql-execution-lock",
        type=Path,
        default=PGDB_ROOT / ".nqo_runtime" / "reproduction" / ".alpha-sql.lock",
    )
    parser.add_argument("--sql-execution-slots", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.sql_execution_slots < 1:
        parser.error("worker and SQL slot counts must be positive")
    if args.action_host is None:
        args.action_host = subprocess.check_output(
            [
                "docker",
                "inspect",
                args.container,
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            text=True,
        ).strip()
        if not args.action_host:
            raise RuntimeError(f"cannot resolve container IP for {args.container}")

    learned, pg = load_benchmark_rows(args.input)
    output = ResultCsv(args.output, overwrite=args.overwrite)
    add_learned_rows(output, learned, pg)

    totals: dict[str, dict[str, int]] = {}
    for policy, alpha in FIXED_POLICIES.items():
        with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
            futures = {
                pool.submit(
                    evaluate_fold,
                    args=args,
                    output=output,
                    pg=pg,
                    policy=policy,
                    alpha=alpha,
                    fold=fold,
                    port=args.server_port_base + index,
                ): fold
                for index, fold in enumerate(("a", "b", "c"))
            }
            for future in as_completed(futures):
                fold = futures[future]
                totals[f"{policy}/{fold}"] = future.result()
    output.sort()
    print(json.dumps(totals, indent=2, sort_keys=True))
    print(f"wrote {len(output.rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
