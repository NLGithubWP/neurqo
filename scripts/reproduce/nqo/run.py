#!/usr/bin/env python3
"""Run PostgreSQL, a versioned NQO model, or a fixed standalone Action."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg2


REPO = Path(__file__).resolve().parents[3]
PGDB_ROOT = REPO.parent / "pgdb"
MODEL_ROOT = REPO / "results" / "models"
BUFFER_ROOT = REPO / "results" / "buffers"
DEFAULT_OUTPUT = (
    REPO / "results" / "benchmark" / "nqo" / "nqo_runs.csv"
)
DEFAULT_RUNTIME_DIR = (
    PGDB_ROOT / ".nqo_runtime" / "reproduction" / "nqo-evaluator"
)
CSV_FIELDS = (
    "dataset",
    "protocol",
    "fold",
    "sql_path",
    "method",
    "runtime_ms",
    "inference_ms",
    "status",
    "result_hash",
    "result_rows",
    "cache_id",
    "trajectory_hash",
    "checkpoint",
    "actions_json",
)
PROTOCOL_DIRS = {
    "base_query": "base-query",
    "leave_one_out": "leave-one-out",
    "random": "random",
}
SQL_DIRS = {
    "JOB": REPO / "workloads" / "query_job",
    "STACK": REPO / "workloads" / "query_stack",
    "TPCH": REPO / "workloads" / "query_tpch",
}

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarking.workloads import (  # noqa: E402
    WORKLOAD_DATABASES,
    split_folds,
    workload_query_ids,
)
from benchmarking.action_runner import (  # noqa: E402
    acquire_sql_execution_slot,
    release_sql_execution_slot,
)
from benchmarking.run_environment import stage_policy_runtime  # noqa: E402
from benchmarking.execution_cache import (  # noqa: E402
    lookup_cached_execution,
    read_jsonl_since,
)
from benchmarking.policy_server import (  # noqa: E402
    DockerFixedPolicyServer,
    DockerLearnedPolicyServer,
)
from experience.store import (  # noqa: E402
    ExperienceStore,
    canonical_json,
    content_hash,
    semantic_trajectory_hash,
)
from database.catalog import (  # noqa: E402
    read_postgres_catalog,
    write_catalog_snapshot,
)
from optimization.actions import (  # noqa: E402
    ActionProfile,
    first_runtime_timeout_ms,
    hash_result_rows,
    load_jsonl,
    semantic_policy_trajectory,
    timeout_charged_runtime_ms,
    validate_policy_state_contract,
)


ACTION_CONFIG = Path(__file__).with_name("action_config.json")
ACTION_PROFILE_CATALOG = json.loads(ACTION_CONFIG.read_text())
METHOD_LABELS = dict(ACTION_PROFILE_CATALOG["method_labels"])
STANDALONE_METHODS = tuple(METHOD_LABELS)
DATASET_METHODS = {
    dataset: tuple(config["methods"])
    for dataset, config in ACTION_PROFILE_CATALOG["datasets"].items()
}


def profile_for(dataset: str, method: str) -> ActionProfile:
    dataset = dataset.upper()
    if method not in DATASET_METHODS[dataset]:
        raise ValueError(f"{dataset} has no benchmark result for {method}")
    values = dict(
        ACTION_PROFILE_CATALOG["datasets"][dataset]["base_profile"]
    )
    values.update(ACTION_PROFILE_CATALOG["method_overrides"][method])
    return ActionProfile.from_mapping(name=method, values=values)


def representative_actions(profile: ActionProfile) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = [
        {
            "phase": "dec",
            "action": {
                "dec_action": profile.dec,
                "order_decision": "only_cost",
            },
        }
    ]
    if profile.dec == "apply":
        actions.append(
            {
                "phase": "sched",
                "action": {"sched_alpha": profile.sched_alpha},
            }
        )
    actions.extend(
        [
            {
                "phase": "enum",
                "action": {
                    "enum_action": profile.enum,
                    "enum_k": profile.enum_k if profile.enum != "native" else 1,
                },
            },
            {
                "phase": "adapt",
                "action": {
                    "ajoin_action": profile.ajoin,
                    "filter_action": profile.filter,
                },
            },
        ]
    )
    return actions


JOB_PROFILE = ActionProfile(
    name="learned",
    max_rounds=16,
    search_max_rels=4,
    search_exact_cardinality=False,
    aja_conservative_rows=100_000,
    aja_aggressive_rows=100_000,
    aja_max_nestloop_cost_ratio_pct=150,
    aja_aggressive_max_nestloop_cost_ratio_pct=125,
    lip_max_build_relation_rows=500_000,
    lip_selective_plan_rows=10_000,
    lip_max_build_selectivity_pct=5,
    lip_min_probe_ratio=2,
    lip_max_filters=4,
)

TPCH_PROFILE = ActionProfile(
    name="learned",
    max_rounds=16,
    search_max_rels=8,
    search_exact_cardinality=False,
    aja_conservative_rows=100_000,
    aja_aggressive_rows=10_000,
    aja_max_nestloop_cost_ratio_pct=150,
    aja_aggressive_max_nestloop_cost_ratio_pct=125,
    lip_max_build_relation_rows=100_000,
    lip_selective_plan_rows=10_000,
    lip_max_build_selectivity_pct=5,
    lip_min_probe_ratio=2,
    lip_max_filters=4,
)

STACK_PROFILE = ActionProfile(
    name="learned",
    max_rounds=16,
    search_max_rels=12,
    search_exact_cardinality=False,
    aja_conservative_rows=1_000,
    aja_aggressive_rows=3_624_434,
    aja_max_nestloop_cost_ratio_pct=150,
    aja_aggressive_max_nestloop_cost_ratio_pct=125,
    lip_max_build_relation_rows=10_000,
    lip_selective_plan_rows=10_000,
    lip_max_build_selectivity_pct=1,
    lip_min_probe_ratio=2,
    lip_max_filters=4,
)

POSTGRES_PROFILE = ActionProfile(name="pg", nqo_enabled=False)
POSTGRES_TIMEOUT_MS = 60_000
NQO_ACTION_CONFIG_HASHES = {
    "JOB": "9642f7cae030aac1e9f3d84519e19a41c8032645100b99942323e97fe234c665",
    "STACK": "ce4bc2e8dc5dee52150e62a5fee5dbdb9c90b439518ebfbcbe821045e4e312ad",
    "TPCH": "5945cde270af1c9297551845b6e1479b12db9070bb59c60de43d07fa233de4f5",
}


def connect_postgres(args: argparse.Namespace):
    connection = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        dbname=args.database or WORKLOAD_DATABASES[args.workload],
    )
    connection.autocommit = True
    return connection


def set_session_config(cursor: Any, name: str, value: Any) -> None:
    cursor.execute("SELECT set_config(%s, %s, false)", (name, str(value)))


def execute_sql_once(
    args: argparse.Namespace,
    *,
    sql: str,
    profile: ActionProfile,
    timeout_ms: int,
    db_trace_container: str,
    server_url: str | None,
) -> dict[str, Any]:
    connection = connect_postgres(args)
    cursor = connection.cursor()
    started = None
    status = "ok"
    error = None
    result_hash = None
    result_rows = None
    try:
        set_session_config(cursor, "statement_timeout", timeout_ms)
        set_session_config(cursor, "client_min_messages", "warning")
        set_session_config(cursor, "search_path", "public")
        if profile.nqo_enabled:
            if server_url is None:
                raise RuntimeError("NQO requires a policy server")
            for name, value in profile.guc_settings().items():
                set_session_config(cursor, name, value)
            set_session_config(cursor, "nqo.server_url", server_url)
            set_session_config(cursor, "nqo.trajectory_log", db_trace_container)
            set_session_config(cursor, "nqo", "on")
        else:
            set_session_config(cursor, "nqo", "off")

        started = time.perf_counter()
        cursor.execute(sql)
        rows = [] if cursor.description is None else cursor
        result_hash, result_rows = hash_result_rows(rows)
    except psycopg2.errors.QueryCanceled as exc:
        status = "timeout"
        error = str(exc).replace("\n", " ")[:1000]
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = repr(exc)[:1000]
    finally:
        wall_ms = (
            (time.perf_counter() - started) * 1000.0 if started is not None else 0.0
        )
        cursor.close()
        connection.close()
    return {
        "status": status,
        "error": error,
        "client_wall_ms": wall_ms,
        "charged_wall_ms": wall_ms,
        "result_hash": result_hash,
        "result_rows": result_rows,
    }


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cache: Path
    profile: ActionProfile
    action_config_hash: str
    protocols: tuple[str, ...]
    fallback_queries: frozenset[str] = frozenset()


DATASETS = {
    # JOB's compact cache is backed exclusively by repetition-zero,
    # non-warmup physical executions.
    "job": DatasetSpec(
        name="JOB",
        cache=BUFFER_ROOT / "job_light.sql",
        profile=JOB_PROFILE,
        action_config_hash=NQO_ACTION_CONFIG_HASHES["JOB"],
        protocols=("base_query", "leave_one_out", "random"),
    ),
    # Unsupported TPC-H query shapes have no policy decisions and use the
    # paired PostgreSQL fallback.
    "tpch": DatasetSpec(
        name="TPCH",
        cache=BUFFER_ROOT / "tpch_light.sql",
        profile=TPCH_PROFILE,
        action_config_hash=NQO_ACTION_CONFIG_HASHES["TPCH"],
        protocols=("random",),
        fallback_queries=frozenset({"7", "8", "9", "13", "15", "22"}),
    ),
    "stack": DatasetSpec(
        name="STACK",
        cache=BUFFER_ROOT / "stack_light.sql",
        profile=STACK_PROFILE,
        action_config_hash=NQO_ACTION_CONFIG_HASHES["STACK"],
        protocols=("base_query", "leave_one_out", "random"),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResultCsv:
    def __init__(self, path: Path) -> None:
        self.path = path
        lock_root = DEFAULT_RUNTIME_DIR / "csv-locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_name = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
        self.lock_path = lock_root / f"{lock_name}.lock"
        self.rows: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
        if path.is_file() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                    raise RuntimeError(
                        f"incompatible result CSV columns in {path}: "
                        f"{reader.fieldnames}"
                    )
                for row in reader:
                    key = self.key(row)
                    if key in self.rows:
                        raise RuntimeError(f"duplicate result CSV key: {key}")
                    self.rows[key] = row

    @staticmethod
    def key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(row["dataset"]),
            str(row["protocol"]),
            str(row["fold"]),
            str(row["sql_path"]),
            str(row["method"]),
        )

    def contains(self, key: tuple[str, str, str, str, str]) -> bool:
        return key in self.rows

    def pg_runtime(self, dataset: str, sql_path: str) -> float:
        key = (dataset, "", "", sql_path, "PostgreSQL")
        if key not in self.rows:
            raise KeyError(f"missing PostgreSQL baseline row: {key}")
        return float(self.rows[key]["runtime_ms"])

    def put(self, row: dict[str, Any]) -> bool:
        """Append one row under an inter-process lock, without duplicates."""
        normalized = {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in CSV_FIELDS
        }
        key = self.key(normalized)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing: set[tuple[str, str, str, str, str]] = set()
            if self.path.is_file() and self.path.stat().st_size:
                with self.path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                        raise RuntimeError(
                            f"incompatible result CSV columns in {self.path}"
                        )
                    existing = {self.key(item) for item in reader}
            if key in existing:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                self.rows.setdefault(key, normalized)
                return False
            write_header = not self.path.is_file() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=CSV_FIELDS, lineterminator="\n"
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(normalized)
                handle.flush()
                os.fsync(handle.fileno())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        self.rows[key] = normalized
        return True


def collect_postgres(
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    results: ResultCsv,
) -> dict[str, int]:
    """Execute each selected PostgreSQL query once."""
    query_ids = [str(value) for value in workload_query_ids(spec.name)]
    if args.query_id:
        wanted = set(args.query_id)
        query_ids = [query_id for query_id in query_ids if query_id in wanted]
    if args.limit is not None:
        query_ids = query_ids[: args.limit]

    counters = {
        "executed": 0,
        "skipped": 0,
        "timeouts": 0,
        "errors": 0,
    }
    connection_args = argparse.Namespace(
        host=args.host,
        port=args.pg_port,
        user=args.user,
        workload=spec.name,
        database=args.database,
    )
    for index, query_id in enumerate(query_ids, start=1):
        sql_path = query_relative_path(spec.name, query_id)
        key = (spec.name, "", "", sql_path, "PostgreSQL")
        if results.contains(key):
            counters["skipped"] += 1
            print(
                f"[{spec.name}/PostgreSQL {index}/{len(query_ids)}] "
                f"{query_id}: already recorded",
                flush=True,
            )
            continue

        sql_lock_handle, sql_slot = acquire_sql_execution_slot(
            args.sql_execution_lock, args.sql_execution_slots
        )
        print(
            f"[{spec.name}/PostgreSQL {index}/{len(query_ids)}] {query_id}: "
            f"acquired SQL slot {sql_slot + 1}/{args.sql_execution_slots}",
            flush=True,
        )
        try:
            execution = execute_sql_once(
                connection_args,
                sql=load_query_sql(spec.name, query_id),
                profile=POSTGRES_PROFILE,
                timeout_ms=POSTGRES_TIMEOUT_MS,
                db_trace_container="",
                server_url=None,
            )
        finally:
            release_sql_execution_slot(sql_lock_handle)

        # Charge failed executions at the statement-timeout limit.
        if execution["status"] != "ok":
            execution["charged_wall_ms"] = float(POSTGRES_TIMEOUT_MS)
        if execution["status"] == "timeout":
            counters["timeouts"] += 1
        elif execution["status"] != "ok":
            counters["errors"] += 1
        counters["executed"] += 1

        results.put(
            {
                "dataset": spec.name,
                "protocol": "",
                "fold": "",
                "sql_path": sql_path,
                "method": "PostgreSQL",
                "runtime_ms": f"{float(execution['charged_wall_ms']):.12f}",
                "inference_ms": "0.000000000000",
                "status": execution["status"],
                "result_hash": execution.get("result_hash") or "",
                "result_rows": (
                    ""
                    if execution.get("result_rows") is None
                    else execution["result_rows"]
                ),
                "cache_id": "",
                "trajectory_hash": "",
                "checkpoint": "",
                "actions_json": "",
            }
        )
        print(
            f"[{spec.name}/PostgreSQL {index}/{len(query_ids)}] {query_id}: "
            f"status={execution['status']}, "
            f"runtime_ms={float(execution['charged_wall_ms']):.3f}, "
            f"rows={execution.get('result_rows')}",
            flush=True,
        )
    return counters


def policy_inference_ms(
    db_events: Iterable[dict[str, Any]],
    policy_events: Iterable[dict[str, Any]],
) -> float:
    measured = sum(
        float((event.get("timing_ms") or {}).get("policy") or 0.0)
        for event in db_events
    )
    if measured > 0.0:
        return measured
    return sum(
        float(event.get("latency_ms") or 0.0)
        for event in policy_events
        if event.get("phase") == "policy_decision"
    )


def actions_json(trajectory: list[dict[str, Any]]) -> str:
    """Preserve the ordered raw decisions; derive counts only in analysis."""
    actions = [
        {"phase": str(item["phase"]), "action": dict(item["action"])}
        for item in trajectory
    ]
    return json.dumps(actions, sort_keys=True, separators=(",", ":"))


def repo_relative(path: Path) -> str:
    """Return a stable path relative to the repository root."""
    relative = Path(os.path.relpath(path.resolve(), REPO.resolve())).as_posix()
    if relative == "." or relative.startswith(".."):
        return relative
    return f"./{relative}"


def query_relative_path(dataset: str, query_id: str) -> str:
    path = SQL_DIRS[dataset] / f"{query_id}.sql"
    if not path.is_file():
        raise FileNotFoundError(path)
    return repo_relative(path)


def load_query_sql(dataset: str, query_id: str) -> str:
    path = SQL_DIRS[dataset] / f"{query_id}.sql"
    if not path.is_file():
        raise FileNotFoundError(path)
    return re.sub(r"/\*\+.*?\*/", "", path.read_text(), flags=re.DOTALL)


def model_path(dataset: str, protocol: str, fold: str) -> Path:
    return MODEL_ROOT / dataset / PROTOCOL_DIRS[protocol] / fold / "best.pt"


def stage_model(checkpoint: Path, runtime_dir: Path) -> tuple[Path, str]:
    stage_policy_runtime(PGDB_ROOT)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = sha256(checkpoint)
    staged = runtime_dir / "models" / f"{digest}.pt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.is_file() or sha256(staged) != digest:
        shutil.copy2(checkpoint, staged)
    return staged, digest


def stage_catalog_snapshot(
    *,
    workload: str,
    host: str,
    port: int,
    user: str,
    database: str | None,
    runtime_dir: Path,
    runtime_container_dir: str,
) -> tuple[Path, str]:
    """Capture and stage the target DB catalog for a policy server."""
    path = runtime_dir / f"catalog-{workload.lower()}.snapshot.json"
    connection = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        dbname=database or WORKLOAD_DATABASES[workload.upper()],
    )
    try:
        snapshot = read_postgres_catalog(connection, schema="public")
    finally:
        connection.close()
    write_catalog_snapshot(path, snapshot)
    path.chmod(0o644)
    return path, f"{runtime_container_dir}/{path.name}"


def execute_miss(
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    query_id: str,
    sql: str,
    pg_first_ms: float,
    expected_result_hash: str | None,
    profile: ActionProfile,
    server: DockerFixedPolicyServer,
    policy_offset: int,
    runtime_dir: Path,
    runtime_container_dir: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    timeout_ms = first_runtime_timeout_ms(pg_first_ms, factor=5.0, cap_ms=60_000)
    charged_failure_ms = timeout_charged_runtime_ms(
        pg_first_ms, factor=5.0, cap_ms=360_000
    )
    connection_args = argparse.Namespace(
        host=args.host,
        port=args.pg_port,
        user=args.user,
        workload=spec.name,
        database=args.database,
    )
    trace_name = (
        f"miss-{spec.name.lower()}-{os.getpid()}-{uuid.uuid4().hex}.db.jsonl"
    )
    trace_host = runtime_dir / trace_name
    trace_container = f"{runtime_container_dir}/{trace_name}"
    policy_start = policy_offset
    execution = execute_sql_once(
        connection_args,
        sql=sql,
        profile=profile,
        timeout_ms=timeout_ms,
        db_trace_container=trace_container,
        server_url=server.action_url,
    )
    policy_events, policy_offset = read_jsonl_since(
        server.policy_log_host, policy_start
    )
    db_events = load_jsonl(trace_host)
    trace_host.unlink(missing_ok=True)
    validate_policy_state_contract(policy_events)
    if (
        execution["status"] == "ok"
        and expected_result_hash is not None
        and execution.get("result_hash") != expected_result_hash
    ):
        execution["status"] = "wrong_result"
        execution["charged_wall_ms"] = charged_failure_ms
    elif execution["status"] != "ok":
        execution["charged_wall_ms"] = charged_failure_ms
    return execution, policy_events, db_events, policy_offset, timeout_ms


def evaluate_case(
    *,
    args: argparse.Namespace,
    dataset_key: str,
    spec: DatasetSpec,
    protocol: str,
    fold: str,
    results: ResultCsv,
    store: ExperienceStore,
    runtime_dir: Path,
) -> dict[str, int]:
    fold_spec = split_folds(spec.name, protocol)[f"{protocol}_{fold}"]
    query_ids = [str(value) for value in fold_spec["test"]]
    if args.query_id:
        wanted = set(args.query_id)
        query_ids = [query_id for query_id in query_ids if query_id in wanted]
    if args.limit is not None:
        query_ids = query_ids[: args.limit]
    pending = []
    for query_id in query_ids:
        sql_path = query_relative_path(spec.name, query_id)
        key = (spec.name, protocol, fold, sql_path, "NQO")
        if not results.contains(key):
            pending.append(query_id)
    if not pending:
        print(f"[{spec.name}/{protocol}/{fold}] complete; nothing to do")
        return {"hits": 0, "misses": 0, "executed": 0, "skipped": len(query_ids)}

    checkpoint = model_path(dataset_key, protocol, fold)
    staged, digest = stage_model(checkpoint, runtime_dir)
    store.set_binding(
        protocol=protocol, fold=fold, checkpoint_sha256=digest
    )
    store.set_action_config_hash(spec.action_config_hash)
    try:
        runtime_relative = runtime_dir.resolve().relative_to(PGDB_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"runtime directory must be inside {PGDB_ROOT}: {runtime_dir}"
        ) from exc
    runtime_container_dir = f"/code/pgdb-dev/{runtime_relative.as_posix()}"
    staged_container = f"{runtime_container_dir}/models/{staged.name}"
    _catalog_path, catalog_container_path = stage_catalog_snapshot(
        workload=spec.name,
        host=args.host,
        port=args.pg_port,
        user=args.user,
        database=args.database,
        runtime_dir=runtime_dir,
        runtime_container_dir=runtime_container_dir,
    )
    run_label = (
        f"released-{spec.name.lower()}-{protocol}-{fold}-{digest[:12]}"
    )
    counters = {"hits": 0, "misses": 0, "executed": 0, "skipped": 0}
    server_context = DockerLearnedPolicyServer(
        container=args.container,
        port=args.server_port,
        listen_host=args.server_listen_host,
        action_host=args.server_action_host,
        profile=spec.profile,
        runtime_host_dir=runtime_dir,
        runtime_container_dir=runtime_container_dir,
        run_label=run_label,
        model_container_path=staged_container,
        model_method="standardmdp_rl",
        model_hidden=128,
        workload=spec.name,
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
        policy_version=f"released-{spec.name.lower()}-{protocol}-{fold}",
        action_ablation="none",
    )
    with server_context as server:
        policy_offset = 0
        for index, query_id in enumerate(pending, start=1):
            sql_path = query_relative_path(spec.name, query_id)
            sql = load_query_sql(spec.name, query_id)
            sql_hash = content_hash(sql)
            expected_hash = store.expected_result_hash(sql_hash)
            pg_first_ms = results.pg_runtime(spec.name, sql_path)
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
            cache_state = "hit"
            cache_id = ""
            if cached is None:
                counters["misses"] += 1
                cache_state = "miss"
                print(
                    f"[{spec.name}/{protocol}/{fold} {index}/{len(pending)}] "
                    f"{query_id}: cache miss",
                    flush=True,
                )
                if args.cache_miss == "error":
                    continue
                sql_lock_handle, sql_slot = acquire_sql_execution_slot(
                    args.sql_execution_lock, args.sql_execution_slots
                )
                print(
                    f"[{spec.name}/{protocol}/{fold}] {query_id}: acquired "
                    f"SQL slot {sql_slot + 1}/{args.sql_execution_slots}",
                    flush=True,
                )
                try:
                    (
                        execution,
                        policy_events,
                        db_events,
                        policy_offset,
                        timeout_ms,
                    ) = execute_miss(
                            args=args,
                            spec=spec,
                            query_id=query_id,
                            sql=sql,
                            pg_first_ms=pg_first_ms,
                            expected_result_hash=expected_hash,
                            profile=spec.profile,
                            server=server,
                            policy_offset=policy_offset,
                            runtime_dir=runtime_dir,
                            runtime_container_dir=runtime_container_dir,
                        )
                finally:
                    release_sql_execution_slot(sql_lock_handle)
                trajectory = semantic_policy_trajectory(policy_events)
                if execution["status"] in {"ok", "timeout"}:
                    cache_id, inserted = store.append_execution(
                        query_id=query_id,
                        sql_hash=sql_hash,
                        trajectory=trajectory,
                        db_events=db_events,
                        status=execution["status"],
                        first_runtime_ms=execution["client_wall_ms"],
                        charged_runtime_ms=execution["charged_wall_ms"],
                        timeout_limit_ms=timeout_ms,
                        result_hash=execution.get("result_hash"),
                        result_rows=execution.get("result_rows"),
                    )
                    print(
                        f"[{spec.name}/{protocol}/{fold}] {query_id}: "
                        f"cache {'appended' if inserted else 'already present'} "
                        f"{cache_id[:12]}",
                        flush=True,
                    )
                counters["executed"] += 1
            else:
                counters["hits"] += 1
                execution = cached["execution"]
                policy_events = cached["policy_events"]
                db_events = cached["db_events"]
                cache_id = str(cached["cache_source"])

            trajectory = semantic_policy_trajectory(policy_events)
            if query_id in spec.fallback_queries:
                if trajectory:
                    raise RuntimeError(
                        f"fallback query {spec.name}/{query_id} produced "
                        "policy decisions"
                    )
                runtime_ms = pg_first_ms
                inference_ms = 0.0
                result_status = "fallback"
            else:
                runtime_ms = float(execution["charged_wall_ms"])
                inference_ms = policy_inference_ms(db_events, policy_events)
                result_status = str(execution["status"])
            results.put(
                {
                    "dataset": spec.name,
                    "protocol": protocol,
                    "fold": fold,
                    "sql_path": sql_path,
                    "method": "NQO",
                    "runtime_ms": f"{runtime_ms:.12f}",
                    "inference_ms": f"{inference_ms:.12f}",
                    "status": result_status,
                    "result_hash": execution.get("result_hash") or "",
                    "result_rows": (
                        ""
                        if execution.get("result_rows") is None
                        else execution["result_rows"]
                    ),
                    "cache_id": cache_id,
                    "trajectory_hash": semantic_trajectory_hash(trajectory),
                    "checkpoint": repo_relative(checkpoint),
                    "actions_json": actions_json(trajectory),
                }
            )
            print(
                f"[{spec.name}/{protocol}/{fold} {index}/{len(pending)}] "
                f"{query_id}: {cache_state}, status={result_status}, "
                f"runtime_ms={runtime_ms:.3f}",
                flush=True,
            )
    return counters


def evaluate_standalone(
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    method: str,
    results: ResultCsv,
    store: ExperienceStore,
    runtime_dir: Path,
) -> dict[str, int]:
    profile = profile_for(spec.name, method)
    profile_hash = content_hash(profile.to_dict())
    compatible_profile_hashes = profile.compatible_hashes()
    method_label = METHOD_LABELS[method]
    query_ids = [str(value) for value in workload_query_ids(spec.name)]
    if args.query_id:
        wanted = set(args.query_id)
        query_ids = [query_id for query_id in query_ids if query_id in wanted]
    if args.limit is not None:
        query_ids = query_ids[: args.limit]

    counters = {"hits": 0, "misses": 0, "executed": 0, "skipped": 0}
    missing: list[tuple[str, str, str, str, float]] = []
    fixed_actions = json.dumps(
        representative_actions(profile), sort_keys=True, separators=(",", ":")
    )
    for index, query_id in enumerate(query_ids, start=1):
        sql_path = query_relative_path(spec.name, query_id)
        key = (spec.name, "", "", sql_path, method_label)
        if results.contains(key):
            counters["skipped"] += 1
            continue
        sql = load_query_sql(spec.name, query_id)
        sql_hash = content_hash(sql)
        cached = next(
            (
                match
                for compatible_hash in compatible_profile_hashes
                if (
                    match := store.fixed_execution(
                        sql_hash=sql_hash,
                        action_config_hash=compatible_hash,
                    )
                )
                is not None
            ),
            None,
        )
        if cached is None:
            counters["misses"] += 1
            missing.append(
                (
                    query_id,
                    sql_path,
                    sql,
                    sql_hash,
                    results.pg_runtime(spec.name, sql_path),
                )
            )
            print(
                f"[{spec.name}/{method_label} {index}/{len(query_ids)}] "
                f"{query_id}: cache miss",
                flush=True,
            )
            continue
        counters["hits"] += 1
        execution = cached["execution"]
        results.put(
            {
                "dataset": spec.name,
                "protocol": "",
                "fold": "",
                "sql_path": sql_path,
                "method": method_label,
                "runtime_ms": f"{float(execution['charged_wall_ms']):.12f}",
                "inference_ms": "0.000000000000",
                "status": execution["status"],
                "result_hash": execution.get("result_hash") or "",
                "result_rows": (
                    ""
                    if execution.get("result_rows") is None
                    else execution["result_rows"]
                ),
                "cache_id": cached["cache_id"],
                "trajectory_hash": cached["trajectory_hash"],
                "checkpoint": "",
                "actions_json": fixed_actions,
            }
        )

    if not missing or args.cache_miss == "error":
        return counters

    try:
        runtime_relative = runtime_dir.resolve().relative_to(PGDB_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"runtime directory must be inside {PGDB_ROOT}: {runtime_dir}"
        ) from exc
    runtime_container_dir = f"/code/pgdb-dev/{runtime_relative.as_posix()}"
    server_context = DockerFixedPolicyServer(
        container=args.container,
        port=args.server_port,
        profile=profile,
        runtime_host_dir=runtime_dir,
        runtime_container_dir=runtime_container_dir,
        run_label=f"released-{spec.name.lower()}-{method}",
    )
    with server_context as server:
        policy_offset = 0
        for index, (query_id, sql_path, sql, sql_hash, pg_first_ms) in enumerate(
            missing, start=1
        ):
            sql_lock_handle, sql_slot = acquire_sql_execution_slot(
                args.sql_execution_lock, args.sql_execution_slots
            )
            print(
                f"[{spec.name}/{method_label} {index}/{len(missing)}] "
                f"{query_id}: acquired SQL slot "
                f"{sql_slot + 1}/{args.sql_execution_slots}",
                flush=True,
            )
            try:
                (
                    execution,
                    policy_events,
                    db_events,
                    policy_offset,
                    timeout_ms,
                ) = execute_miss(
                    args=args,
                    spec=spec,
                    query_id=query_id,
                    sql=sql,
                    pg_first_ms=pg_first_ms,
                    expected_result_hash=store.expected_result_hash(sql_hash),
                    profile=profile,
                    server=server,
                    policy_offset=policy_offset,
                    runtime_dir=runtime_dir,
                    runtime_container_dir=runtime_container_dir,
                )
            finally:
                release_sql_execution_slot(sql_lock_handle)
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
                    action_config_hash=profile_hash,
                    result_hash=execution.get("result_hash"),
                    result_rows=execution.get("result_rows"),
                )
            counters["executed"] += 1
            results.put(
                {
                    "dataset": spec.name,
                    "protocol": "",
                    "fold": "",
                    "sql_path": sql_path,
                    "method": method_label,
                    "runtime_ms": f"{float(execution['charged_wall_ms']):.12f}",
                    "inference_ms": "0.000000000000",
                    "status": execution["status"],
                    "result_hash": execution.get("result_hash") or "",
                    "result_rows": (
                        ""
                        if execution.get("result_rows") is None
                        else execution["result_rows"]
                    ),
                    "cache_id": cache_id,
                    "trajectory_hash": semantic_trajectory_hash(trajectory),
                    "checkpoint": "",
                    "actions_json": (
                        actions_json(trajectory) if trajectory else fixed_actions
                    ),
                }
            )
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one PostgreSQL measurement, a versioned NQO checkpoint, "
            "or a fixed standalone Action"
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="job")
    parser.add_argument(
        "--method",
        choices=("nqo", "postgres", *STANDALONE_METHODS),
        default="nqo",
    )
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_DIRS))
    parser.add_argument("--fold", choices=("a", "b", "c"))
    parser.add_argument("--query-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cache-miss", choices=("error", "execute"), default="error"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--container", default="pgdb_dev_opt")
    parser.add_argument("--server-port", type=int, default=18095)
    parser.add_argument("--server-listen-host", default="127.0.0.1")
    parser.add_argument("--server-action-host", default="127.0.0.1")
    parser.add_argument(
        "--model-device",
        default="auto",
        help="model device inside the database container (default: auto)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument(
        "--database",
        help="PostgreSQL database override for the selected workload",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="lightweight replay-cache override",
    )
    parser.add_argument(
        "--sql-execution-lock",
        type=Path,
        default=(
            PGDB_ROOT
            / ".nqo_runtime"
            / "online"
            / ".nqo-sql.lock"
        ),
    )
    parser.add_argument("--sql-execution-slots", type=int, default=1)
    args = parser.parse_args()

    if (
        args.cache_miss == "execute"
        and args.method != "postgres"
        and args.cache is None
    ):
        parser.error(
            "--cache-miss execute requires an explicit writable --cache; "
            "released buffers are read-only"
        )

    spec = DATASETS[args.dataset]
    if args.cache is not None:
        spec = DatasetSpec(
            name=spec.name,
            cache=args.cache,
            profile=spec.profile,
            action_config_hash=spec.action_config_hash,
            protocols=spec.protocols,
            fallback_queries=spec.fallback_queries,
        )
    if args.sql_execution_slots < 1:
        parser.error("--sql-execution-slots must be positive")
    if args.method == "nqo":
        if args.protocol is None or args.fold is None:
            parser.error("--method nqo requires --protocol and --fold")
        if args.protocol not in spec.protocols:
            parser.error(
                f"{spec.name} does not support protocol {args.protocol!r}"
            )
        if not args.output.is_file():
            raise FileNotFoundError(
                "collect PostgreSQL rows before NQO evaluation with: "
                f"python3 scripts/reproduce/nqo/run.py --method postgres "
                f"--dataset {args.dataset} --output {args.output}"
            )
        runtime_label = f"{args.protocol}-{args.fold}"
    elif args.method == "postgres":
        if args.protocol is not None or args.fold is not None:
            parser.error(
                "--method postgres is workload-wide; omit --protocol and --fold"
            )
        runtime_label = "postgres"
    else:
        if args.method not in DATASET_METHODS[spec.name]:
            parser.error(
                f"{spec.name} has no benchmark result for standalone method "
                f"{args.method!r}"
            )
        if args.protocol is not None or args.fold is not None:
            parser.error(
                "standalone methods are workload-wide; omit --protocol and --fold"
            )
        if not args.output.is_file():
            raise FileNotFoundError(
                "collect PostgreSQL rows before standalone evaluation"
            )
        runtime_label = args.method
    args.runtime_dir = args.runtime_dir or (
        DEFAULT_RUNTIME_DIR / args.dataset / runtime_label
    )
    results = ResultCsv(args.output)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "postgres":
        print(f"== {spec.name}/PostgreSQL (one run) ==", flush=True)
        totals = collect_postgres(args=args, spec=spec, results=results)
        summary = {
            "dataset": spec.name,
            "method": "PostgreSQL",
            "measurements_per_query": 1,
            "timing": (
                "execute + complete result fetch + canonical SHA-256 hash"
            ),
            **totals,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if totals["errors"] else 0

    stage_policy_runtime(PGDB_ROOT)
    with ExperienceStore(
        spec.cache, writable=args.cache_miss == "execute"
    ) as store:
        if args.method == "nqo":
            print(f"== {spec.name}/{args.protocol}/{args.fold} ==", flush=True)
            totals = evaluate_case(
                args=args,
                dataset_key=args.dataset,
                spec=spec,
                protocol=args.protocol,
                fold=args.fold,
                results=results,
                store=store,
                runtime_dir=args.runtime_dir,
            )
            summary = {
                "dataset": spec.name,
                "protocol": args.protocol,
                "fold": args.fold,
                "method": "NQO",
                "cache_miss_mode": args.cache_miss,
                **totals,
            }
        else:
            print(
                f"== {spec.name}/{METHOD_LABELS[args.method]} ==", flush=True
            )
            totals = evaluate_standalone(
                args=args,
                spec=spec,
                method=args.method,
                results=results,
                store=store,
                runtime_dir=args.runtime_dir,
            )
            summary = {
                "dataset": spec.name,
                "method": METHOD_LABELS[args.method],
                "cache_miss_mode": args.cache_miss,
                **totals,
            }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if totals["misses"] and args.cache_miss == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
