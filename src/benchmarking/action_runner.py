#!/usr/bin/env python3
"""Run reproducible PostgreSQL/NQO action benchmarks."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PGDB_ROOT = ROOT.parent / "pgdb"

from benchmarking.workloads import (  # noqa: E402
    query_sql,
    split_folds,
    workload_query_ids,
)

from experience.store import (  # noqa: E402
    ExperienceStore,
    content_hash,
    semantic_trajectory_hash,
)
from database.catalog import write_catalog_snapshot  # noqa: E402
from optimization.actions import (  # noqa: E402
    ONLINE_ACTION_SPACE_VERSION,
    STATEMENT_TIMEOUT_MS,
    TIMEOUT_CHARGE_CAP_MS,
    TIMEOUT_CHARGE_FACTOR,
    ActionProfile,
    apply_action_config,
    builtin_profiles,
    first_runtime_timeout_ms,
    global_experience_path,
    load_action_config,
    load_jsonl,
    query_runtime_summary,
    semantic_policy_trajectory,
    timeout_charged_runtime_ms,
    validate_policy_state_contract,
    workload_metrics,
)
from benchmarking.utils import safe_name, utc_stamp, write_json_atomic  # noqa: E402
from benchmarking.policy_server import (  # noqa: E402
    DockerFixedPolicyServer,
    DockerLearnedPolicyServer,
)
from benchmarking.database_runner import (  # noqa: E402
    create_catalog_snapshot,
    prewarm_database,
    run_query_once,
)
from benchmarking.execution_cache import (  # noqa: E402
    lookup_cached_execution,
    read_jsonl_since,
)
from benchmarking.trajectory import (  # noqa: E402
    EpisodeCsv,
    ingest_benchmark_trajectory,
)
from benchmarking.run_environment import (  # noqa: E402
    file_sha256,
    portable_path,
    repository_version,
    stage_policy_runtime,
    validate_resume_manifest,
)
from benchmarking.run_results import write_result_tables  # noqa: E402



def acquire_sql_execution_slot(lock_path: Path, slots: int):
    """Acquire one member of a cross-process SQL execution lock pool."""
    candidates = (
        [lock_path]
        if slots == 1
        else [Path(f"{lock_path}.slot-{index}") for index in range(slots)]
    )
    while True:
        for index, candidate in enumerate(candidates):
            handle = candidate.open("a+")
            try:
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                handle.close()
                continue
            return handle, index
        time.sleep(0.05)


def release_sql_execution_slot(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()











def query_ids_for(args: argparse.Namespace) -> list[str]:
    if args.query_id:
        return list(args.query_id)
    if args.role == "all":
        query_ids = workload_query_ids(args.workload)
    else:
        fold_key = f"{args.protocol}_{args.fold}"
        folds = split_folds(args.workload, args.protocol)
        query_ids = list(folds[fold_key][args.role])
    if args.limit is not None:
        query_ids = query_ids[: args.limit]
    return query_ids


def load_profiles(args: argparse.Namespace) -> list[ActionProfile]:
    catalog = builtin_profiles()
    if args.model_path is not None:
        catalog["learned"] = ActionProfile(name="learned")
    requested = [item.strip() for item in args.profiles.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(catalog))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    profiles = []
    for name in requested:
        profile = replace(
            catalog[name],
            max_rounds=args.max_rounds,
            search_max_rels=args.search_max_rels,
            search_exact_cardinality=args.search_exact_cardinality,
            aja_conservative_rows=args.aja_conservative_rows,
            aja_aggressive_rows=args.aja_aggressive_rows,
            aja_max_nestloop_cost_ratio_pct=(args.aja_max_nestloop_cost_ratio_pct),
            aja_aggressive_max_nestloop_cost_ratio_pct=(
                args.aja_aggressive_max_nestloop_cost_ratio_pct
            ),
            lip_max_build_relation_rows=args.lip_max_build_relation_rows,
            lip_selective_plan_rows=args.lip_selective_plan_rows,
            lip_max_build_selectivity_pct=(args.lip_max_build_selectivity_pct),
            lip_min_probe_ratio=args.lip_min_probe_ratio,
            lip_max_filters=args.lip_max_filters,
        )
        if name == "query_split":
            profile = replace(
                profile,
                dec_rounds=args.dec_rounds,
                sched_alpha=args.sched_alpha,
                sched_alpha_sequence=tuple(
                    float(item.strip())
                    for item in args.sched_alpha_sequence.split(",")
                    if item.strip()
                ),
            )
        profiles.append(profile)
    return profiles




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nqo-benchmark run")
    parser.add_argument("--workload", choices=("JOB", "STACK", "TPCH"), default="JOB")
    parser.add_argument(
        "--profiles",
        default=(
            "pg,nqo_none,query_split,split_search,top5,top10,lip_full,"
            "lip_selective,aja_conservative,aja_aggressive"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("base_query", "leave_one_out", "random"),
        default="random",
    )
    parser.add_argument("--fold", choices=("a", "b", "c"), default="a")
    parser.add_argument(
        "--role",
        choices=("train", "validation", "test", "all"),
        default="all",
    )
    parser.add_argument("--query-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=1)
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--experience-db", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--action-config",
        type=Path,
        help="frozen dataset-level Action implementation configuration",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--container", default="pgdb_dev_opt")
    parser.add_argument("--pgdb-root", type=Path, default=DEFAULT_PGDB_ROOT)
    parser.add_argument("--sql-execution-lock", type=Path)
    parser.add_argument(
        "--sql-execution-slots",
        type=int,
        default=1,
        help="number of physical SQL executions admitted by the lock pool",
    )
    parser.add_argument("--ai-port", type=int, default=18090)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--catalog-path",
        type=Path,
        help=(
            "pre-captured database catalog; when omitted, learned runs "
            "capture the target database at startup"
        ),
    )
    parser.add_argument("--model-method", default="standardmdp_rl")
    parser.add_argument("--model-hidden", type=int, default=128)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument(
        "--model-nqo-src",
        default="/code/pgdb-dev/.nqo_runtime/nqo/src",
        help="container path to the model source matching the checkpoint",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exploration-epsilon", type=float, default=0.0)
    parser.add_argument("--coverage-counts", type=Path)
    parser.add_argument("--coverage-mix", type=float, default=0.0)
    parser.add_argument("--coverage-power", type=float, default=0.5)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--stochastic-heads",
        help=(
            "comma-separated learned-policy phases sampled stochastically; "
            "defaults to all phases"
        ),
    )
    parser.add_argument("--policy-version")
    parser.add_argument(
        "--action-ablation",
        choices=("none", "no_dec", "no_enum", "no_filter", "no_ajoin"),
        default="none",
    )
    parser.add_argument(
        "--replay-group",
        help=(
            "counterfactual label group; runs with different implementation "
            "parameters must use different groups"
        ),
    )
    parser.add_argument(
        "--execution-cache",
        choices=("off", "read-only", "write-only", "read-write"),
        default="off",
    )
    parser.add_argument(
        "--cache-match-mode",
        choices=("prefix", "statewise", "legacy"),
        default="statewise",
        help=(
            "prefix traverses immutable trajectories using complete model "
            "inputs; statewise evaluates each semantic state once; legacy "
            "replays complete trajectories one at a time"
        ),
    )
    parser.add_argument("--baseline-timeout-ms", type=int)
    parser.add_argument("--timeout-factor", type=float, default=2.0)
    parser.add_argument("--timeout-slack-ms", type=int, default=2_000)
    parser.add_argument("--timeout-min-ms", type=int, default=5_000)
    parser.add_argument("--timeout-max-ms", type=int)
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=STATEMENT_TIMEOUT_MS,
    )
    parser.add_argument(
        "--timeout-charge-factor",
        type=float,
        default=TIMEOUT_CHARGE_FACTOR,
    )
    parser.add_argument(
        "--timeout-charge-cap-ms",
        type=int,
        default=TIMEOUT_CHARGE_CAP_MS,
    )
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument(
        "--dec-rounds",
        type=int,
        default=-1,
        help="fixed number of rounds in which Dec may choose Apply",
    )
    parser.add_argument(
        "--sched-alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sched-alpha-sequence",
        default="",
        help=(
            "comma-separated per-round alpha values for fixed query-split "
            "counterfactual collection"
        ),
    )
    parser.add_argument("--search-max-rels", type=int, default=12)
    parser.add_argument(
        "--search-exact-cardinality",
        action="store_true",
        help="plan every connected DP subset instead of pairwise composition",
    )
    parser.add_argument("--aja-conservative-rows", type=int, default=362_443)
    parser.add_argument("--aja-aggressive-rows", type=int, default=3_624_434)
    parser.add_argument("--aja-max-nestloop-cost-ratio-pct", type=int, default=150)
    parser.add_argument(
        "--aja-aggressive-max-nestloop-cost-ratio-pct",
        type=int,
        default=125,
    )
    parser.add_argument("--lip-max-build-relation-rows", type=int, default=500_000)
    parser.add_argument("--lip-selective-plan-rows", type=int, default=10_000)
    parser.add_argument("--lip-max-build-selectivity-pct", type=int, default=10)
    parser.add_argument("--lip-min-probe-ratio", type=int, default=2)
    parser.add_argument("--lip-max-filters", type=int, default=4)
    parser.add_argument(
        "--correctness",
        choices=("strict", "warn", "off"),
        default="strict",
    )
    args = parser.parse_args(argv)
    frozen_action_config = None
    if args.action_config is not None:
        args.action_config = args.action_config.resolve()
        frozen_action_config = load_action_config(
            args.action_config,
            workload=args.workload,
        )
        apply_action_config(args, frozen_action_config)
        execution_protocol = frozen_action_config["execution_protocol"]
        for key in (
            "statement_timeout_ms",
            "timeout_charge_factor",
            "timeout_charge_cap_ms",
        ):
            if key in execution_protocol:
                setattr(args, key, execution_protocol[key])
    if (
        args.warmups < 0
        or args.measurements != 1
        or args.warmups + args.measurements > 3
    ):
        parser.error("use one measured execution and at most three total executions")
    if args.statement_timeout_ms != STATEMENT_TIMEOUT_MS:
        parser.error(
            f"formal online experiments require "
            f"--statement-timeout-ms={STATEMENT_TIMEOUT_MS}"
        )
    # All formal Action profiles stop physical execution at
    # ceil(min(5 * PG-official, 60s)).  Timeout reward/metric accounting keeps
    # the paper's wider min(5 * PG-official, 360s) charge.  ``PG-official`` is
    # the last non-warmup sample in the supplied baseline (the third physical
    # execution for the 2-warmup + 1-measurement protocol).
    args.timeout_charge_factor = 5.0
    args.timeout_charge_cap_ms = 360_000
    if args.timeout_charge_factor <= 0.0 or args.timeout_charge_cap_ms <= 0:
        parser.error("timeout charge factor and cap must be positive")
    if args.sql_execution_slots < 1:
        parser.error("--sql-execution-slots must be positive")
    if args.dec_rounds < -1:
        parser.error("--dec-rounds must be -1 or nonnegative")
    if args.sql_execution_slots > 1 and args.sql_execution_lock is None:
        parser.error("--sql-execution-slots > 1 requires --sql-execution-lock")
    if not 0.0 <= args.coverage_mix < 1.0:
        parser.error("--coverage-mix must be in [0, 1)")
    if args.coverage_power < 0.0:
        parser.error("--coverage-power must be nonnegative")
    if args.coverage_mix > 0.0 and args.coverage_counts is None:
        parser.error("--coverage-counts is required when coverage mix is enabled")
    if args.execution_cache in {"read-only", "read-write"} and args.warmups:
        parser.error("execution-cache reads require --warmups=0")
    if args.sql_execution_lock is not None:
        args.sql_execution_lock = args.sql_execution_lock.resolve()
        args.sql_execution_lock.parent.mkdir(parents=True, exist_ok=True)

    args.experiment_id = args.experiment_id or f"{args.workload.lower()}-{utc_stamp()}"
    args.replay_group = args.replay_group or args.experiment_id
    output_dir = args.output_root.resolve() / args.workload.lower() / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    experience_path = (
        args.experience_db.resolve()
        if args.experience_db is not None
        else global_experience_path(args.pgdb_root, args.workload)
    )
    baseline_path = output_dir / "baseline.json"
    baseline_source = args.baseline_json or baseline_path
    baseline = (
        json.loads(baseline_source.read_text(encoding="utf-8"))
        if baseline_source.is_file()
        else {}
    )
    query_ids = query_ids_for(args)
    profiles = load_profiles(args)
    buffers_root = (ROOT / "results" / "buffers").resolve()
    if (
        any(not profile.is_postgres for profile in profiles)
        and experience_path.is_relative_to(buffers_root)
    ):
        raise RuntimeError(
            "benchmark collection cannot write directly to results/buffers; "
            "copy the versioned buffer to a temporary *_light.sql file first"
        )
    if any(profile.name != "pg" for profile in profiles):
        stage_policy_runtime(args.pgdb_root)
    if not query_ids:
        raise RuntimeError("the selected split contains no queries")
    if profiles[0].name != "pg" and not baseline:
        raise RuntimeError("run the pg profile first or resume with baseline.json")

    dataset_snapshot_id = args.workload.upper()
    code_versions = {
        "nqo": repository_version(ROOT),
        "pgdb": repository_version(args.pgdb_root.resolve()),
    }
    implementation_version = content_hash(code_versions)
    total_repetitions = args.warmups + args.measurements
    source_model = None
    source_model_hash = None
    if args.model_path is not None:
        source_model = args.model_path.resolve()
        if not source_model.is_file():
            raise FileNotFoundError(source_model)
        source_model_hash = file_sha256(source_model)
        args.policy_version = args.policy_version or source_model.stem
    model_metadata = {
        "source": portable_path(source_model) if source_model is not None else None,
        "source_sha256": source_model_hash,
        "policy_version": args.policy_version,
        "method": args.model_method,
        "hidden": args.model_hidden,
        "inference_mode": args.inference_mode,
        "temperature": args.temperature,
        "exploration_epsilon": args.exploration_epsilon,
        "coverage_counts": (
            portable_path(args.coverage_counts)
            if args.coverage_counts is not None
            else None
        ),
        "coverage_counts_hash": (
            file_sha256(args.coverage_counts.resolve())
            if args.coverage_counts is not None
            else None
        ),
        "coverage_mix": args.coverage_mix,
        "coverage_power": args.coverage_power,
        "stochastic_heads": args.stochastic_heads,
        "deterministic_rule": "masked_argmax",
        "sampling_seed": args.sampling_seed,
        "action_ablation": args.action_ablation,
    }
    resume_contract = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "workload": args.workload,
        "protocol": args.protocol,
        "fold": args.fold,
        "role": args.role,
        "replay_group": args.replay_group,
        "query_ids": query_ids,
        "dataset_snapshot_id": dataset_snapshot_id,
        "implementation_version": implementation_version,
        "profiles": [profile.to_dict() for profile in profiles],
        "model": model_metadata,
        "warmups": args.warmups,
        "measurements": args.measurements,
        "prewarm": bool(args.prewarm),
        "timeout": {
            "statement_timeout_ms": args.statement_timeout_ms,
            "charge_factor": args.timeout_charge_factor,
            "charge_cap_ms": args.timeout_charge_cap_ms,
            "execution_mode": "ceil(min(5*pg_official_ms,60000))",
            "charge_mode": "min(5*pg_official_ms,360000)",
        },
        "sql_execution_slots": args.sql_execution_slots,
        "correctness": args.correctness,
        "action_config_hash": (
            frozen_action_config["config_hash"]
            if frozen_action_config is not None
            else None
        ),
        "experience_path": portable_path(experience_path),
        "execution_cache": {
            "mode": args.execution_cache,
            "action_space_version": ONLINE_ACTION_SPACE_VERSION,
            "match_mode": args.cache_match_mode,
        },
        "baseline_hash": (None if profiles[0].name == "pg" else content_hash(baseline)),
    }
    existing_manifest = validate_resume_manifest(
        output_dir / "manifest.json",
        resume_contract,
    )
    episodes_csv = EpisodeCsv(output_dir / "episodes.csv")
    prewarm = prewarm_database(args) if args.prewarm else {"enabled": False}
    runtime_host_dir = (
        args.pgdb_root.resolve() / ".nqo_runtime" / "benchmark" / args.experiment_id
    )
    runtime_container_dir = (
        f"/code/pgdb-dev/.nqo_runtime/benchmark/{args.experiment_id}"
    )
    runtime_host_dir.mkdir(parents=True, exist_ok=True)
    model_container_path = None
    coverage_counts_container_path = None
    catalog_container_path = None
    catalog_hash = None
    catalog_source = None
    if any(profile.name == "learned" for profile in profiles):
        staged_catalog = runtime_host_dir / "catalog.snapshot.json"
        if args.catalog_path is not None:
            source_catalog = args.catalog_path.resolve()
            if not source_catalog.is_file():
                raise FileNotFoundError(source_catalog)
            snapshot = json.loads(source_catalog.read_text(encoding="utf-8"))
            catalog_hash = write_catalog_snapshot(staged_catalog, snapshot)
            catalog_source = portable_path(source_catalog)
        else:
            catalog_hash = create_catalog_snapshot(args, staged_catalog)
            catalog_source = "postgresql"
        staged_catalog.chmod(0o644)
        catalog_container_path = f"{runtime_container_dir}/{staged_catalog.name}"
    if source_model is not None:
        staged_model_dir = runtime_host_dir / "models"
        staged_model_dir.mkdir(parents=True, exist_ok=True)
        staged_model = staged_model_dir / source_model.name
        shutil.copy2(source_model, staged_model)
        staged_model.chmod(0o644)
        model_container_path = f"{runtime_container_dir}/models/{source_model.name}"
    if args.coverage_counts is not None:
        source_coverage = args.coverage_counts.resolve()
        if not source_coverage.is_file():
            raise FileNotFoundError(source_coverage)
        staged_coverage_dir = runtime_host_dir / "coverage"
        staged_coverage_dir.mkdir(parents=True, exist_ok=True)
        staged_coverage = staged_coverage_dir / source_coverage.name
        shutil.copy2(source_coverage, staged_coverage)
        staged_coverage.chmod(0o644)
        coverage_counts_container_path = (
            f"{runtime_container_dir}/coverage/{source_coverage.name}"
        )

    manifest = {
        "manifest_schema_version": 2,
        "experiment_id": args.experiment_id,
        "created_at": (
            existing_manifest.get("created_at")
            if existing_manifest is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "last_started_at": datetime.now(timezone.utc).isoformat(),
        "workload": args.workload,
        "protocol": args.protocol,
        "fold": args.fold,
        "role": args.role,
        "replay_group": args.replay_group,
        "query_ids": query_ids,
        "dataset_snapshot_id": dataset_snapshot_id,
        "code_versions": code_versions,
        "implementation_version": implementation_version,
        "profiles": [profile.to_dict() for profile in profiles],
        "model": model_metadata,
        "catalog": {
            "source": catalog_source,
            "snapshot_sha256": catalog_hash,
        },
        "warmups": args.warmups,
        "measurements": args.measurements,
        "prewarm": prewarm,
        "timeout": {
            "statement_timeout_ms": args.statement_timeout_ms,
            "charge_factor": args.timeout_charge_factor,
            "charge_cap_ms": args.timeout_charge_cap_ms,
            "execution_mode": "ceil(min(5*pg_official_ms,60000))",
            "charge_mode": "min(5*pg_official_ms,360000)",
        },
        "sql_execution_slots": args.sql_execution_slots,
        "action_config": {
            "path": portable_path(args.action_config) if args.action_config is not None else None,
            "config_hash": (
                frozen_action_config["config_hash"]
                if frozen_action_config is not None
                else None
            ),
        },
        "execution_cache": {
            "mode": args.execution_cache,
            "action_space_version": ONLINE_ACTION_SPACE_VERSION,
            "match_mode": args.cache_match_mode,
        },
        "resume_contract": resume_contract,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)

    with ExperienceStore(experience_path) as store:
        for profile_index, profile in enumerate(profiles):
            profile_label = (
                f"learned@{args.policy_version}"
                if profile.name == "learned"
                else profile.name
            )
            profile_environment_hash = content_hash(
                {
                    "implementation": implementation_version,
                    "dataset": dataset_snapshot_id,
                    "runtime_gucs": {
                        key: value
                        for key, value in profile.guc_settings().items()
                        if key != "nqo.search_topk"
                    },
                }
            )
            run_id = safe_name(
                f"run_{args.experiment_id}_{profile_label}_{args.protocol}_"
                f"{args.fold}_{args.role}"
            )

            if profile.is_postgres:
                server_context = nullcontext(None)
            elif profile.name == "learned":
                if (
                    model_container_path is None
                    or args.policy_version is None
                    or catalog_container_path is None
                ):
                    raise RuntimeError("learned profile requires --model-path")
                server_context = DockerLearnedPolicyServer(
                    container=args.container,
                    port=args.ai_port + profile_index,
                    profile=profile,
                    runtime_host_dir=runtime_host_dir,
                    runtime_container_dir=runtime_container_dir,
                    run_label=safe_name(profile_label),
                    model_container_path=model_container_path,
                    model_method=args.model_method,
                    model_hidden=args.model_hidden,
                    workload=args.workload,
                    catalog_container_path=catalog_container_path,
                    model_device=args.model_device,
                    nqo_src=args.model_nqo_src,
                    inference_mode=args.inference_mode,
                    temperature=args.temperature,
                    exploration_epsilon=args.exploration_epsilon,
                    coverage_counts_container_path=coverage_counts_container_path,
                    coverage_mix=args.coverage_mix,
                    coverage_power=args.coverage_power,
                    stochastic_heads=args.stochastic_heads,
                    sampling_seed=args.sampling_seed,
                    policy_version=args.policy_version,
                    action_ablation=args.action_ablation,
                )
            else:
                server_context = DockerFixedPolicyServer(
                    container=args.container,
                    port=args.ai_port + profile_index,
                    profile=profile,
                    runtime_host_dir=runtime_host_dir,
                    runtime_container_dir=runtime_container_dir,
                    run_label=safe_name(profile_label),
                )
            with server_context as server:
                policy_offset = 0
                if server is not None and server.policy_log_host.is_file():
                    policy_offset = server.policy_log_host.stat().st_size
                for query_index, query_id in enumerate(query_ids, start=1):
                    sql = query_sql(args.workload, query_id)
                    sql_hash = content_hash(sql)
                    pg_ms = (
                        float(baseline[query_id]["median_charged_ms"])
                        if query_id in baseline
                        else None
                    )
                    timeout_ms = (
                        first_runtime_timeout_ms(
                            pg_ms,
                            factor=args.timeout_charge_factor,
                            cap_ms=60_000,
                        )
                        if (
                            not profile.is_postgres
                            and pg_ms is not None
                        )
                        else args.statement_timeout_ms
                    )
                    failure_charge_ms = (
                        float(timeout_ms)
                        if profile.is_postgres or pg_ms is None
                        else timeout_charged_runtime_ms(
                            pg_ms,
                            factor=args.timeout_charge_factor,
                            cap_ms=args.timeout_charge_cap_ms,
                        )
                    )
                    existing_query_records = [
                        item
                        for item in episodes_csv.profile_records(profile_label)
                        if item["query_id"] == query_id
                    ]
                    completed = {
                        int(item["repetition"]) for item in existing_query_records
                    }
                    terminal_measurement = next(
                        (
                            item
                            for item in existing_query_records
                            if not item["is_warmup"] and item.get("status") != "ok"
                        ),
                        None,
                    )
                    if terminal_measurement is not None:
                        print(
                            f"[{profile.name} {query_index}/{len(query_ids)}] "
                            f"{query_id}: resume terminal "
                            f"status={terminal_measurement['status']}",
                            flush=True,
                        )
                        continue
                    for repetition in range(total_repetitions):
                        result_key = f"{run_id}:{query_id}:{repetition}"
                        if (
                            repetition in completed
                            and result_key in episodes_csv.records
                        ):
                            print(
                                f"[{profile.name} {query_index}/{len(query_ids)}] "
                                f"{query_id} rep={repetition + 1}: resume",
                                flush=True,
                            )
                            continue

                        trace_name = safe_name(
                            f"{profile_label}.{query_id}.{repetition}.db.jsonl"
                        )
                        db_trace_host = runtime_host_dir / trace_name
                        db_trace_container = f"{runtime_container_dir}/{trace_name}"
                        db_trace_host.unlink(missing_ok=True)
                        cache_hit = False
                        cache_source = ""
                        cached_trajectory_hash = ""
                        cache_saved_wall_ms = 0.0
                        policy_events: list[dict[str, Any]] = []
                        db_events: list[dict[str, Any]] = []
                        execution = None
                        sql_lock_handle = None
                        expected_hash = baseline.get(query_id, {}).get("result_hash")
                        if server is not None and args.execution_cache in {
                            "read-only",
                            "read-write",
                        }:
                            cached_match, policy_offset = lookup_cached_execution(
                                store=store,
                                sql_hash=sql_hash,
                                expected_result_hash=expected_hash,
                                container=args.container,
                                server_url=server.action_url,
                                policy_log=server.policy_log_host,
                                policy_offset=policy_offset,
                                match_mode=args.cache_match_mode,
                            )
                            if cached_match is not None:
                                cache_hit = True
                                cache_source = cached_match["cache_source"]
                                cached_trajectory_hash = cached_match["trajectory_hash"]
                                cache_saved_wall_ms = cached_match[
                                    "cache_saved_wall_ms"
                                ]
                                policy_events = cached_match["policy_events"]
                                db_events = cached_match["db_events"]
                                execution = cached_match["execution"]
                        if execution is None and args.sql_execution_lock is not None:
                            print(
                                f"[{profile.name} {query_id}] waiting for SQL slot",
                                flush=True,
                            )
                            sql_lock_handle, sql_slot = acquire_sql_execution_slot(
                                args.sql_execution_lock,
                                args.sql_execution_slots,
                            )
                            print(
                                f"[{profile.name} {query_id}] acquired SQL "
                                f"slot {sql_slot + 1}/{args.sql_execution_slots}",
                                flush=True,
                            )
                            if server is not None and args.execution_cache in {
                                "read-only",
                                "read-write",
                            }:
                                cached_match, policy_offset = lookup_cached_execution(
                                    store=store,
                                    sql_hash=sql_hash,
                                    expected_result_hash=expected_hash,
                                    container=args.container,
                                    server_url=server.action_url,
                                    policy_log=server.policy_log_host,
                                    policy_offset=policy_offset,
                                    match_mode=args.cache_match_mode,
                                )
                                if cached_match is not None:
                                    cache_hit = True
                                    cache_source = cached_match["cache_source"]
                                    cached_trajectory_hash = cached_match[
                                        "trajectory_hash"
                                    ]
                                    cache_saved_wall_ms = cached_match[
                                        "cache_saved_wall_ms"
                                    ]
                                    policy_events = cached_match["policy_events"]
                                    db_events = cached_match["db_events"]
                                    execution = cached_match["execution"]
                        if execution is None:
                            policy_start = policy_offset
                            execution = run_query_once(
                                args,
                                sql=sql,
                                profile=profile,
                                timeout_ms=timeout_ms,
                                db_trace_container=db_trace_container,
                                server_url=(
                                    server.action_url if server is not None else None
                                ),
                            )
                            if execution["status"] != "ok":
                                execution["charged_wall_ms"] = failure_charge_ms
                            if server is not None:
                                (
                                    policy_events,
                                    policy_offset,
                                ) = read_jsonl_since(
                                    server.policy_log_host,
                                    policy_start,
                                )
                            db_events = load_jsonl(db_trace_host)
                        if execution["status"] != "ok":
                            # Cache hits and physical misses use the same
                            # PG-first timeout charge for the current run.
                            execution["charged_wall_ms"] = failure_charge_ms
                        validate_policy_state_contract(policy_events)
                        trajectory = semantic_policy_trajectory(policy_events)
                        trajectory_hash = semantic_trajectory_hash(trajectory)
                        if cache_hit and trajectory_hash != cached_trajectory_hash:
                            raise RuntimeError(
                                "cached policy replay produced a different "
                                "semantic trajectory hash"
                            )
                        db_total_ms = sum(
                            float((event.get("timing_ms") or {}).get("total") or 0.0)
                            for event in db_events
                        )
                        materialized_rows = sum(
                            int((event.get("materialized") or {}).get("rows") or 0)
                            for event in db_events
                            if event.get("phase") == "split"
                        )
                        materialized_bytes = sum(
                            int((event.get("materialized") or {}).get("bytes") or 0)
                            for event in db_events
                            if event.get("phase") == "split"
                        )
                        search_applied = any(
                            bool((event.get("action") or {}).get("search_applied"))
                            for event in db_events
                        )
                        lip_filters = sum(
                            int((event.get("action") or {}).get("lip_filters") or 0)
                            for event in db_events
                        )
                        aja_decided = sum(
                            int((event.get("aja") or {}).get("decided") or 0)
                            for event in db_events
                        )

                        status = execution["status"]
                        correctness_validation = (
                            "off" if args.correctness == "off" else "not_checked"
                        )
                        if (
                            status == "ok"
                            and not profile.is_postgres
                            and args.correctness != "off"
                            and query_id in baseline
                            and baseline[query_id].get("result_hash") is not None
                        ):
                            expected = baseline[query_id]
                            if expected["result_hash"] == execution["result_hash"]:
                                correctness_validation = "exact_hash"
                            else:
                                semantic = expected.get("semantic_validation") or {}
                                semantic_ok = False
                                if semantic.get("mode") == "full_result":
                                    validation_trace_name = safe_name(
                                        f"{profile_label}.{query_id}."
                                        f"{repetition}.validation.db.jsonl"
                                    )
                                    validation_trace_host = (
                                        runtime_host_dir / validation_trace_name
                                    )
                                    validation_trace_container = (
                                        f"{runtime_container_dir}/"
                                        f"{validation_trace_name}"
                                    )
                                    validation_trace_host.unlink(missing_ok=True)
                                    validation = run_query_once(
                                        args,
                                        sql=str(semantic["sql"]),
                                        profile=profile,
                                        timeout_ms=timeout_ms,
                                        db_trace_container=(validation_trace_container),
                                        server_url=(
                                            server.action_url
                                            if server is not None
                                            else None
                                        ),
                                    )
                                    if server is not None:
                                        (
                                            _discarded_validation_events,
                                            policy_offset,
                                        ) = read_jsonl_since(
                                            server.policy_log_host,
                                            policy_offset,
                                        )
                                    semantic_ok = (
                                        validation["status"] == "ok"
                                        and validation["result_hash"]
                                        == semantic.get("result_hash")
                                        and validation["result_rows"]
                                        == semantic.get("result_rows")
                                        and execution["result_rows"]
                                        == expected.get("result_rows")
                                    )
                                    correctness_validation = (
                                        "full_result"
                                        if semantic_ok
                                        else "full_result_failed"
                                    )
                                if not semantic_ok:
                                    status = "wrong_result"
                                    execution["charged_wall_ms"] = failure_charge_ms
                                    execution["error"] = (
                                        "result hash differs from PostgreSQL "
                                        "baseline and semantic validation "
                                        "did not pass"
                                    )
                                    if args.correctness == "strict":
                                        print(
                                            f"ERROR {profile.name} "
                                            f"{query_id}: "
                                            f"{execution['error']}",
                                            flush=True,
                                        )
                                elif args.correctness == "strict":
                                    print(
                                        f"[{profile.name} {query_id}] "
                                        "accepted nondeterministic LIMIT "
                                        "result after full-result validation",
                                        flush=True,
                                    )
                        elif status == "ok" and (
                            profile.is_postgres
                            or args.correctness == "off"
                            or query_id not in baseline
                            or baseline[query_id].get("result_hash") is None
                        ):
                            correctness_validation = (
                                "off" if args.correctness == "off" else "not_available"
                            )

                        is_warmup = repetition < args.warmups
                        episode_id = f"episode_{content_hash(result_key)[:24]}"
                        counts = ingest_benchmark_trajectory(
                            episode_id=episode_id,
                            environment_hash=profile_environment_hash,
                            implementation_version=implementation_version,
                            db_events=db_events,
                            policy_events=policy_events,
                            timeout_limit_ms=timeout_ms,
                            timeout_charged_ms=failure_charge_ms,
                            episode_status=status,
                            cache_hit=cache_hit,
                        )
                        if (
                            not profile.is_postgres
                            and not is_warmup
                            and server is not None
                            and status in {"ok", "timeout"}
                        ):
                            store.append_execution(
                                query_id=query_id,
                                sql_hash=sql_hash,
                                trajectory=counts["trajectory"],
                                db_events=db_events,
                                status=status,
                                first_runtime_ms=execution["client_wall_ms"],
                                charged_runtime_ms=execution["charged_wall_ms"],
                                timeout_limit_ms=timeout_ms,
                                action_config_hash=str(
                                    getattr(args, "action_config_hash", None)
                                    or content_hash(profile.to_dict())
                                ),
                                result_hash=execution["result_hash"],
                                result_rows=execution["result_rows"],
                                source_episode_id=episode_id,
                            )
                        if sql_lock_handle is not None:
                            release_sql_execution_slot(sql_lock_handle)
                        record = {
                            "result_key": result_key,
                            "experiment_id": args.experiment_id,
                            "run_id": run_id,
                            "profile": profile_label,
                            "workload": args.workload,
                            "protocol": args.protocol,
                            "fold": args.fold,
                            "role": args.role,
                            "query_id": query_id,
                            "repetition": repetition,
                            "is_warmup": is_warmup,
                            "status": status,
                            "client_wall_ms": execution["client_wall_ms"],
                            "charged_wall_ms": execution["charged_wall_ms"],
                            "db_total_ms": db_total_ms,
                            "pg_baseline_ms": pg_ms if pg_ms is not None else "",
                            "speedup": (
                                pg_ms / execution["charged_wall_ms"]
                                if pg_ms is not None
                                else ""
                            ),
                            "timeout_limit_ms": timeout_ms,
                            "result_hash": execution["result_hash"] or "",
                            "result_rows": (
                                execution["result_rows"]
                                if execution["result_rows"] is not None
                                else ""
                            ),
                            "correctness_validation": (correctness_validation),
                            "rounds": counts["rounds"],
                            "decisions": counts["decisions"],
                            "materialized_rows": materialized_rows,
                            "materialized_bytes": materialized_bytes,
                            "search_applied": search_applied,
                            "lip_filters": lip_filters,
                            "aja_decided": aja_decided,
                            "cache_hit": cache_hit,
                            "cache_source": cache_source,
                            "cache_saved_wall_ms": cache_saved_wall_ms,
                            "error": execution["error"] or "",
                        }
                        episodes_csv.append(record)
                        print(
                            f"[{profile_label} {query_index}/{len(query_ids)}] "
                            f"{query_id} rep={repetition + 1}/{total_repetitions} "
                            f"status={status} wall={execution['charged_wall_ms']:.1f}ms "
                            f"rounds={counts['rounds']} "
                            f"cache={'hit' if cache_hit else 'miss'}",
                            flush=True,
                        )
                        if status != "ok" and not is_warmup:
                            break

                    if profile.is_postgres:
                        pg_records = [
                            item
                            for item in episodes_csv.profile_records("pg")
                            if item["query_id"] == query_id
                        ]
                        query_summary = query_runtime_summary(pg_records)
                        if query_id in query_summary:
                            pg_query = query_summary[query_id]
                            if (
                                pg_query["timeouts"]
                                or pg_query["errors"]
                                or pg_query["wrong_results"]
                            ):
                                raise RuntimeError(
                                    f"PostgreSQL baseline failed for "
                                    f"{query_id}: {pg_query}"
                                )
                            hashes = {
                                item["result_hash"]
                                for item in pg_records
                                if item.get("status") == "ok"
                            }
                            if len(hashes) > 1:
                                raise RuntimeError(
                                    f"PostgreSQL result changed across runs for {query_id}"
                                )
                            baseline[query_id] = {
                                **query_summary[query_id],
                                "result_hash": next(iter(hashes)) if hashes else None,
                            }
                            write_json_atomic(baseline_path, baseline)

            records = episodes_csv.profile_records(profile_label)
            profile_summary = query_runtime_summary(records)
            missing_profile_queries = sorted(set(query_ids) - set(profile_summary))
            if missing_profile_queries:
                raise RuntimeError(
                    f"{profile_label} is missing completed records for "
                    f"{missing_profile_queries}"
                )
            all_summaries = (
                json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
                if (output_dir / "summary.json").is_file()
                else {}
            )
            all_summaries[profile_label] = {
                "run_id": run_id,
                "action_config": profile.to_dict(),
                "action_config_hash": getattr(
                    args,
                    "action_config_hash",
                    content_hash(profile.to_dict()),
                ),
                "profile_config_hash": content_hash(profile.to_dict()),
                "metrics": workload_metrics(
                    profile_summary,
                    baseline,
                    expected_query_ids=query_ids,
                ),
                "queries": profile_summary,
                "experience": {
                    "episodes": sum(
                        not bool(item.get("is_warmup")) for item in records
                    ),
                    "decisions": sum(
                        int(item.get("decisions") or 0)
                        for item in records
                        if not bool(item.get("is_warmup"))
                    ),
                },
                "execution_cache": {
                    **store.trajectory_cache_summary(),
                    "mode": args.execution_cache,
                    "match_mode": args.cache_match_mode,
                    "hits": sum(
                        bool(item.get("cache_hit"))
                        for item in records
                        if not item.get("is_warmup")
                    ),
                    "lookups": sum(1 for item in records if not item.get("is_warmup")),
                    "saved_wall_ms": sum(
                        float(item.get("cache_saved_wall_ms") or 0.0)
                        for item in records
                        if not item.get("is_warmup")
                    ),
                },
            }
            all_summaries[profile_label].update(all_summaries[profile_label]["metrics"])
            write_json_atomic(output_dir / "summary.json", all_summaries)
            write_result_tables(
                output_dir,
                episodes_csv,
                all_summaries,
                baseline,
            )
            print(
                f"[{profile.name}] WS="
                f"{all_summaries[profile_label]['workload_speedup']} "
                f"summary={output_dir / 'summary.json'}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
