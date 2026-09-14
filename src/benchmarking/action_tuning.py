#!/usr/bin/env python3
"""Calibrate and freeze independent NQO actions once per dataset."""
from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PGDB_ROOT = ROOT.parent / "pgdb"

from benchmarking.workloads import (  # noqa: E402
    split_folds,
    workload_query_ids,
)
from benchmarking.orchestration import (  # noqa: E402
    action_runner_command,
    run_command,
    sync_runtime,
)
from benchmarking.training_strategy import runtime_stratified_sample  # noqa: E402
from benchmarking.tuning_selection import select_candidate  # noqa: E402
from benchmarking.utils import utc_stamp, write_json_atomic  # noqa: E402

from experience.store import ExperienceStore  # noqa: E402
from optimization.actions import (  # noqa: E402
    ACTION_CONFIG_SCHEMA_VERSION,
    STATEMENT_TIMEOUT_MS,
    TIMEOUT_CHARGE_CAP_MS,
    TIMEOUT_CHARGE_FACTOR,
    action_config_hash,
    global_experience_path,
)
from optimization.decomposition_eligibility import (  # noqa: E402
    workload_supports_decomposition,
)

SELECTION_RULE = (
    "maximum correct charged-runtime workload speedup with timeout_rate<=20%"
)


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def comma_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]




def profile_summary(
    args: argparse.Namespace,
    *,
    experiment_id: str,
    profile: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    path = args.output_root / args.workload.lower() / experiment_id / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))[profile]
    queries = data["queries"]
    query_ids = sorted(queries)
    pg_total_ms = sum(
        float(baseline[query_id]["median_charged_ms"]) for query_id in query_ids
    )
    action_total_ms = sum(
        float(queries[query_id]["median_charged_ms"]) for query_id in query_ids
    )
    per_query_speedups = {
        query_id: (
            float(baseline[query_id]["median_charged_ms"])
            / float(queries[query_id]["median_charged_ms"])
        )
        for query_id in query_ids
        if float(queries[query_id]["median_charged_ms"]) > 0.0
    }
    ordered_speedups = sorted(per_query_speedups.items())
    best_query, best_speedup = max(
        ordered_speedups,
        key=lambda item: item[1],
        default=(None, None),
    )
    worst_query, worst_speedup = min(
        ordered_speedups,
        key=lambda item: item[1],
        default=(None, None),
    )
    return {
        "experiment_id": experiment_id,
        "profile": profile,
        "action_config_hash": data.get("action_config_hash"),
        "profile_config_hash": data.get("profile_config_hash"),
        "query_count": len(query_ids),
        "pg_total_ms": pg_total_ms,
        "action_total_ms": action_total_ms,
        "workload_speedup": (
            pg_total_ms / action_total_ms if action_total_ms > 0.0 else None
        ),
        "geometric_mean_speedup": (
            math.exp(
                sum(math.log(speedup) for speedup in per_query_speedups.values())
                / len(per_query_speedups)
            )
            if per_query_speedups
            else None
        ),
        "improved_queries": sum(
            speedup > 1.0 for speedup in per_query_speedups.values()
        ),
        "improved_pct": (
            100.0
            * sum(speedup > 1.0 for speedup in per_query_speedups.values())
            / len(query_ids)
            if query_ids
            else None
        ),
        "regressed_queries": sum(
            speedup < 1.0 for speedup in per_query_speedups.values()
        ),
        "best_query": best_query,
        "best_speedup": best_speedup,
        "worst_query": worst_query,
        "worst_speedup": worst_speedup,
        "timeouts": sum(int(item["timeouts"]) for item in queries.values()),
        "wrong_results": sum(int(item["wrong_results"]) for item in queries.values()),
        "errors": sum(int(item.get("errors", 0)) for item in queries.values()),
        "search_applied_queries": sum(
            bool(item.get("search_applied")) for item in queries.values()
        ),
        "search_application_pct": (
            100.0
            * sum(bool(item.get("search_applied")) for item in queries.values())
            / len(query_ids)
            if query_ids
            else None
        ),
        "lip_applied_queries": sum(
            int(item.get("lip_filters") or 0) > 0 for item in queries.values()
        ),
        "lip_application_pct": (
            100.0
            * sum(int(item.get("lip_filters") or 0) > 0 for item in queries.values())
            / len(query_ids)
            if query_ids
            else None
        ),
        "aja_applied_queries": sum(
            int(item.get("aja_decided") or 0) > 0 for item in queries.values()
        ),
        "aja_application_pct": (
            100.0
            * sum(int(item.get("aja_decided") or 0) > 0 for item in queries.values())
            / len(query_ids)
            if query_ids
            else None
        ),
        "action_applied_queries": sum(
            bool(item.get("action_applied")) for item in queries.values()
        ),
        "action_application_pct": (
            100.0
            * sum(bool(item.get("action_applied")) for item in queries.values())
            / len(query_ids)
            if query_ids
            else None
        ),
        "per_query_speedups": per_query_speedups,
        "queries": queries,
    }


def run_candidate(
    args: argparse.Namespace,
    *,
    master_id: str,
    candidate_id: str,
    profile: str,
    query_ids: list[str],
    role: str,
    experience_db: Path,
    baseline_path: Path,
    baseline: dict[str, Any],
    sched_alpha: float | None = None,
    sched_alpha_sequence: tuple[float, ...] | None = None,
    conservative_rows: int | None = None,
    aggressive_rows: int | None = None,
    aja_max_nestloop_cost_ratio_pct: int | None = None,
    aja_aggressive_max_nestloop_cost_ratio_pct: int | None = None,
    lip_build_rows: int | None = None,
    lip_selective_plan_rows: int | None = None,
    lip_build_selectivity_pct: int | None = None,
    lip_max_filters: int | None = None,
    search_max_rels: int | None = None,
    max_rounds: int | None = None,
    replay_group: str | None = None,
    reuse_master_id: str | None = None,
    warmups: int,
    measurements: int,
) -> dict[str, Any]:
    candidate_args = copy.copy(args)
    if sched_alpha is not None:
        candidate_args.initial_sched_alpha = sched_alpha
    if sched_alpha_sequence is not None:
        candidate_args.sched_alpha_sequence = ",".join(
            str(alpha) for alpha in sched_alpha_sequence
        )
    if conservative_rows is not None:
        candidate_args.aja_conservative_rows = conservative_rows
    if aggressive_rows is not None:
        candidate_args.aja_aggressive_rows = aggressive_rows
    if aja_max_nestloop_cost_ratio_pct is not None:
        candidate_args.aja_max_nestloop_cost_ratio_pct = aja_max_nestloop_cost_ratio_pct
    if aja_aggressive_max_nestloop_cost_ratio_pct is not None:
        candidate_args.aja_aggressive_max_nestloop_cost_ratio_pct = (
            aja_aggressive_max_nestloop_cost_ratio_pct
        )
    if lip_build_rows is not None:
        candidate_args.lip_max_build_relation_rows = lip_build_rows
    if lip_selective_plan_rows is not None:
        candidate_args.lip_selective_plan_rows = lip_selective_plan_rows
    if lip_build_selectivity_pct is not None:
        candidate_args.lip_max_build_selectivity_pct = lip_build_selectivity_pct
    if lip_max_filters is not None:
        candidate_args.lip_max_filters = lip_max_filters
    if search_max_rels is not None:
        candidate_args.search_max_rels = search_max_rels
    if max_rounds is not None:
        candidate_args.max_rounds = max_rounds
    experiment_id = f"{master_id}-{role}-{candidate_id}"
    if reuse_master_id is not None:
        reused_experiment_id = f"{reuse_master_id}-{role}-{candidate_id}"
        reused_summary = (
            args.output_root
            / args.workload.lower()
            / reused_experiment_id
            / "summary.json"
        )
        if not reused_summary.is_file():
            raise RuntimeError(
                "missing completed calibration summary for reuse: " f"{reused_summary}"
            )
        result = profile_summary(
            args,
            experiment_id=reused_experiment_id,
            profile=profile,
            baseline=baseline,
        )
        if sorted(result["queries"]) != sorted(query_ids):
            raise RuntimeError(
                f"{reused_experiment_id} query set does not match the "
                "current calibration sample"
            )
        result["reused_by_experiment_id"] = experiment_id
    else:
        run_command(
            action_runner_command(
                candidate_args,
                experiment_id=experiment_id,
                profile=profile,
                role=role,
                query_ids=query_ids,
                experience_db=experience_db,
                baseline_json=baseline_path,
                replay_group=replay_group,
                warmups=warmups,
                measurements=measurements,
                execution_cache=(
                    "read-write" if warmups == 0 and role == "train" else "off"
                ),
            )
        )
        result = profile_summary(
            args,
            experiment_id=experiment_id,
            profile=profile,
            baseline=baseline,
        )
    result["parameters"] = {
        "sched_alpha": sched_alpha,
        "sched_alpha_sequence": (
            list(sched_alpha_sequence)
            if sched_alpha_sequence is not None
            else None
        ),
        "aja_conservative_rows": conservative_rows,
        "aja_aggressive_rows": aggressive_rows,
        "aja_max_nestloop_cost_ratio_pct": (aja_max_nestloop_cost_ratio_pct),
        "aja_aggressive_max_nestloop_cost_ratio_pct": (
            aja_aggressive_max_nestloop_cost_ratio_pct
        ),
        "lip_max_build_relation_rows": lip_build_rows,
        "lip_selective_plan_rows": lip_selective_plan_rows,
        "lip_max_build_selectivity_pct": lip_build_selectivity_pct,
        "lip_max_filters": lip_max_filters,
        "search_max_rels": search_max_rels,
        "max_rounds": max_rounds,
    }
    return result


def candidate_run_parameters(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    parameters = candidate["parameters"]
    mapping = {
        "sched_alpha": "sched_alpha",
        "sched_alpha_sequence": "sched_alpha_sequence",
        "aja_conservative_rows": "conservative_rows",
        "aja_aggressive_rows": "aggressive_rows",
        "aja_max_nestloop_cost_ratio_pct": ("aja_max_nestloop_cost_ratio_pct"),
        "aja_aggressive_max_nestloop_cost_ratio_pct": (
            "aja_aggressive_max_nestloop_cost_ratio_pct"
        ),
        "lip_max_build_relation_rows": "lip_build_rows",
        "lip_selective_plan_rows": "lip_selective_plan_rows",
        "lip_max_build_selectivity_pct": "lip_build_selectivity_pct",
        "lip_max_filters": "lip_max_filters",
        "search_max_rels": "search_max_rels",
        "max_rounds": "max_rounds",
    }
    result = {}
    for source, target in mapping.items():
        value = parameters.get(source)
        if value is None:
            continue
        if source == "sched_alpha_sequence":
            value = tuple(value)
        result[target] = value
    return result


def annotate_standalone_summaries(
    args: argparse.Namespace,
    standalone_test: dict[str, dict[str, Any]],
    frozen_config_hash: str,
) -> None:
    """Attach the shared dataset config hash to every standalone artifact."""
    for result in standalone_test.values():
        result["action_config_hash"] = frozen_config_hash
        experiment_id = result["experiment_id"]
        profile = result["profile"]
        path = args.output_root / args.workload.lower() / experiment_id / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary[profile]["action_config_hash"] = frozen_config_hash
        write_json_atomic(path, summary)


def collect_prefix_replay(
    args: argparse.Namespace,
    *,
    master_id: str,
    query_ids: list[str],
    experience_db: Path,
    baseline_path: Path,
    baseline: dict[str, Any],
    alpha_values: list[float],
    selected_alpha: float,
    prefix_depths: list[int],
    branch_alpha_depths: list[int],
    replay_group: str,
    reuse_master_id: str | None = None,
    collect_core_prefix: bool = True,
) -> dict[str, Any]:
    prefix_replay: dict[str, Any] = {
        "alpha_counterfactual_depth_1": {},
        "selected_alpha_depths": {},
        "per_round_alpha_counterfactuals": {},
    }
    if collect_core_prefix:
        for alpha in alpha_values:
            result = run_candidate(
                args,
                master_id=master_id,
                candidate_id=f"replay-prefix-alpha-{alpha:.2f}-depth-1",
                profile="query_split",
                query_ids=query_ids,
                role="train",
                experience_db=experience_db,
                baseline_path=baseline_path,
                baseline=baseline,
                sched_alpha=alpha,
                max_rounds=1,
                replay_group=replay_group,
                reuse_master_id=reuse_master_id,
                warmups=args.replay_warmups,
                measurements=args.replay_measurements,
            )
            alpha_key = f"{alpha:.2f}"
            prefix_replay["alpha_counterfactual_depth_1"][alpha_key] = result
            if math.isclose(alpha, selected_alpha):
                prefix_replay["selected_alpha_depths"]["1"] = result
        for depth in prefix_depths:
            if depth == 1:
                continue
            prefix_replay["selected_alpha_depths"][str(depth)] = run_candidate(
                args,
                master_id=master_id,
                candidate_id=(
                    f"replay-prefix-alpha-{selected_alpha:.2f}-" f"depth-{depth}"
                ),
                profile="query_split",
                query_ids=query_ids,
                role="train",
                experience_db=experience_db,
                baseline_path=baseline_path,
                baseline=baseline,
                sched_alpha=selected_alpha,
                max_rounds=depth,
                replay_group=replay_group,
                reuse_master_id=reuse_master_id,
                warmups=args.replay_warmups,
                measurements=args.replay_measurements,
            )
    for depth in branch_alpha_depths:
        depth_results = {}
        for alpha in alpha_values:
            if math.isclose(alpha, selected_alpha):
                continue
            sequence = (selected_alpha,) * (depth - 1) + (alpha,)
            alpha_key = f"{alpha:.2f}"
            depth_results[alpha_key] = run_candidate(
                args,
                master_id=master_id,
                candidate_id=(
                    f"replay-prefix-sequence-depth-{depth}-" f"alpha-{alpha:.2f}"
                ),
                profile="query_split",
                query_ids=query_ids,
                role="train",
                experience_db=experience_db,
                baseline_path=baseline_path,
                baseline=baseline,
                sched_alpha=selected_alpha,
                sched_alpha_sequence=sequence,
                max_rounds=depth,
                replay_group=replay_group,
                reuse_master_id=reuse_master_id,
                warmups=args.replay_warmups,
                measurements=args.replay_measurements,
            )
        prefix_replay["per_round_alpha_counterfactuals"][str(depth)] = depth_results
    return prefix_replay




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nqo-benchmark tune")
    parser.add_argument(
        "--workload",
        choices=("JOB", "STACK", "TPCH"),
        default="JOB",
    )
    parser.add_argument(
        "--protocol",
        choices=("base_query", "leave_one_out", "random"),
        default="random",
    )
    parser.add_argument("--fold", choices=("a", "b", "c"), default="a")
    parser.add_argument(
        "--calibration-scope",
        choices=("dataset", "fold"),
        default="dataset",
        help=(
            "dataset calibrates one implementation config shared by every "
            "protocol/fold; fold is retained for targeted diagnostics"
        ),
    )
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--reuse-tuning-experiment-id",
        help=(
            "reuse completed grid and prefix summaries from a compatible "
            "calibration experiment; selected-action replays and full "
            "tests are always rerun"
        ),
    )
    parser.add_argument(
        "--reuse-experience-db",
        type=Path,
        help=(
            "Experience Store used by the reused calibration; defaults to "
            "that experiment's runtime directory"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "runs",
    )
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--experience-db", type=Path)
    parser.add_argument(
        "--action-config-out",
        type=Path,
        help="path for the immutable dataset-level Action configuration",
    )
    parser.add_argument(
        "--pgdb-root",
        type=Path,
        default=DEFAULT_PGDB_ROOT,
    )
    parser.add_argument("--container", default="pgdb_dev_opt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--ai-port", type=int, default=18090)
    parser.add_argument("--tune-warmups", type=int, default=0)
    parser.add_argument("--tune-measurements", type=int, default=1)
    parser.add_argument("--test-warmups", type=int, default=2)
    parser.add_argument("--test-measurements", type=int, default=1)
    parser.add_argument("--replay-warmups", type=int, default=0)
    parser.add_argument("--replay-measurements", type=int, default=1)
    parser.add_argument("--query-id", action="append")
    parser.add_argument(
        "--prefix-replay-only",
        action="store_true",
        help="collect reusable alpha/depth prefix experience and stop",
    )
    parser.add_argument(
        "--branch-alpha-only",
        action="store_true",
        help=(
            "with --prefix-replay-only, skip depth-1/core-prefix reruns and "
            "collect only per-round alpha branches"
        ),
    )
    parser.add_argument("--prefix-replay-group")
    parser.add_argument("--tune-limit", type=int, default=12)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--alpha-grid", default="0.5")
    parser.add_argument(
        "--collect-prefix-replay",
        action="store_true",
        help="optionally collect decomposition counterfactuals after tuning",
    )
    parser.add_argument(
        "--prefix-depth-grid",
        default="1,2,3,4,5,6",
        help=(
            "split-depth caps used to collect one-step high/select replay "
            "after alpha selection"
        ),
    )
    parser.add_argument(
        "--branch-alpha-depth-grid",
        default="2,3",
        help=(
            "depths where the selected prefix is held fixed and only the "
            "current round's alpha is varied"
        ),
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help="deprecated compatibility option; all fold train SQL is tuned",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lip-build-row-grid", default="10000,100000,500000")
    parser.add_argument("--lip-selectivity-grid", default="1,5,10,25")
    parser.add_argument(
        "--aja-threshold-grid",
        default="1,100,1000,5000,10000,100000,500000",
    )
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--search-max-rels", type=int, default=12)
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
    parser.add_argument("--initial-sched-alpha", type=float, default=0.5)
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
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--model-device", default="cpu")
    args = parser.parse_args(argv)
    if args.statement_timeout_ms != STATEMENT_TIMEOUT_MS:
        parser.error(
            f"online experiments require "
            f"--statement-timeout-ms={STATEMENT_TIMEOUT_MS}"
        )
    if args.workload == "TPCH" and args.protocol != "random":
        parser.error("TPCH currently defines only the random protocol")
    for label in ("tune", "test", "replay"):
        warmups = getattr(args, f"{label}_warmups")
        measurements = getattr(args, f"{label}_measurements")
        if warmups < 0 or measurements != 1 or warmups + measurements > 3:
            parser.error(
                f"{label} requires one measurement and at most three "
                "total executions"
            )
    if args.branch_alpha_only and not args.prefix_replay_only:
        parser.error("--branch-alpha-only requires --prefix-replay-only")
    if args.reuse_experience_db is not None and args.reuse_tuning_experiment_id is None:
        parser.error("--reuse-experience-db requires " "--reuse-tuning-experiment-id")
    prefix_depths = list(dict.fromkeys(comma_ints(args.prefix_depth_grid)))
    if not prefix_depths or any(depth < 1 for depth in prefix_depths):
        parser.error("--prefix-depth-grid must contain positive integers")
    branch_alpha_depths = list(dict.fromkeys(comma_ints(args.branch_alpha_depth_grid)))
    if any(depth < 2 for depth in branch_alpha_depths):
        parser.error("--branch-alpha-depth-grid values must be at least 2")
    args.output_root = args.output_root.resolve()
    args.pgdb_root = args.pgdb_root.resolve()
    args.runtime_project = args.pgdb_root / ".nqo_runtime" / "nqo"
    sync_runtime(args.pgdb_root, args.runtime_project)

    master_id = args.experiment_id or (
        f"{args.workload.lower()}-actions-{args.protocol}-" f"{args.fold}-{utc_stamp()}"
    )
    result_dir = args.output_root / args.workload.lower() / master_id
    result_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.pgdb_root / ".nqo_runtime" / "benchmark" / master_id
    runtime_dir.mkdir(parents=True, exist_ok=True)
    experience_db = (
        args.experience_db.resolve()
        if args.experience_db is not None
        else global_experience_path(args.pgdb_root, args.workload)
    )
    experience_db.parent.mkdir(parents=True, exist_ok=True)
    reused_experience = None
    if args.reuse_tuning_experiment_id is not None:
        reused_experience_db = (
            args.reuse_experience_db.resolve()
            if args.reuse_experience_db is not None
            else global_experience_path(args.pgdb_root, args.workload)
        )
        if reused_experience_db.resolve() == experience_db.resolve():
            reused_experience = {
                "source": str(reused_experience_db),
                "same_store": True,
                "inserted": {},
            }
        else:
            with ExperienceStore(experience_db) as store:
                reused_experience = {
                    "source": str(reused_experience_db),
                    "same_store": False,
                    "inserted": store.merge_from(reused_experience_db),
                }
        experience_db.chmod(0o666)

    if args.calibration_scope == "dataset":
        train_query_ids = workload_query_ids(args.workload)
        test_query_ids = workload_query_ids(args.workload)
    else:
        fold = split_folds(args.workload, args.protocol)[f"{args.protocol}_{args.fold}"]
        train_query_ids = list(fold["train"])
        test_query_ids = list(fold["test"])
    validation_query_ids: list[str] = []

    if args.baseline_json is None:
        baseline_id = f"{master_id}-baseline"
        run_command(
            action_runner_command(
                args,
                experiment_id=baseline_id,
                profile="pg",
                role="all",
                query_ids=workload_query_ids(args.workload),
                experience_db=experience_db,
                warmups=2,
                measurements=1,
            )
        )
        baseline_path = (
            args.output_root / args.workload.lower() / baseline_id / "baseline.json"
        )
    else:
        baseline_path = args.baseline_json.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    missing_baselines = sorted(set(workload_query_ids(args.workload)) - set(baseline))
    if missing_baselines:
        raise RuntimeError(
            f"baseline is missing {len(missing_baselines)} workload queries"
        )
    if args.query_id:
        requested = list(dict.fromkeys(args.query_id))
        unknown = sorted(set(requested) - set(workload_query_ids(args.workload)))
        if unknown:
            parser.error(f"unknown workload query ids: {unknown}")
        train_query_ids = requested
    else:
        train_query_ids = runtime_stratified_sample(
            train_query_ids,
            baseline,
            args.tune_limit,
        )
    test_query_ids = runtime_stratified_sample(
        test_query_ids,
        baseline,
        args.test_limit,
    )

    groups: dict[str, list[dict[str, Any]]] = {
        "query_split": [],
        "split_search": [],
        "top5": [],
        "top10": [],
        "lip_full": [],
        "lip_selective": [],
        "aja_conservative": [],
        "aja_aggressive": [],
    }
    reused_group_id = args.reuse_tuning_experiment_id or master_id
    decomposition_replay_group = f"{reused_group_id}:decomposition-grid"
    selected_replay_group = f"{master_id}:selected-actions"
    prefix_replay_group = f"{reused_group_id}:decomposition-prefix"
    split_supported = workload_supports_decomposition(args.workload)
    alpha_values = list(dict.fromkeys(comma_floats(args.alpha_grid)))
    if not alpha_values or any(alpha < 0.0 or alpha > 1.0 for alpha in alpha_values):
        parser.error("--alpha-grid values must be in [0, 1]")
    if not args.prefix_replay_only and alpha_values != [0.5]:
        parser.error("standalone Query Split is fixed at --alpha-grid=0.5")
    if args.prefix_replay_only:
        selected_alpha = float(args.initial_sched_alpha)
        if not any(math.isclose(alpha, selected_alpha) for alpha in alpha_values):
            parser.error(
                "--initial-sched-alpha must occur in --alpha-grid "
                "for --prefix-replay-only"
            )
        replay_group = args.prefix_replay_group or prefix_replay_group
        prefix_replay = collect_prefix_replay(
            args,
            master_id=master_id,
            query_ids=train_query_ids,
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
            alpha_values=alpha_values,
            selected_alpha=selected_alpha,
            prefix_depths=prefix_depths,
            branch_alpha_depths=branch_alpha_depths,
            replay_group=replay_group,
            reuse_master_id=args.reuse_tuning_experiment_id,
            collect_core_prefix=not args.branch_alpha_only,
        )
        report = {
            "workload": args.workload,
            "protocol": args.protocol,
            "fold": args.fold,
            "query_ids": train_query_ids,
            "selected_alpha": selected_alpha,
            "alpha_values": alpha_values,
            "prefix_depths": prefix_depths,
            "branch_alpha_depths": branch_alpha_depths,
            "branch_alpha_only": args.branch_alpha_only,
            "replay_group": replay_group,
            "experience_db": str(experience_db),
            "reused_experience": reused_experience,
            "prefix_replay": prefix_replay,
        }
        report_path = result_dir / "prefix_replay_report.json"
        write_json_atomic(report_path, report)
        print(json.dumps({"report": str(report_path), **report}))
        return 0
    if split_supported:
        groups["query_split"].append(
            run_candidate(
                args,
                master_id=master_id,
                candidate_id="split-alpha-0.50",
                profile="query_split",
                query_ids=train_query_ids,
                role="train",
                experience_db=experience_db,
                baseline_path=baseline_path,
                baseline=baseline,
                sched_alpha=0.5,
                replay_group=decomposition_replay_group,
                reuse_master_id=args.reuse_tuning_experiment_id,
                warmups=args.tune_warmups,
                measurements=args.tune_measurements,
            )
        )
    for profile in ("split_search", "top5", "top10"):
        groups[profile].append(
            run_candidate(
                args,
                master_id=master_id,
                candidate_id=f"{profile}-max-rels-{args.search_max_rels}",
                profile=profile,
                query_ids=train_query_ids,
                role="train",
                experience_db=experience_db,
                baseline_path=baseline_path,
                baseline=baseline,
                search_max_rels=args.search_max_rels,
                reuse_master_id=args.reuse_tuning_experiment_id,
                warmups=args.tune_warmups,
                measurements=args.tune_measurements,
            )
        )
    for build_rows in comma_ints(args.lip_build_row_grid):
        groups["lip_full"].append(
            run_candidate(
                args,
                master_id=master_id,
                candidate_id=f"lip_full-build-{build_rows}",
                profile="lip_full",
                query_ids=train_query_ids,
                role="train",
                experience_db=experience_db,
                baseline_path=baseline_path,
                baseline=baseline,
                lip_build_rows=build_rows,
                lip_max_filters=args.lip_max_filters,
                reuse_master_id=args.reuse_tuning_experiment_id,
                warmups=args.tune_warmups,
                measurements=args.tune_measurements,
            )
        )
        for selectivity_pct in comma_ints(args.lip_selectivity_grid):
            groups["lip_selective"].append(
                run_candidate(
                    args,
                    master_id=master_id,
                    candidate_id=(
                        "lip_selective-build-"
                        f"{build_rows}-selectivity-{selectivity_pct}"
                    ),
                    profile="lip_selective",
                    query_ids=train_query_ids,
                    role="train",
                    experience_db=experience_db,
                    baseline_path=baseline_path,
                    baseline=baseline,
                    lip_build_rows=build_rows,
                    lip_selective_plan_rows=args.lip_selective_plan_rows,
                    lip_build_selectivity_pct=selectivity_pct,
                    lip_max_filters=args.lip_max_filters,
                    reuse_master_id=args.reuse_tuning_experiment_id,
                    warmups=args.tune_warmups,
                    measurements=args.tune_measurements,
                )
            )
    for threshold in comma_ints(args.aja_threshold_grid):
        for profile in ("aja_conservative", "aja_aggressive"):
            groups[profile].append(
                run_candidate(
                    args,
                    master_id=master_id,
                    candidate_id=f"{profile}-{threshold}",
                    profile=profile,
                    query_ids=train_query_ids,
                    role="train",
                    experience_db=experience_db,
                    baseline_path=baseline_path,
                    baseline=baseline,
                    conservative_rows=(
                        threshold if profile == "aja_conservative" else None
                    ),
                    aggressive_rows=(
                        threshold if profile == "aja_aggressive" else None
                    ),
                    aja_max_nestloop_cost_ratio_pct=(
                        args.aja_max_nestloop_cost_ratio_pct
                    ),
                    aja_aggressive_max_nestloop_cost_ratio_pct=(
                        args.aja_aggressive_max_nestloop_cost_ratio_pct
                    ),
                    reuse_master_id=args.reuse_tuning_experiment_id,
                    warmups=args.tune_warmups,
                    measurements=args.tune_measurements,
                )
            )

    selected = {
        "split_search": select_candidate(
            groups["split_search"],
            allow_no_application=True,
        ),
        "top5": select_candidate(
            groups["top5"],
            allow_no_application=True,
        ),
        "top10": select_candidate(
            groups["top10"],
            allow_no_application=True,
        ),
        "lip_full": select_candidate(
            groups["lip_full"],
            allow_no_application=True,
        ),
        "lip_selective": select_candidate(
            groups["lip_selective"],
            allow_no_application=True,
        ),
        "aja_conservative": select_candidate(
            groups["aja_conservative"],
            allow_no_application=True,
        ),
        "aja_aggressive": select_candidate(
            groups["aja_aggressive"],
            allow_no_application=True,
        ),
    }
    if split_supported:
        selected = {
            "query_split": select_candidate(groups["query_split"]),
            **selected,
        }
    best_search = max(
        (selected[name] for name in ("split_search", "top5", "top10")),
        key=lambda candidate: float(candidate["workload_speedup"]),
    )
    best_lip = max(
        (selected[name] for name in ("lip_full", "lip_selective")),
        key=lambda candidate: float(candidate["workload_speedup"]),
    )
    search_parameters = best_search["parameters"]
    lip_parameters = best_lip["parameters"]
    selective_parameters = selected["lip_selective"]["parameters"]
    selected_parameters = {
        "sched_alpha": 0.5,
        "search_max_rels": search_parameters["search_max_rels"],
        "lip_max_build_relation_rows": (lip_parameters["lip_max_build_relation_rows"]),
        "lip_max_build_selectivity_pct": (
            selective_parameters["lip_max_build_selectivity_pct"]
        ),
        "aja_conservative_rows": selected["aja_conservative"]["parameters"][
            "aja_conservative_rows"
        ],
        "aja_aggressive_rows": selected["aja_aggressive"]["parameters"][
            "aja_aggressive_rows"
        ],
        "aja_max_nestloop_cost_ratio_pct": (args.aja_max_nestloop_cost_ratio_pct),
        "aja_aggressive_max_nestloop_cost_ratio_pct": (
            args.aja_aggressive_max_nestloop_cost_ratio_pct
        ),
        "lip_selective_plan_rows": args.lip_selective_plan_rows,
        "lip_max_filters": args.lip_max_filters,
        "lip_min_probe_ratio": args.lip_min_probe_ratio,
        "max_rounds": args.max_rounds,
        "search_exact_cardinality": False,
    }
    selected_profile_parameters = {
        mechanism: candidate["parameters"] for mechanism, candidate in selected.items()
    }
    prefix_replay = {}
    if split_supported and args.collect_prefix_replay:
        prefix_replay = collect_prefix_replay(
            args,
            master_id=master_id,
            query_ids=train_query_ids,
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
            alpha_values=alpha_values,
            selected_alpha=0.5,
            prefix_depths=prefix_depths,
            branch_alpha_depths=branch_alpha_depths,
            replay_group=prefix_replay_group,
            reuse_master_id=args.reuse_tuning_experiment_id,
        )
    selected_train_replay = {
        "nqo_none": run_candidate(
            args,
            master_id=master_id,
            candidate_id="replay-nqo-none",
            profile="nqo_none",
            query_ids=train_query_ids,
            role="train",
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
            replay_group=selected_replay_group,
            warmups=args.replay_warmups,
            measurements=args.replay_measurements,
        )
    }
    for mechanism, candidate in selected.items():
        selected_train_replay[mechanism] = run_candidate(
            args,
            master_id=master_id,
            candidate_id=f"replay-selected-{mechanism}",
            profile=candidate["profile"],
            query_ids=train_query_ids,
            role="train",
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
            **candidate_run_parameters(candidate),
            replay_group=selected_replay_group,
            warmups=args.replay_warmups,
            measurements=args.replay_measurements,
        )
    standalone_test = {
        "nqo_none": run_candidate(
            args,
            master_id=master_id,
            candidate_id="nqo-none",
            profile="nqo_none",
            query_ids=test_query_ids,
            role="test",
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
            warmups=args.test_warmups,
            measurements=args.test_measurements,
        )
    }
    for mechanism, candidate in selected.items():
        standalone_test[mechanism] = run_candidate(
            args,
            master_id=master_id,
            candidate_id=f"selected-{mechanism}",
            profile=candidate["profile"],
            query_ids=test_query_ids,
            role="test",
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
            **candidate_run_parameters(candidate),
            warmups=args.test_warmups,
            measurements=args.test_measurements,
        )

    report = {
        "experiment_id": master_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": args.workload,
        "protocol": args.protocol,
        "fold": args.fold,
        "calibration_scope": args.calibration_scope,
        "train_query_ids": train_query_ids,
        "validation_query_ids": validation_query_ids,
        "tuning_sample": {
            "method": "runtime_stratified",
            "limit": args.tune_limit,
        },
        "test_query_ids": test_query_ids,
        "selection_rule": SELECTION_RULE,
        "candidates": groups,
        "selected": selected,
        "selected_parameters": selected_parameters,
        "parameter_selection": "best valid setting per Action variant",
        "selected_profile_parameters": selected_profile_parameters,
        "selected_train_replay": selected_train_replay,
        "prefix_replay": prefix_replay,
        "replay_groups": [
            *([decomposition_replay_group] if split_supported else []),
            selected_replay_group,
            *([prefix_replay_group] if prefix_replay else []),
        ],
        "standalone_test": standalone_test,
        "baseline": str(baseline_path),
        "experience_db": str(experience_db),
        "reused_experience": reused_experience,
    }
    report_path = result_dir / "action_tuning_report.json"
    action_config = {
        "schema_version": ACTION_CONFIG_SCHEMA_VERSION,
        "status": "frozen",
        "workload": args.workload,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": selected_parameters,
        "profile_parameters": selected_profile_parameters,
        "execution_protocol": {
            "warmups": args.test_warmups,
            "measurements": args.test_measurements,
            "fresh_session_per_execution": True,
            "official_sample": "last_execution",
            "failure_charge": "min(360s,5*Tpg)",
            "statement_timeout_ms": args.statement_timeout_ms,
            "timeout_charge_factor": args.timeout_charge_factor,
            "timeout_charge_cap_ms": args.timeout_charge_cap_ms,
        },
        "calibration": {
            "experiment_id": master_id,
            "scope": args.calibration_scope,
            "query_ids": train_query_ids,
            "selection_rule": report["selection_rule"],
            "report": str(report_path),
        },
        "standalone_actions": standalone_test,
        "replay": {
            "experience_db": str(experience_db),
            "groups": report["replay_groups"],
        },
        "baseline": str(baseline_path),
    }
    action_config["config_hash"] = action_config_hash(action_config)
    action_config_path = (
        args.action_config_out.resolve()
        if args.action_config_out is not None
        else result_dir / "frozen_action_config.json"
    )
    if action_config_path.is_file():
        existing = json.loads(action_config_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != action_config["config_hash"]:
            raise RuntimeError(
                f"refusing to replace a different frozen Action config: "
                f"{action_config_path}"
            )
    annotate_standalone_summaries(
        args,
        standalone_test,
        action_config["config_hash"],
    )
    write_json_atomic(report_path, report)
    write_json_atomic(action_config_path, action_config)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "action_config": str(action_config_path),
                "action_config_hash": action_config["config_hash"],
                "selected_train_ws": {
                    key: value["workload_speedup"] for key, value in selected.items()
                },
                "standalone_test_ws": {
                    key: value["workload_speedup"]
                    for key, value in standalone_test.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
