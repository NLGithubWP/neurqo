#!/usr/bin/env python3
"""Run a resumable online NQO train/validate/test experiment."""
from __future__ import annotations

import argparse
import csv
import fcntl
import math
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PGDB_ROOT = ROOT.parent / "pgdb"

from benchmarking.workloads import (  # noqa: E402
    WORKLOAD_DATABASES,
    split_folds,
    workload_query_ids,
)
from benchmarking.orchestration import (  # noqa: E402
    action_runner_command,
    host_to_container,
    run_command,
    runner_run_id,
    sync_runtime,
)
from benchmarking.training_strategy import (  # noqa: E402
    INDEPENDENT_ACTION_PROFILES,
    build_coverage_snapshot,
    coverage_iteration_query_batch,
    early_stopping_counter,
    runtime_stratified_sample,
    select_checkpoint_finalists,
    select_initial_policy_profile,
    stochastic_heads_for_iteration,
)

from database.catalog import (  # noqa: E402
    read_postgres_catalog,
    write_catalog_snapshot,
)

from experience.store import (  # noqa: E402
    ExperienceStore,
    content_hash,
)
from optimization.actions import (  # noqa: E402
    STATEMENT_TIMEOUT_MS,
    TIMEOUT_CHARGE_CAP_MS,
    TIMEOUT_CHARGE_FACTOR,
    apply_action_config,
    global_experience_path,
    load_action_config,
)
from optimization.decomposition_eligibility import (  # noqa: E402
    workload_supports_decomposition,
)
from benchmarking.utils import utc_stamp, write_json_atomic  # noqa: E402

TRAINER_MODULE = "training.experience_trainer"
TRAINING_STOP_FILE = ROOT / ".local" / "STOP_ALL_TRAINING"
FIRST_RUNTIME_SEMANTICS = "pg_first_over_nqo_first"
FORMAL_RUNTIME_SEMANTICS = "pg_third_over_nqo_third"




def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from an append-only JSONL file."""
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]






def trainer_command(
    args: argparse.Namespace,
    *,
    experience_db: Path,
    output_checkpoint: Path,
    output_policy_version: str,
    base_checkpoint: Path | None = None,
    source_policy_version: str | None = None,
    initialize_only: bool = False,
    replay_only: bool = False,
) -> list[str]:
    runtime_project = args.runtime_project
    command = [
        "docker",
        "exec",
        "-e",
        f"PYTHONPATH={host_to_container(runtime_project / 'src', args.pgdb_root)}",
        "-w",
        host_to_container(runtime_project, args.pgdb_root),
        args.trainer_container,
        "python3",
        "-m",
        TRAINER_MODULE,
        "--experience-db",
        host_to_container(experience_db, args.pgdb_root),
        "--catalog-path",
        host_to_container(args.catalog_snapshot, args.pgdb_root),
        "--workload",
        args.workload.lower(),
        "--output",
        host_to_container(output_checkpoint, args.pgdb_root),
        "--policy-version",
        output_policy_version,
        "--hidden",
        str(args.hidden),
        "--device",
        args.trainer_device,
        "--torch-threads",
        str(args.trainer_torch_threads),
        "--lr",
        str(args.learning_rate),
        "--epochs",
        str(args.ppo_epochs),
        "--batch-size",
        str(args.ppo_batch_size),
        "--entropy",
        str(args.entropy),
        "--gamma",
        str(args.gamma),
        "--lambda-gae",
        str(args.lambda_gae),
        "--initial-sched-alpha",
        str(args.initial_sched_alpha),
        "--initial-action-bias",
        str(args.initial_action_bias),
        "--initial-policy-profile",
        args.resolved_initial_policy_profile,
        "--reward-scale-ms",
        str(args.reward_scale_ms),
        "--reward-clip",
        str(args.reward_clip),
        "--action-ablation",
        args.action_ablation,
        "--state-ablation",
        args.state_ablation,
        "--split-protocol",
        args.protocol,
        "--fold",
        args.fold,
        "--replay-epochs",
        str(args.replay_epochs),
        "--replay-batch-size",
        str(args.replay_batch_size),
        "--replay-minimum-samples",
        str(args.replay_minimum_samples),
        "--replay-importance-power",
        str(args.replay_importance_power),
        "--sched-replay-temperature",
        str(args.sched_replay_temperature),
        "--replay-action-cost-temperature",
        str(args.replay_action_cost_temperature),
        "--seed",
        str(args.seed),
    ]
    if args.replay_action_cost_regression:
        command.append("--replay-action-cost-regression")
    for phase in ("dec", "sched", "enum", "adapt"):
        phase_epochs = getattr(args, f"{phase}_replay_epochs")
        if phase_epochs is not None:
            command.extend([f"--{phase}-replay-epochs", str(phase_epochs)])
        phase_importance_power = getattr(args, f"{phase}_replay_importance_power")
        if phase_importance_power is not None:
            command.extend(
                [
                    f"--{phase}-replay-importance-power",
                    str(phase_importance_power),
                ]
            )
    for query_id in args.training_query_ids:
        command.extend(["--replay-query-id", query_id])
    for replay_group in args.replay_groups:
        command.extend(["--replay-group", replay_group])
    if args.staged_independent_action_summary is not None and (
        not args.bootstrap_only_independent_prior or replay_only
    ):
        command.extend(
            [
                "--independent-action-summary",
                host_to_container(
                    args.staged_independent_action_summary,
                    args.pgdb_root,
                ),
                "--independent-prior-epochs",
                str(args.independent_prior_epochs),
                "--independent-prior-temperature",
                str(args.independent_prior_temperature),
                "--independent-prior-mode",
                args.independent_prior_mode,
                "--independent-crossfit-confidence-z",
                str(args.independent_crossfit_confidence_z),
                "--independent-crossfit-trees",
                str(args.independent_crossfit_trees),
            ]
        )
        if args.independent_prior_cost_regression:
            command.append("--independent-prior-cost-regression")
        if args.residual_split_prior_epochs > 0:
            command.extend(
                [
                    "--residual-split-prior-epochs",
                    str(args.residual_split_prior_epochs),
                    "--residual-split-root-objective",
                    args.residual_split_root_objective,
                ]
            )
        for phase in ("dec", "sched", "enum", "adapt"):
            phase_epochs = getattr(args, f"independent_{phase}_prior_epochs")
            if phase_epochs is not None:
                command.extend(
                    [f"--independent-{phase}-prior-epochs", str(phase_epochs)]
                )
    if args.disable_runtime_replay:
        command.append("--disable-runtime-replay")
    if base_checkpoint is not None:
        command.extend(
            [
                "--base-checkpoint",
                host_to_container(base_checkpoint, args.pgdb_root),
            ]
        )
    if source_policy_version is not None:
        command.extend(["--source-policy-version", source_policy_version])
    if initialize_only:
        command.append("--initialize-only")
    if replay_only:
        command.append("--replay-only")
    return command


def bootstrap_fixed_replay_enabled(args: argparse.Namespace) -> bool:
    phase_epochs = [
        (
            args.replay_epochs
            if getattr(args, f"{phase}_replay_epochs") is None
            else getattr(args, f"{phase}_replay_epochs")
        )
        for phase in ("dec", "sched", "enum", "adapt")
    ]
    runtime_bootstrap = bool(
        args.fixed_replay_group
        and not args.disable_runtime_replay
        and any(epochs > 0 for epochs in phase_epochs)
    )
    independent_bootstrap = bool(
        getattr(args, "independent_action_summary", None) is not None
        and any(
            (
                getattr(args, "independent_prior_epochs", 0)
                if getattr(args, f"independent_{phase}_prior_epochs", None) is None
                else getattr(args, f"independent_{phase}_prior_epochs")
            )
            > 0
            for phase in ("dec", "sched", "enum", "adapt")
        )
    )
    return bool(
        not args.disable_bootstrap_fixed_replay
        and (runtime_bootstrap or independent_bootstrap)
    )


def load_profile_summary(
    args: argparse.Namespace,
    *,
    experiment_id: str,
    policy_version: str,
    baseline: dict[str, Any],
    runtime_semantics: str = FIRST_RUNTIME_SEMANTICS,
) -> dict[str, Any]:
    path = args.output_root / args.workload.lower() / experiment_id / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    profile = summary[f"learned@{policy_version}"]
    queries = profile["queries"]
    query_ids = sorted(queries)
    pg_total_ms = sum(
        float(baseline[query_id]["median_charged_ms"]) for query_id in query_ids
    )
    learned_total_ms = sum(
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
    workload_speedup_first = (
        pg_total_ms / learned_total_ms if learned_total_ms > 0.0 else None
    )
    geometric_mean_speedup_first = (
        math.exp(
            sum(math.log(value) for value in per_query_speedups.values())
            / len(per_query_speedups)
        )
        if per_query_speedups
        else None
    )
    improved_queries_first = sum(
        value > 1.0 for value in per_query_speedups.values()
    )
    improved_pct_first = (
        100.0 * improved_queries_first / len(query_ids) if query_ids else None
    )
    result = {
        "experiment_id": experiment_id,
        "runtime_semantics": runtime_semantics,
        "query_count": len(query_ids),
        "pg_total_ms": pg_total_ms,
        "learned_total_ms": learned_total_ms,
        # Keep the legacy names as aliases so aggregation code remains valid.
        "workload_speedup": workload_speedup_first,
        "geometric_mean_speedup": geometric_mean_speedup_first,
        "improved_queries": improved_queries_first,
        "improved_pct": improved_pct_first,
        "per_query_speedups": per_query_speedups,
        "timeouts": sum(int(item["timeouts"]) for item in queries.values()),
        "wrong_results": sum(int(item["wrong_results"]) for item in queries.values()),
        "errors": sum(int(item.get("errors", 0)) for item in queries.values()),
        "queries": queries,
    }
    if runtime_semantics == FIRST_RUNTIME_SEMANTICS:
        result.update(
            {
                "workload_speedup_first": workload_speedup_first,
                "geometric_mean_speedup_first": geometric_mean_speedup_first,
                "improved_queries_first": improved_queries_first,
                "improved_pct_first": improved_pct_first,
            }
        )
    return result


def load_first_pg_runtimes(
    episodes_path: Path,
    *,
    expected_query_ids: Iterable[str],
    workload: str,
) -> dict[str, float]:
    """Load PostgreSQL runtimes from an episode or benchmark-result CSV."""
    if not episodes_path.is_file():
        raise FileNotFoundError(
            "first-run evaluation requires the PostgreSQL episodes CSV: "
            f"{episodes_path}"
        )
    first: dict[str, float] = {}
    with episodes_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        benchmark_schema = {"dataset", "method", "runtime_ms", "sql_path"}.issubset(
            reader.fieldnames or ()
        )
        for row in reader:
            if benchmark_schema:
                if (
                    row.get("dataset", "").upper() != workload.upper()
                    or row.get("method") != "PostgreSQL"
                ):
                    continue
                query_id = Path(row["sql_path"]).stem
                runtime_ms = float(row["runtime_ms"])
            else:
                if row.get("profile") != "pg" or int(row["repetition"]) != 0:
                    continue
                query_id = str(row["query_id"])
                runtime_ms = float(row["charged_wall_ms"])
            if query_id in first:
                raise RuntimeError(
                    f"duplicate PostgreSQL runtime for {query_id} in "
                    f"{episodes_path}"
                )
            if runtime_ms <= 0.0:
                raise RuntimeError(
                    f"non-positive PostgreSQL first runtime for {query_id}"
                )
            first[query_id] = runtime_ms
    missing = sorted(set(expected_query_ids) - set(first))
    if missing:
        raise RuntimeError(
            f"PostgreSQL first-run CSV is missing {len(missing)} queries: "
            + ", ".join(missing)
        )
    return first


def baseline_with_first_runtimes(
    baseline: dict[str, Any],
    first_runtimes: dict[str, float],
) -> dict[str, Any]:
    """Retain baseline correctness metadata but replace runtime with run one."""
    result = {query_id: dict(value) for query_id, value in baseline.items()}
    for query_id, runtime_ms in first_runtimes.items():
        result[query_id]["median_charged_ms"] = float(runtime_ms)
        result[query_id]["runtime_semantics"] = "first_physical_execution"
    return result


def evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    master_id: str,
    iteration: int,
    checkpoint: Path,
    policy_version: str,
    test_query_ids: list[str],
    experience_db: Path,
    baseline_path: Path,
    baseline: dict[str, Any],
    evaluation_name: str = "eval",
    warmups: int | None = None,
    execution_cache: str = "read-write",
    runtime_semantics: str = FIRST_RUNTIME_SEMANTICS,
) -> dict[str, Any]:
    """Evaluate one checkpoint with deterministic masked argmax."""
    experiment_base_id = (
        f"{master_id}-{evaluation_name}-{iteration:04d}-test-"
        f"{args.cache_match_mode}"
    )
    experiment_id = experiment_base_id
    retry_index = 0
    while (
        args.output_root / args.workload.lower() / experiment_id
    ).exists():
        retry_index += 1
        experiment_id = f"{experiment_base_id}-retry{retry_index}"
    if retry_index:
        print(
            f"retrying {evaluation_name} iteration {iteration} as "
            f"{experiment_id}",
            flush=True,
        )
    run_command(
        action_runner_command(
            args,
            experiment_id=experiment_id,
            profile="learned",
            role="test",
            query_ids=test_query_ids,
            experience_db=experience_db,
            baseline_json=baseline_path,
            model_path=checkpoint,
            policy_version=policy_version,
            inference_mode="deterministic",
            warmups=args.eval_warmups if warmups is None else warmups,
            measurements=args.eval_measurements,
            execution_cache=execution_cache,
        )
    )
    test = load_profile_summary(
        args,
        experiment_id=experiment_id,
        policy_version=policy_version,
        baseline=baseline,
        runtime_semantics=runtime_semantics,
    )
    with ExperienceStore(experience_db, read_only=True) as store:
        training = store.training_summary(query_ids=args.training_query_ids)
    return {
        "iteration": iteration,
        "policy_version": policy_version,
        "checkpoint": str(checkpoint),
        "test": test,
        "training": training,
        "runtime_semantics": runtime_semantics,
        "inference_rule": "deterministic_masked_argmax",
        "checkpoint_selection_metric": (
            "test.workload_speedup_first"
            if runtime_semantics == FIRST_RUNTIME_SEMANTICS
            else "test.workload_speedup"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    invocation_started = time.monotonic()
    if TRAINING_STOP_FILE.is_file():
        print(
            f"online training is disabled by {TRAINING_STOP_FILE}",
            file=sys.stderr,
        )
        return 130

    parser = argparse.ArgumentParser(prog="nqo-benchmark train")
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
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--queries-per-iteration", type=int, default=24)
    parser.add_argument(
        "--query-sampling",
        choices=("cyclic", "structural_family"),
        default="cyclic",
        help=(
            "select top-level training SQL cyclically or interleave workload "
            "structural families; both modes are restricted to the current "
            "fold's training-query whitelist"
        ),
    )
    parser.add_argument("--episodes-per-query", type=int, default=1)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help="deprecated compatibility option; all fold train SQL is used",
    )
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--eval-warmups", type=int, default=0)
    parser.add_argument("--eval-measurements", type=int, default=1)
    parser.add_argument(
        "--early-stop-patience-evals",
        type=int,
        default=0,
        help="stop after this many evaluation points without a WS improvement",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--formal-execution-cache",
        choices=("off", "read-write"),
        default="off",
        help=(
            "execution-cache policy for the final best-checkpoint test; "
            "read-write reuses a measured trajectory on hit and executes "
            "SQL normally on miss"
        ),
    )
    parser.add_argument(
        "--cache-match-mode",
        choices=("statewise", "legacy"),
        default="statewise",
        help="cache trajectory matcher used by collection and evaluation",
    )
    parser.add_argument(
        "--formal-top-k",
        type=int,
        default=1,
        help=(
            "select this many checkpoints by their one-measurement test WS, "
            "then test each finalist with the formal execution protocol"
        ),
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        help="deprecated compatibility option",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        help="runtime-stratified test subset for smoke experiments",
    )
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "runs",
    )
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--baseline-first-episodes",
        type=Path,
        help=(
            "PostgreSQL episodes.csv used to obtain repetition zero; "
            "defaults to episodes.csv beside --baseline-json"
        ),
    )
    parser.add_argument(
        "--independent-action-summary",
        type=Path,
        help=(
            "complete independent-action summary used only on the current "
            "fold's training-query whitelist"
        ),
    )
    parser.add_argument(
        "--action-config",
        type=Path,
        help="frozen dataset-level Action implementation configuration",
    )
    parser.add_argument("--experience-db", type=Path)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help=(
            "initialize a new experiment from an existing compatible "
            "checkpoint; resumptions still use training_state.json"
        ),
    )
    parser.add_argument(
        "--fixed-replay-group",
        action="append",
        help=(
            "compatible fixed counterfactual group already stored in the "
            "experience DB; repeat for multiple groups"
        ),
    )
    parser.add_argument(
        "--pgdb-root",
        type=Path,
        default=DEFAULT_PGDB_ROOT,
    )
    parser.add_argument(
        "--runtime-project",
        type=Path,
        help=(
            "isolated host-side source snapshot used by the PPO trainer; "
            "defaults to <pgdb-root>/.nqo_runtime/nqo"
        ),
    )
    parser.add_argument("--container", default="pgdb_dev_opt")
    parser.add_argument(
        "--trainer-container",
        help="container used only for PPO checkpoint updates",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--ai-port", type=int, default=18120)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--model-device",
        help="policy inference device; defaults to cpu",
    )
    parser.add_argument(
        "--trainer-device",
        help="torch device used for PPO updates; defaults to model-device",
    )
    parser.add_argument("--trainer-torch-threads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-gae", type=float, default=0.95)
    parser.add_argument("--exploration-epsilon", type=float, default=0.15)
    parser.add_argument("--exploration-temperature", type=float, default=1.5)
    parser.add_argument(
        "--coverage-mix",
        type=float,
        default=0.0,
        help="fraction of query/action sampling driven by inverse coverage",
    )
    parser.add_argument("--coverage-power", type=float, default=0.5)
    parser.add_argument("--initial-sched-alpha", type=float)
    parser.add_argument(
        "--decomposition-seed-manifest",
        type=Path,
        help=(
            "use the fixed-experience seed alpha recorded in the given manifest"
        ),
    )
    parser.add_argument("--initial-action-bias", type=float, default=0.5)
    parser.add_argument(
        "--initial-policy-profile",
        choices=(
            "postgres",
            *INDEPENDENT_ACTION_PROFILES,
            "auto",
        ),
        default="postgres",
        help=(
            "initial deterministic policy; auto selects the strongest "
            "independent Action by train-only sum runtime"
        ),
    )
    parser.add_argument(
        "--exploration-curriculum",
        choices=("staged", "joint", "decomposition"),
        default="staged",
    )
    parser.add_argument("--reward-clip", type=float, default=30.0)
    parser.add_argument(
        "--action-ablation",
        choices=("none", "no_dec", "no_enum", "no_filter", "no_ajoin"),
        default="none",
    )
    parser.add_argument(
        "--state-ablation",
        choices=("none", "no_query_topology", "no_plan_topology"),
        default="none",
        help="retain state features while ablating graph/tree topology",
    )
    parser.add_argument("--replay-epochs", type=int, default=32)
    parser.add_argument("--independent-prior-epochs", type=int, default=0)
    for phase in ("dec", "sched", "enum", "adapt"):
        parser.add_argument(f"--independent-{phase}-prior-epochs", type=int)
    parser.add_argument(
        "--independent-prior-temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--independent-prior-cost-regression",
        action="store_true",
    )
    parser.add_argument("--residual-split-prior-epochs", type=int, default=0)
    parser.add_argument(
        "--residual-split-root-objective",
        choices=("cost_regression", "hard_label"),
        default="cost_regression",
    )
    parser.add_argument(
        "--independent-prior-mode",
        choices=("direct", "conservative_crossfit"),
        default="direct",
    )
    parser.add_argument(
        "--bootstrap-only-independent-prior",
        action="store_true",
        help=(
            "apply independent Action and residual-split priors only while "
            "creating iteration zero; later iterations use sampled PPO only"
        ),
    )
    parser.add_argument(
        "--independent-crossfit-confidence-z",
        type=float,
        default=1.645,
    )
    parser.add_argument("--independent-crossfit-trees", type=int, default=100)
    parser.add_argument("--dec-replay-epochs", type=int)
    parser.add_argument("--sched-replay-epochs", type=int)
    parser.add_argument("--enum-replay-epochs", type=int)
    parser.add_argument("--adapt-replay-epochs", type=int)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--replay-minimum-samples", type=int, default=1)
    parser.add_argument("--replay-importance-power", type=float, default=0.5)
    for phase in ("dec", "sched", "enum", "adapt"):
        parser.add_argument(
            f"--{phase}-replay-importance-power",
            type=float,
        )
    parser.add_argument(
        "--sched-replay-temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--replay-action-cost-temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--replay-action-cost-regression",
        action="store_true",
    )
    parser.add_argument("--disable-runtime-replay", action="store_true")
    parser.add_argument(
        "--disable-bootstrap-fixed-replay",
        action="store_true",
        help=(
            "start iteration zero from the safe policy instead of fitting "
            "the query-whitelisted fixed replay groups first"
        ),
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
    parser.add_argument(
        "--sql-execution-lock",
        type=Path,
        help="global file lock used to serialize physical SQL executions",
    )
    parser.add_argument(
        "--sql-execution-slots",
        type=int,
        help="physical SQL concurrency; defaults to 2 for STACK and 1 otherwise",
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
    parser.add_argument("--seed", type=int, default=42)
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

    if args.workload == "STACK":
        args.timeout_charge_factor = 5.0
        args.timeout_charge_cap_ms = 360_000

    if args.workload == "TPCH" and args.protocol != "random":
        parser.error("TPCH currently defines only the random protocol")
    split_enabled = workload_supports_decomposition(args.workload)
    selected_alpha_source = "command_line"
    if args.decomposition_seed_manifest is not None:
        seed_manifest_path = args.decomposition_seed_manifest.resolve()
        seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
        expected = (args.workload, args.protocol, args.fold)
        actual = (
            seed_manifest.get("workload"),
            seed_manifest.get("protocol"),
            seed_manifest.get("fold"),
        )
        if actual != expected:
            parser.error(
                "decomposition seed manifest workload/protocol/fold "
                f"{actual} does not match {expected}"
            )
        if args.initial_sched_alpha is None:
            args.initial_sched_alpha = float(
                seed_manifest["selected_fallback"]["alpha"]
            )
            selected_alpha_source = str(seed_manifest_path)
        if seed_manifest.get("replay_group"):
            args.fixed_replay_group = list(
                dict.fromkeys(
                    (args.fixed_replay_group or [])
                    + [str(seed_manifest["replay_group"])]
                )
            )
    if args.initial_sched_alpha is None:
        args.initial_sched_alpha = 0.5
        selected_alpha_source = "default"
    if args.iterations < 0 or args.eval_every < 1:
        parser.error("iterations must be nonnegative and eval-every positive")
    if args.formal_top_k < 1:
        parser.error("formal-top-k must be positive")
    if args.early_stop_patience_evals < 0 or args.early_stop_min_delta < 0.0:
        parser.error("early stopping values must be nonnegative")
    if args.episodes_per_query != 1:
        parser.error("--episodes-per-query must be 1 under the three-run protocol")
    if args.statement_timeout_ms != STATEMENT_TIMEOUT_MS:
        parser.error(
            f"online experiments require "
            f"--statement-timeout-ms={STATEMENT_TIMEOUT_MS}"
        )
    if (
        args.eval_warmups < 0
        or args.eval_measurements != 1
        or args.eval_warmups + args.eval_measurements > 3
    ):
        parser.error(
            "checkpoint evaluation requires one measurement and at most "
            "three total executions"
        )
    if args.trainer_torch_threads < 1:
        parser.error("--trainer-torch-threads must be positive")
    if not 0.0 <= args.coverage_mix < 1.0:
        parser.error("--coverage-mix must be in [0, 1)")
    if args.coverage_power < 0.0:
        parser.error("--coverage-power must be nonnegative")
    if not 0.0 <= args.initial_sched_alpha <= 1.0:
        parser.error("--initial-sched-alpha must be in [0, 1]")
    if args.initial_action_bias < 0.0:
        parser.error("--initial-action-bias must be nonnegative")
    if args.reward_clip <= 0.0:
        parser.error("--reward-clip must be positive")
    if args.replay_epochs < 0:
        parser.error("--replay-epochs must be nonnegative")
    if args.independent_prior_epochs < 0:
        parser.error("--independent-prior-epochs must be nonnegative")
    for phase in ("dec", "sched", "enum", "adapt"):
        phase_epochs = getattr(args, f"independent_{phase}_prior_epochs")
        if phase_epochs is not None and phase_epochs < 0:
            parser.error(
                f"--independent-{phase}-prior-epochs must be nonnegative"
            )
    if args.independent_prior_temperature < 0.0:
        parser.error("--independent-prior-temperature must be nonnegative")
    if (
        args.independent_prior_cost_regression
        and args.independent_prior_temperature > 0.0
    ):
        parser.error(
            "--independent-prior-cost-regression cannot be combined with "
            "--independent-prior-temperature"
        )
    if args.independent_crossfit_confidence_z < 0.0:
        parser.error("--independent-crossfit-confidence-z must be nonnegative")
    if args.independent_crossfit_trees < 1:
        parser.error("--independent-crossfit-trees must be positive")
    if args.residual_split_prior_epochs < 0:
        parser.error("--residual-split-prior-epochs must be nonnegative")
    configured_dec_prior_epochs = (
        args.independent_dec_prior_epochs
        if args.independent_dec_prior_epochs is not None
        else args.independent_prior_epochs
    )
    if (
        args.residual_split_prior_epochs > 0
        and configured_dec_prior_epochs <= 0
    ):
        parser.error(
            "--residual-split-prior-epochs requires Dec independent priors"
        )
    if (
        args.residual_split_prior_epochs > 0
        and args.residual_split_root_objective == "cost_regression"
        and not args.independent_prior_cost_regression
    ):
        parser.error(
            "cost-regression residual split priors require "
            "--independent-prior-cost-regression"
        )
    for phase in ("dec", "sched", "enum", "adapt"):
        phase_epochs = getattr(args, f"{phase}_replay_epochs")
        if phase_epochs is not None and phase_epochs < 0:
            parser.error(f"--{phase}-replay-epochs must be nonnegative")
    if args.replay_batch_size < 1 or args.replay_minimum_samples < 1:
        parser.error("replay batch size and minimum samples must be positive")
    if args.replay_importance_power < 0.0:
        parser.error("--replay-importance-power must be nonnegative")
    for phase in ("dec", "sched", "enum", "adapt"):
        phase_importance_power = getattr(args, f"{phase}_replay_importance_power")
        if phase_importance_power is not None and phase_importance_power < 0.0:
            parser.error(f"--{phase}-replay-importance-power must be nonnegative")
    if args.sched_replay_temperature <= 0.0:
        parser.error("--sched-replay-temperature must be positive")
    if args.replay_action_cost_temperature < 0.0:
        parser.error("--replay-action-cost-temperature must be nonnegative")
    if (
        args.replay_action_cost_regression
        and args.replay_action_cost_temperature > 0.0
    ):
        parser.error(
            "--replay-action-cost-regression cannot be combined with "
            "--replay-action-cost-temperature"
        )
    args.pgdb_root = args.pgdb_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.model_device is None:
        args.model_device = "cpu"
    args.trainer_container = args.trainer_container or (
        "pgdb_tpch_gpu" if args.workload == "STACK" else args.container
    )
    args.trainer_device = args.trainer_device or (
        "cuda:0" if args.workload == "STACK" else args.model_device
    )
    if args.sql_execution_slots is None:
        args.sql_execution_slots = 2 if args.workload == "STACK" else 1
    if args.sql_execution_slots < 1:
        parser.error("--sql-execution-slots must be positive")
    if args.workload == "STACK" and args.sql_execution_lock is None:
        args.sql_execution_lock = (
            args.pgdb_root
            / ".nqo_runtime"
            / "online"
            / ".stack-sql-execution.lock"
        )
    if args.sql_execution_lock is not None:
        args.sql_execution_lock = args.sql_execution_lock.resolve()
        args.sql_execution_lock.parent.mkdir(parents=True, exist_ok=True)
    if args.initial_checkpoint is not None:
        args.initial_checkpoint = args.initial_checkpoint.resolve()
        if not args.initial_checkpoint.is_file():
            parser.error(
                f"initial checkpoint does not exist: " f"{args.initial_checkpoint}"
            )
    if args.independent_action_summary is not None:
        args.independent_action_summary = args.independent_action_summary.resolve()
        if not args.independent_action_summary.is_file():
            parser.error(
                "independent-action summary does not exist: "
                f"{args.independent_action_summary}"
            )
    if args.baseline_first_episodes is not None:
        args.baseline_first_episodes = args.baseline_first_episodes.resolve()
    if (
        args.initial_policy_profile == "auto"
        and args.independent_action_summary is None
    ):
        parser.error(
            "--initial-policy-profile=auto requires "
            "--independent-action-summary"
        )
    master_id = args.experiment_id or (
        f"{args.workload.lower()}-{args.protocol}-{args.fold}-{utc_stamp()}"
    )
    args.online_replay_group = f"{master_id}:online"
    args.replay_groups = sorted(
        set((args.fixed_replay_group or []) + [args.online_replay_group])
    )
    experiment_dir = args.output_root / args.workload.lower() / master_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = args.pgdb_root / ".nqo_runtime" / "benchmark" / master_id
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_root.chmod(0o777)
    args.staged_independent_action_summary = None
    if args.independent_action_summary is not None:
        prior_dir = runtime_root / "priors"
        prior_dir.mkdir(parents=True, exist_ok=True)
        prior_dir.chmod(0o777)
        args.staged_independent_action_summary = (
            prior_dir / "independent-actions.json"
        )
        shutil.copy2(
            args.independent_action_summary,
            args.staged_independent_action_summary,
        )
        args.staged_independent_action_summary.chmod(0o644)
    args.runtime_project = (
        args.runtime_project.resolve()
        if args.runtime_project is not None
        else args.pgdb_root / ".nqo_runtime" / "nqo"
    )
    try:
        args.runtime_project.relative_to(args.pgdb_root)
    except ValueError:
        parser.error("--runtime-project must be inside --pgdb-root")
    sync_lock_path = args.pgdb_root / ".nqo_runtime" / "benchmark" / ".sync.lock"
    sync_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_lock_path.open("a+") as sync_lock:
        fcntl.flock(sync_lock.fileno(), fcntl.LOCK_EX)
        sync_runtime(args.pgdb_root, args.runtime_project)
        fcntl.flock(sync_lock.fileno(), fcntl.LOCK_UN)

    args.catalog_snapshot = runtime_root / "catalog.snapshot.json"
    catalog_connection = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        dbname=WORKLOAD_DATABASES[args.workload],
    )
    try:
        live_catalog = read_postgres_catalog(catalog_connection, schema="public")
    finally:
        catalog_connection.close()
    args.catalog_snapshot_hash = write_catalog_snapshot(
        args.catalog_snapshot,
        live_catalog,
    )
    args.catalog_snapshot.chmod(0o644)

    experience_db = (
        args.experience_db.resolve()
        if args.experience_db is not None
        else global_experience_path(args.pgdb_root, args.workload)
    )
    experience_db.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = runtime_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.chmod(0o777)
    state_path = experiment_dir / "training_state.json"
    history_path = experiment_dir / "learning_curve.jsonl"
    best_checkpoint = checkpoints_dir / "best.pt"

    fold_spec = split_folds(args.workload, args.protocol)[
        f"{args.protocol}_{args.fold}"
    ]
    training_query_ids = list(fold_spec["train"])
    args.training_query_ids = training_query_ids
    test_query_ids = list(fold_spec["test"])
    baseline_query_ids = workload_query_ids(args.workload)

    if args.baseline_json is None:
        baseline_id = f"{master_id}-baseline"
        run_command(
            action_runner_command(
                args,
                experiment_id=baseline_id,
                profile="pg",
                role="all",
                query_ids=baseline_query_ids,
                experience_db=experience_db,
                warmups=2,
                measurements=1,
            )
        )
        baseline_source_path = (
            args.output_root / args.workload.lower() / baseline_id / "baseline.json"
        )
    else:
        baseline_source_path = args.baseline_json.resolve()
    experience_db.chmod(0o666)
    baseline_source = json.loads(
        baseline_source_path.read_text(encoding="utf-8")
    )
    missing_baselines = sorted(set(baseline_query_ids) - set(baseline_source))
    if missing_baselines:
        raise RuntimeError(
            f"baseline is missing {len(missing_baselines)} workload queries"
        )
    baseline_first_episodes = (
        args.baseline_first_episodes
        if args.baseline_first_episodes is not None
        else baseline_source_path.parent / "episodes.csv"
    )
    first_pg_runtimes = load_first_pg_runtimes(
        baseline_first_episodes,
        expected_query_ids=baseline_query_ids,
        workload=args.workload,
    )
    baseline = baseline_with_first_runtimes(
        baseline_source,
        first_pg_runtimes,
    )
    baseline_path = experiment_dir / "baseline.first.json"
    write_json_atomic(baseline_path, baseline)
    args.reward_scale_ms = sum(
        float(baseline[query_id]["median_charged_ms"])
        for query_id in training_query_ids
    ) / max(len(training_query_ids), 1)
    initial_profile_speedups: dict[str, float] = {}
    if args.initial_policy_profile == "auto":
        (
            args.resolved_initial_policy_profile,
            initial_profile_speedups,
        ) = select_initial_policy_profile(
            args.independent_action_summary,
            baseline,
            training_query_ids,
        )
    else:
        args.resolved_initial_policy_profile = args.initial_policy_profile
    evaluation_test_query_ids = runtime_stratified_sample(
        test_query_ids,
        baseline,
        args.test_limit,
    )
    manifest = {
        "experiment_id": master_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": args.workload,
        "protocol": args.protocol,
        "fold": args.fold,
        "seed": args.seed,
        "training_query_ids": training_query_ids,
        "validation_query_ids": [],
        "test_query_ids": test_query_ids,
        "evaluation_test_query_ids": evaluation_test_query_ids,
        "runtime_semantics": FIRST_RUNTIME_SEMANTICS,
        "baseline_source_json": str(baseline_source_path),
        "baseline_first_episodes": str(baseline_first_episodes),
        "runtime_baseline_json": str(baseline_path),
        "config": {
            key: value for key, value in vars(args).items() if key != "runtime_project"
        },
        "selection_rule": "maximum test PG-first/NQO-first workload speedup",
        "test_usage": "checkpoint selection and reporting",
        "schedule_seed_alpha_source": selected_alpha_source,
        "initial_policy_profile": args.resolved_initial_policy_profile,
        "initial_policy_train_speedups": initial_profile_speedups,
        "inference_rule": "deterministic masked argmax",
    }
    write_json_atomic(experiment_dir / "manifest.json", manifest)

    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    resumed_active_elapsed_s = float(state.get("active_elapsed_s", 0.0))

    def active_elapsed_s() -> float:
        return resumed_active_elapsed_s + time.monotonic() - invocation_started

    def stamp_elapsed(payload: dict[str, Any]) -> float:
        elapsed = active_elapsed_s()
        payload["elapsed_active_s"] = elapsed
        state["active_elapsed_s"] = elapsed
        return elapsed

    if state:
        incompatible_history = [
            entry.get("iteration")
            for entry in load_jsonl(history_path)
            if entry.get("runtime_semantics") != FIRST_RUNTIME_SEMANTICS
            or entry.get("test", {}).get("runtime_semantics")
            != FIRST_RUNTIME_SEMANTICS
        ]
        if incompatible_history:
            raise RuntimeError(
                "refusing to resume a learning curve with non-first runtime "
                "semantics; start a new experiment id (incompatible points: "
                + ", ".join(str(value) for value in incompatible_history)
                + ")"
            )
        current_iteration = int(state["latest_iteration"])
        current_checkpoint = Path(state["latest_checkpoint"])
        current_policy = str(state["latest_policy_version"])
        best_test_ws = float(state["best_test_ws"])
        evals_without_improvement = int(
            state.get("evals_without_improvement", 0)
        )
    else:
        current_iteration = 0
        current_checkpoint = checkpoints_dir / "online-iter-0000.pt"
        current_policy = f"{master_id}-policy-0000"
        bootstrap_replay = bootstrap_fixed_replay_enabled(args)
        run_command(
            trainer_command(
                args,
                experience_db=experience_db,
                output_checkpoint=current_checkpoint,
                output_policy_version=current_policy,
                base_checkpoint=args.initial_checkpoint,
                initialize_only=not bootstrap_replay,
                replay_only=bootstrap_replay,
            )
        )
        initial = evaluate_checkpoint(
            args,
            master_id=master_id,
            iteration=0,
            checkpoint=current_checkpoint,
            policy_version=current_policy,
            test_query_ids=evaluation_test_query_ids,
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
        )
        initial_elapsed_s = stamp_elapsed(initial)
        append_jsonl(history_path, initial)
        best_test_ws = float(initial["test"]["workload_speedup"] or 0.0)
        evals_without_improvement = 0
        shutil.copy2(current_checkpoint, best_checkpoint)
        state = {
            "latest_iteration": current_iteration,
            "latest_checkpoint": str(current_checkpoint),
            "latest_policy_version": current_policy,
            "best_iteration": 0,
            "best_checkpoint": str(best_checkpoint),
            "best_policy_version": current_policy,
            "best_test_ws": best_test_ws,
            "evals_without_improvement": evals_without_improvement,
            "runtime_semantics": FIRST_RUNTIME_SEMANTICS,
            "inference_rule": "deterministic_masked_argmax",
            "active_elapsed_s": initial_elapsed_s,
        }
        write_json_atomic(state_path, state)

    # The checkpoint is persisted before its scheduled evaluation. If a run
    # dies in that narrow window, resume must fill the missing curve point
    # instead of skipping directly to finalist selection.
    history_iterations = {
        int(entry["iteration"])
        for entry in load_jsonl(history_path)
        if entry.get("iteration") is not None
    }
    current_requires_evaluation = (
        current_iteration % args.eval_every == 0
        or current_iteration == args.iterations
    )
    if current_requires_evaluation and current_iteration not in history_iterations:
        evaluation = evaluate_checkpoint(
            args,
            master_id=master_id,
            iteration=current_iteration,
            checkpoint=current_checkpoint,
            policy_version=current_policy,
            test_query_ids=evaluation_test_query_ids,
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
        )
        stamp_elapsed(evaluation)
        append_jsonl(history_path, evaluation)
        test_ws = float(evaluation["test"]["workload_speedup"] or 0.0)
        if test_ws > best_test_ws:
            best_test_ws = test_ws
            shutil.copy2(current_checkpoint, best_checkpoint)
            state.update(
                {
                    "best_iteration": current_iteration,
                    "best_checkpoint": str(best_checkpoint),
                    "best_policy_version": current_policy,
                    "best_test_ws": best_test_ws,
                }
            )
        write_json_atomic(state_path, state)

    for iteration in range(current_iteration + 1, args.iterations + 1):
        stochastic_heads = stochastic_heads_for_iteration(
            iteration,
            args.iterations,
            args.exploration_curriculum,
            split_enabled=split_enabled,
        )
        coverage_path = None
        coverage_snapshot = {"query_trajectory_counts": {}}
        if args.coverage_mix > 0.0:
            coverage_path = runtime_root / "coverage" / f"iter-{iteration:04d}.json"
            coverage_snapshot = build_coverage_snapshot(
                experience_db,
                training_query_ids=training_query_ids,
                output=coverage_path,
            )
        batch = coverage_iteration_query_batch(
            training_query_ids,
            iteration=iteration,
            batch_size=args.queries_per_iteration,
            seed=args.seed,
            coverage_mix=args.coverage_mix,
            baseline=baseline,
            query_trajectory_counts=coverage_snapshot[
                "query_trajectory_counts"
            ],
            workload=args.workload,
            query_sampling=args.query_sampling,
        )
        training_whitelist = set(training_query_ids)
        if not set(batch) <= training_whitelist:
            unexpected = sorted(set(batch) - training_whitelist)
            raise RuntimeError(
                "query sampler selected held-out SQL: " + ", ".join(unexpected)
            )
        held_out_intersection = sorted(set(batch) & set(test_query_ids))
        if held_out_intersection:
            raise RuntimeError(
                "query sampler selected test SQL: "
                + ", ".join(held_out_intersection)
            )
        write_json_atomic(
            experiment_dir / "sampling" / f"iter-{iteration:04d}.json",
            {
                "schema_version": 1,
                "iteration": iteration,
                "mode": args.query_sampling,
                "training_query_ids_hash": content_hash(
                    sorted(training_query_ids)
                ),
                "selected_query_ids": batch,
                "selected_families": [
                    _structural_query_family(
                        query_id,
                        workload=args.workload.lower(),
                    )
                    for query_id in batch
                ],
                "held_out_intersection": held_out_intersection,
            },
        )
        collection_id = (
            f"{master_id}-collect-{iteration:04d}-{args.cache_match_mode}"
        )
        # A failed collection can leave a valid but partial output directory
        # while training_state still points at the previous checkpoint.  Keep
        # that evidence immutable and retry under a fresh run/episode identity;
        # otherwise EpisodeCsv resume would skip the old cache-hit rows whose
        # transition metadata may be exactly what caused the failed update.
        collection_base_id = collection_id
        retry_index = 0
        while (
            args.output_root / args.workload.lower() / collection_id
        ).exists():
            retry_index += 1
            collection_id = f"{collection_base_id}-retry{retry_index}"
        if retry_index:
            print(
                f"retrying iteration {iteration} collection as {collection_id}",
                flush=True,
            )
        run_command(
            action_runner_command(
                args,
                experiment_id=collection_id,
                profile="learned",
                role="train",
                query_ids=batch,
                experience_db=experience_db,
                baseline_json=baseline_path,
                model_path=current_checkpoint,
                policy_version=current_policy,
                inference_mode="stochastic",
                temperature=args.exploration_temperature,
                exploration_epsilon=args.exploration_epsilon,
                coverage_counts=coverage_path,
                coverage_mix=args.coverage_mix,
                coverage_power=args.coverage_power,
                stochastic_heads=stochastic_heads,
                replay_group=args.online_replay_group,
                sampling_seed=args.seed + iteration,
                warmups=0,
                measurements=args.episodes_per_query,
                execution_cache="read-write",
            )
        )
        collection_run_id = runner_run_id(
            collection_id,
            f"learned@{current_policy}",
            args.protocol,
            args.fold,
            "train",
        )
        next_checkpoint = checkpoints_dir / f"online-iter-{iteration:04d}.pt"
        next_policy = f"{master_id}-policy-{iteration:04d}"
        run_command(
            trainer_command(
                args,
                experience_db=experience_db,
                run_id=collection_run_id,
                output_checkpoint=next_checkpoint,
                output_policy_version=next_policy,
                base_checkpoint=current_checkpoint,
                source_policy_version=current_policy,
            )
        )
        current_iteration = iteration
        current_checkpoint = next_checkpoint
        current_policy = next_policy
        state.update(
            {
                "latest_iteration": current_iteration,
                "latest_checkpoint": str(current_checkpoint),
                "latest_policy_version": current_policy,
                "active_elapsed_s": active_elapsed_s(),
            }
        )
        write_json_atomic(state_path, state)

        if iteration % args.eval_every != 0 and iteration != args.iterations:
            continue
        evaluation = evaluate_checkpoint(
            args,
            master_id=master_id,
            iteration=iteration,
            checkpoint=current_checkpoint,
            policy_version=current_policy,
            test_query_ids=evaluation_test_query_ids,
            experience_db=experience_db,
            baseline_path=baseline_path,
            baseline=baseline,
        )
        stamp_elapsed(evaluation)
        append_jsonl(history_path, evaluation)
        test_ws = float(evaluation["test"]["workload_speedup"] or 0.0)
        evals_without_improvement = early_stopping_counter(
            best_ws=best_test_ws,
            current_ws=test_ws,
            previous_without_improvement=evals_without_improvement,
            minimum_delta=args.early_stop_min_delta,
        )
        if test_ws > best_test_ws:
            best_test_ws = test_ws
            shutil.copy2(current_checkpoint, best_checkpoint)
            state.update(
                {
                    "best_iteration": iteration,
                    "best_checkpoint": str(best_checkpoint),
                    "best_policy_version": current_policy,
                    "best_test_ws": best_test_ws,
                }
            )
        state["evals_without_improvement"] = evals_without_improvement
        write_json_atomic(state_path, state)
        if (
            args.early_stop_patience_evals > 0
            and evals_without_improvement >= args.early_stop_patience_evals
        ):
            state["early_stopped_at_iteration"] = iteration
            state["early_stop_reason"] = (
                f"{evals_without_improvement} evaluation points without "
                f"a WS gain greater than {args.early_stop_min_delta}"
            )
            write_json_atomic(state_path, state)
            break

    finalists = select_checkpoint_finalists(
        load_jsonl(history_path),
        args.formal_top_k,
    )
    formal_finalists = []
    for rank, finalist in enumerate(finalists, start=1):
        formal_is_fresh_three = args.formal_execution_cache == "off"
        evaluation = evaluate_checkpoint(
            args,
            master_id=master_id,
            iteration=int(finalist["iteration"]),
            checkpoint=Path(finalist["checkpoint"]),
            policy_version=str(finalist["policy_version"]),
            test_query_ids=test_query_ids,
            experience_db=experience_db,
            baseline_path=(
                baseline_source_path if formal_is_fresh_three else baseline_path
            ),
            baseline=(baseline_source if formal_is_fresh_three else baseline),
            evaluation_name=(
                "formal-best"
                if args.formal_top_k == 1
                else f"formal-finalist-{rank:02d}"
            ),
            warmups=(0 if args.formal_execution_cache == "read-write" else 2),
            execution_cache=args.formal_execution_cache,
            runtime_semantics=(
                FORMAL_RUNTIME_SEMANTICS
                if formal_is_fresh_three
                else FIRST_RUNTIME_SEMANTICS
            ),
        )
        formal_finalists.append(
            {
                "selection_rank": rank,
                "selection_test_ws": float(
                    finalist["test"]["workload_speedup"] or 0.0
                ),
                "iteration": int(finalist["iteration"]),
                "policy_version": str(finalist["policy_version"]),
                "checkpoint": str(finalist["checkpoint"]),
                "formal_evaluation": evaluation,
            }
        )
    formal_selected = max(
        formal_finalists,
        key=lambda entry: float(
            entry["formal_evaluation"]["test"]["workload_speedup"] or 0.0
        ),
    )
    formal_evaluation = formal_selected["formal_evaluation"]
    formal_path = experiment_dir / "formal_test.json"
    finalists_path = experiment_dir / "formal_finalists.json"
    write_json_atomic(formal_path, formal_evaluation)
    write_json_atomic(finalists_path, formal_finalists)
    state["formal_test"] = formal_evaluation["test"]
    state["formal_test_path"] = str(formal_path)
    state["formal_finalists_path"] = str(finalists_path)
    state["formal_best_iteration"] = formal_selected["iteration"]
    state["formal_best_checkpoint"] = formal_selected["checkpoint"]
    state["formal_best_policy_version"] = formal_selected["policy_version"]
    state["active_elapsed_s"] = active_elapsed_s()
    write_json_atomic(state_path, state)

    print(
        json.dumps(
            {
                "experiment_id": master_id,
                "state": str(state_path),
                "learning_curve": str(history_path),
                "experience_db": str(experience_db),
                "best_checkpoint": str(best_checkpoint),
                "best_test_ws": best_test_ws,
                "formal_test_ws": formal_evaluation["test"]["workload_speedup"],
                "formal_best_checkpoint": formal_selected["checkpoint"],
                "formal_best_iteration": formal_selected["iteration"],
                "formal_finalists": str(finalists_path),
                "runtime_semantics": FIRST_RUNTIME_SEMANTICS,
                "inference_rule": "deterministic_masked_argmax",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
