#!/usr/bin/env python3
"""Run and aggregate the complete NQO benchmark matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PGDB_ROOT = ROOT.parent / "pgdb"

from benchmarking.workloads import (  # noqa: E402
    SPLIT_PROTOCOLS,
    workload_query_ids,
)
from benchmarking.matrix_reports import (  # noqa: E402
    aggregate_learning_curves,
    aggregate_selected_checkpoints,
    combine_training_stages,
    load_jsonl,
    test_query_multiplicity,
    write_csv_atomic,
)

from optimization.actions import (  # noqa: E402
    STATEMENT_TIMEOUT_MS,
    TIMEOUT_CHARGE_CAP_MS,
    TIMEOUT_CHARGE_FACTOR,
    apply_action_config,
    global_experience_path,
    load_action_config,
)
from benchmarking.utils import utc_stamp, write_json_atomic  # noqa: E402

BENCHMARK_MODULE = "benchmarking.action_runner"
TUNER_MODULE = "benchmarking.action_tuning"
TRAINER_MODULE = "benchmarking.iterative_training"
TRAINING_STOP_FILE = ROOT / ".local" / "STOP_ALL_TRAINING"


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]




def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def set_cli_option(command: list[str], option: str, value: Any) -> list[str]:
    """Return a command copy with one scalar CLI option set exactly once."""
    updated = list(command)
    if option in updated:
        index = updated.index(option)
        if index + 1 >= len(updated):
            raise ValueError(f"{option} has no value")
        updated[index + 1] = str(value)
    else:
        updated.extend([option, str(value)])
    return updated




def common_runtime_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--output-root",
        str(args.output_root),
        "--pgdb-root",
        str(args.pgdb_root),
        "--container",
        args.container,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--user",
        args.user,
        "--ai-port",
        str(args.ai_port),
        "--statement-timeout-ms",
        str(args.statement_timeout_ms),
        "--timeout-charge-factor",
        str(args.timeout_charge_factor),
        "--timeout-charge-cap-ms",
        str(args.timeout_charge_cap_ms),
        "--max-rounds",
        str(args.max_rounds),
        "--search-max-rels",
        str(args.search_max_rels),
        "--lip-min-probe-ratio",
        str(args.lip_min_probe_ratio),
    ]


def training_runtime_arguments(args: argparse.Namespace) -> list[str]:
    result = common_runtime_arguments(args)
    if args.runtime_project:
        result.extend(["--runtime-project", str(args.runtime_project)])
    if args.trainer_container:
        result.extend(["--trainer-container", args.trainer_container])
    if args.trainer_device:
        result.extend(["--trainer-device", args.trainer_device])
    if args.sql_execution_lock:
        result.extend(["--sql-execution-lock", str(args.sql_execution_lock)])
        result.extend(
            ["--sql-execution-slots", str(args.sql_execution_slots)]
        )
    return result




def main(argv: list[str] | None = None) -> int:
    if TRAINING_STOP_FILE.is_file():
        print(
            f"online training is disabled by {TRAINING_STOP_FILE}",
            file=sys.stderr,
        )
        return 130

    parser = argparse.ArgumentParser(prog="nqo-benchmark matrix")
    parser.add_argument(
        "--workload",
        choices=("JOB", "STACK", "TPCH"),
        default="JOB",
    )
    parser.add_argument(
        "--protocols",
        help="comma-separated; defaults to every protocol for the workload",
    )
    parser.add_argument("--folds", default="a,b,c")
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "runs",
    )
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--action-config",
        type=Path,
        help=(
            "reuse an existing frozen dataset-level Action configuration; "
            "without it the matrix calibrates exactly once"
        ),
    )
    parser.add_argument(
        "--shared-experience-db",
        type=Path,
        help=(
            "reuse one dataset-level experience buffer across folds; "
            "fold-local query whitelists still control training eligibility"
        ),
    )
    parser.add_argument("--fixed-replay-group", action="append")
    parser.add_argument("--independent-action-summary", type=Path)
    parser.add_argument(
        "--pgdb-root",
        type=Path,
        default=DEFAULT_PGDB_ROOT,
    )
    parser.add_argument(
        "--runtime-project",
        type=Path,
        help="isolated source snapshot used by online PPO trainers",
    )
    parser.add_argument("--container", default="pgdb_dev_opt")
    parser.add_argument("--trainer-container")
    parser.add_argument("--trainer-device")
    parser.add_argument("--sql-execution-lock", type=Path)
    parser.add_argument(
        "--sql-execution-slots",
        type=int,
        help="physical SQL concurrency; defaults to 2 for STACK and 1 otherwise",
    )
    parser.add_argument(
        "--training-parallelism",
        type=int,
        help="concurrent fold trainers; defaults to all 9 STACK folds",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--ai-port", type=int, default=18120)
    parser.add_argument("--skip-action-tuning", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--tune-limit", type=int, default=12)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--alpha-grid", default="0.5")
    parser.add_argument(
        "--prefix-depth-grid",
        default="1,2,3,4,5,6",
    )
    parser.add_argument("--branch-alpha-depth-grid", default="2,3")
    parser.add_argument("--lip-build-row-grid", default="10000,100000,500000")
    parser.add_argument("--lip-selectivity-grid", default="1,5,10,25")
    parser.add_argument(
        "--aja-threshold-grid",
        default="1,100,1000,5000,10000,100000,500000",
    )
    parser.add_argument("--tune-warmups", type=int, default=0)
    parser.add_argument("--tune-measurements", type=int, default=1)
    parser.add_argument("--test-warmups", type=int, default=2)
    parser.add_argument("--test-measurements", type=int, default=1)
    parser.add_argument("--replay-warmups", type=int, default=0)
    parser.add_argument("--replay-measurements", type=int, default=1)
    parser.add_argument("--baseline-warmups", type=int, default=2)
    parser.add_argument("--baseline-measurements", type=int, default=1)
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--queries-per-iteration", type=int, default=24)
    parser.add_argument(
        "--query-sampling",
        choices=("cyclic", "structural_family"),
        default="cyclic",
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
    parser.add_argument("--early-stop-patience-evals", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--formal-top-k",
        type=int,
        default=1,
        help=(
            "select this many checkpoints per fold by one-measurement test WS "
            "for the final three-execution comparison"
        ),
    )
    parser.add_argument(
        "--eval-validation-limit",
        type=int,
        help="deprecated compatibility option",
    )
    parser.add_argument("--eval-test-limit", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--model-device",
        help="policy inference device; defaults to cpu",
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
    parser.add_argument("--coverage-mix", type=float, default=0.0)
    parser.add_argument("--coverage-power", type=float, default=0.5)
    parser.add_argument("--initial-sched-alpha", type=float, default=0.5)
    parser.add_argument("--initial-action-bias", type=float, default=0.5)
    parser.add_argument(
        "--initial-policy-profile",
        choices=(
            "postgres",
            "query_split",
            "top5",
            "lip_selective",
            "aja_conservative",
            "auto",
        ),
        default="postgres",
    )
    parser.add_argument(
        "--exploration-curriculum",
        choices=("staged", "joint", "decomposition"),
        default="staged",
    )
    parser.add_argument(
        "--refinement-iterations",
        type=int,
        default=0,
        help=(
            "continue every fold from its selected primary checkpoint for "
            "this many uniform refinement iterations"
        ),
    )
    parser.add_argument(
        "--refinement-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--refinement-ppo-epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--refinement-entropy",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--refinement-exploration-curriculum",
        choices=("staged", "joint", "decomposition"),
        default="decomposition",
    )
    parser.add_argument(
        "--refinement-dec-replay-epochs",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--refinement-dec-replay-importance-power",
        type=float,
        default=1.0,
    )
    parser.add_argument("--reward-clip", type=float, default=30.0)
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
    if args.statement_timeout_ms != STATEMENT_TIMEOUT_MS:
        parser.error(
            f"online experiments require "
            f"--statement-timeout-ms={STATEMENT_TIMEOUT_MS}"
        )

    protocols = (
        comma_values(args.protocols)
        if args.protocols
        else list(SPLIT_PROTOCOLS[args.workload])
    )
    folds = comma_values(args.folds)
    invalid_protocols = sorted(set(protocols) - set(SPLIT_PROTOCOLS[args.workload]))
    if invalid_protocols:
        parser.error(f"unsupported protocols: {invalid_protocols}")
    if not folds or set(folds) - {"a", "b", "c"}:
        parser.error("--folds must contain a, b, and/or c")
    if args.skip_action_tuning and args.action_config is None:
        parser.error("--skip-action-tuning requires --action-config")
    for label in ("tune", "test", "replay", "baseline", "eval"):
        warmups = getattr(args, f"{label}_warmups")
        measurements = getattr(args, f"{label}_measurements")
        if warmups < 0 or measurements != 1 or warmups + measurements > 3:
            parser.error(
                f"{label} requires one measurement and at most three "
                "total executions"
            )
    if args.episodes_per_query != 1:
        parser.error("--episodes-per-query must be 1")
    if args.formal_top_k < 1:
        parser.error("--formal-top-k must be positive")
    if args.iterations < 0 or args.refinement_iterations < 0:
        parser.error("training and refinement iterations must be nonnegative")
    if (
        args.refinement_learning_rate <= 0.0
        or args.refinement_ppo_epochs < 0
        or args.refinement_entropy < 0.0
        or args.refinement_dec_replay_epochs < 0
        or args.refinement_dec_replay_importance_power < 0.0
    ):
        parser.error("refinement hyperparameters must be nonnegative")
    if args.replay_epochs < 0 or args.replay_importance_power < 0.0:
        parser.error("replay epochs and importance power must be nonnegative")
    for phase in ("dec", "sched", "enum", "adapt"):
        phase_importance_power = getattr(args, f"{phase}_replay_importance_power")
        if phase_importance_power is not None and phase_importance_power < 0.0:
            parser.error(f"--{phase}-replay-importance-power must be nonnegative")
    if args.trainer_torch_threads < 1:
        parser.error("--trainer-torch-threads must be positive")
    if args.replay_batch_size < 1 or args.replay_minimum_samples < 1:
        parser.error("replay batch size and minimum samples must be positive")
    if args.sched_replay_temperature <= 0.0:
        parser.error("--sched-replay-temperature must be positive")
    if args.independent_prior_temperature < 0.0:
        parser.error("--independent-prior-temperature must be nonnegative")
    if args.replay_action_cost_temperature < 0.0:
        parser.error("--replay-action-cost-temperature must be nonnegative")
    if (
        args.independent_prior_cost_regression
        and args.independent_prior_temperature > 0.0
    ):
        parser.error(
            "--independent-prior-cost-regression cannot be combined with "
            "--independent-prior-temperature"
        )
    if args.residual_split_prior_epochs < 0:
        parser.error("--residual-split-prior-epochs must be nonnegative")
    if (
        args.replay_action_cost_regression
        and args.replay_action_cost_temperature > 0.0
    ):
        parser.error(
            "--replay-action-cost-regression cannot be combined with "
            "--replay-action-cost-temperature"
        )

    args.output_root = args.output_root.resolve()
    args.pgdb_root = args.pgdb_root.resolve()
    if args.model_device is None:
        args.model_device = "cpu"
    args.trainer_device_auto = args.trainer_device is None
    if args.trainer_container is None and args.workload == "STACK":
        args.trainer_container = "pgdb_tpch_gpu"
    if args.trainer_device is None:
        args.trainer_device = "cuda:0" if args.workload == "STACK" else args.model_device
    if args.sql_execution_slots is None:
        args.sql_execution_slots = 2 if args.workload == "STACK" else 1
    if args.training_parallelism is None:
        args.training_parallelism = 9 if args.workload == "STACK" else 1
    if args.sql_execution_slots < 1:
        parser.error("--sql-execution-slots must be positive")
    if args.training_parallelism < 1:
        parser.error("--training-parallelism must be positive")
    if args.workload == "STACK" and args.sql_execution_lock is None:
        args.sql_execution_lock = (
            args.pgdb_root
            / ".nqo_runtime"
            / "online"
            / ".stack-sql-execution.lock"
        )
    matrix_id = args.experiment_id or (
        f"{args.workload.lower()}-online-matrix-{utc_stamp()}"
    )
    matrix_dir = args.output_root / args.workload.lower() / matrix_id
    matrix_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.pgdb_root / ".nqo_runtime" / "benchmark" / matrix_id
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o777)
    frozen_action_config = None
    action_config_path = None
    if args.action_config is not None:
        action_config_path = args.action_config.resolve()
        frozen_action_config = load_action_config(
            action_config_path,
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

    if args.baseline_json is None:
        baseline_id = f"{matrix_id}-baseline"
        # The runner accepts a buffer path uniformly, but PostgreSQL baseline
        # executions never insert rows into it.
        baseline_db = runtime_dir / "baseline_light.sql"
        baseline_command = [
            sys.executable,
            "-m",
            BENCHMARK_MODULE,
            "--workload",
            args.workload,
            "--profiles",
            "pg",
            "--protocol",
            protocols[0],
            "--fold",
            folds[0],
            "--role",
            "all",
            "--experiment-id",
            baseline_id,
            "--experience-db",
            str(baseline_db),
            "--warmups",
            str(args.baseline_warmups),
            "--measurements",
            str(args.baseline_measurements),
            *common_runtime_arguments(args),
        ]
        if args.prewarm:
            baseline_command.append("--prewarm")
        run_command(baseline_command)
        baseline_path = (
            args.output_root / args.workload.lower() / baseline_id / "baseline.json"
        )
    else:
        baseline_path = args.baseline_json.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    missing = sorted(set(workload_query_ids(args.workload)) - set(baseline))
    if missing:
        raise RuntimeError(
            f"shared PostgreSQL baseline is missing {len(missing)} queries"
        )

    dataset_experience_db = (
        args.shared_experience_db.resolve()
        if args.shared_experience_db is not None
        else global_experience_path(args.pgdb_root, args.workload)
    )

    tuning_report = None
    if frozen_action_config is None:
        tuning_id = f"{matrix_id}-tune-{args.workload.lower()}"
        action_config_path = matrix_dir / "frozen_action_config.json"
        command = [
            sys.executable,
            "-m",
            TUNER_MODULE,
            "--workload",
            args.workload,
            "--protocol",
            protocols[0],
            "--fold",
            folds[0],
            "--calibration-scope",
            "dataset",
            "--experiment-id",
            tuning_id,
            "--baseline-json",
            str(baseline_path),
            "--experience-db",
            str(dataset_experience_db),
            "--action-config-out",
            str(action_config_path),
            "--alpha-grid",
            args.alpha_grid,
            "--prefix-depth-grid",
            args.prefix_depth_grid,
            "--branch-alpha-depth-grid",
            args.branch_alpha_depth_grid,
            "--seed",
            str(args.seed),
            "--aja-threshold-grid",
            args.aja_threshold_grid,
            "--lip-build-row-grid",
            args.lip_build_row_grid,
            "--lip-selectivity-grid",
            args.lip_selectivity_grid,
            "--aja-max-nestloop-cost-ratio-pct",
            str(args.aja_max_nestloop_cost_ratio_pct),
            "--aja-aggressive-max-nestloop-cost-ratio-pct",
            str(args.aja_aggressive_max_nestloop_cost_ratio_pct),
            "--lip-max-build-relation-rows",
            str(args.lip_max_build_relation_rows),
            "--lip-selective-plan-rows",
            str(args.lip_selective_plan_rows),
            "--lip-max-build-selectivity-pct",
            str(args.lip_max_build_selectivity_pct),
            "--lip-max-filters",
            str(args.lip_max_filters),
            "--tune-warmups",
            str(args.tune_warmups),
            "--tune-measurements",
            str(args.tune_measurements),
            "--test-warmups",
            str(args.test_warmups),
            "--test-measurements",
            str(args.test_measurements),
            "--replay-warmups",
            str(args.replay_warmups),
            "--replay-measurements",
            str(args.replay_measurements),
            *common_runtime_arguments(args),
        ]
        if args.tune_limit is not None:
            command.extend(["--tune-limit", str(args.tune_limit)])
        if args.test_limit is not None:
            command.extend(["--test-limit", str(args.test_limit)])
        run_command(command)
        report_path = (
            args.output_root
            / args.workload.lower()
            / tuning_id
            / "action_tuning_report.json"
        )
        tuning_report = json.loads(report_path.read_text(encoding="utf-8"))
        frozen_action_config = load_action_config(
            action_config_path,
            workload=args.workload,
        )
        apply_action_config(args, frozen_action_config)

    assert frozen_action_config is not None
    assert action_config_path is not None
    if args.workload == "STACK":
        # A newly tuned or an older frozen Action config must not replace the
        # STACK first-run timeout semantics used by fold training.
        args.timeout_charge_factor = 5.0
        args.timeout_charge_cap_ms = 360_000
    config_replay_db = Path(
        frozen_action_config.get("replay", {}).get("experience_db", "")
    )
    fixed_replay_groups = list(dict.fromkeys(args.fixed_replay_group or []))
    if (
        config_replay_db.is_file()
        and config_replay_db.resolve() == dataset_experience_db.resolve()
    ):
        fixed_replay_groups = list(
            dict.fromkeys(
                fixed_replay_groups
                + list(frozen_action_config.get("replay", {}).get("groups", []))
            )
        )

    matrix_manifest = {
        "experiment_id": matrix_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": args.workload,
        "protocols": protocols,
        "folds": folds,
        "baseline": str(baseline_path),
        "action_config": str(action_config_path),
        "action_config_hash": frozen_action_config["config_hash"],
        "action_calibration": "once_per_dataset",
        "experience_db": str(dataset_experience_db),
        "config": vars(args),
        "ws_definition": "sum(PG test wall time) / sum(NQO test wall time)",
        "checkpoint_selection": "maximum fold-local test WS",
        "test_policy": "deterministic eligibility-mask argmax",
        "test_multiplicity": {
            protocol: test_query_multiplicity(args.workload, protocol, folds)
            for protocol in protocols
        },
    }
    write_json_atomic(matrix_dir / "manifest.json", matrix_manifest)

    training_histories: dict[str, dict[str, list[dict[str, Any]]]] = {
        protocol: {} for protocol in protocols
    }
    training_states: dict[str, dict[str, dict[str, Any]]] = {
        protocol: {} for protocol in protocols
    }

    def train_fold(
        protocol: str,
        fold: str,
    ) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
        training_id = f"{matrix_id}-train-{protocol}-{fold}"
        command = [
            sys.executable,
            "-m",
            TRAINER_MODULE,
            "--workload",
            args.workload,
            "--protocol",
            protocol,
            "--fold",
            fold,
            "--experiment-id",
            training_id,
            "--baseline-json",
            str(baseline_path),
            "--action-config",
            str(action_config_path),
            "--experience-db",
            str(dataset_experience_db),
            "--iterations",
            str(args.iterations),
            "--queries-per-iteration",
            str(args.queries_per_iteration),
            "--query-sampling",
            args.query_sampling,
            "--episodes-per-query",
            str(args.episodes_per_query),
            "--eval-every",
            str(args.eval_every),
            "--eval-warmups",
            str(args.eval_warmups),
            "--eval-measurements",
            str(args.eval_measurements),
            "--early-stop-patience-evals",
            str(args.early_stop_patience_evals),
            "--early-stop-min-delta",
            str(args.early_stop_min_delta),
            "--formal-top-k",
            str(args.formal_top_k),
            "--hidden",
            str(args.hidden),
            "--model-device",
            args.model_device,
            "--trainer-torch-threads",
            str(args.trainer_torch_threads),
            "--learning-rate",
            str(args.learning_rate),
            "--ppo-epochs",
            str(args.ppo_epochs),
            "--ppo-batch-size",
            str(args.ppo_batch_size),
            "--entropy",
            str(args.entropy),
            "--gamma",
            str(args.gamma),
            "--lambda-gae",
            str(args.lambda_gae),
            "--exploration-epsilon",
            str(args.exploration_epsilon),
            "--exploration-temperature",
            str(args.exploration_temperature),
            "--coverage-mix",
            str(args.coverage_mix),
            "--coverage-power",
            str(args.coverage_power),
            "--initial-action-bias",
            str(args.initial_action_bias),
            "--initial-policy-profile",
            args.initial_policy_profile,
            "--exploration-curriculum",
            args.exploration_curriculum,
            "--reward-clip",
            str(args.reward_clip),
            "--replay-epochs",
            str(args.replay_epochs),
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
            "--residual-split-prior-epochs",
            str(args.residual_split_prior_epochs),
            "--residual-split-root-objective",
            args.residual_split_root_objective,
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
            *training_runtime_arguments(args),
        ]
        job_index = protocols.index(protocol) * len(folds) + folds.index(fold)
        command = set_cli_option(
            command,
            "--ai-port",
            args.ai_port + job_index,
        )
        if args.workload == "STACK" and args.trainer_device_auto:
            command = set_cli_option(
                command,
                "--trainer-device",
                f"cuda:{job_index % 8}",
            )
        if args.independent_action_summary is not None:
            command.extend(
                [
                    "--independent-action-summary",
                    str(args.independent_action_summary),
                ]
            )
        if args.independent_prior_cost_regression:
            command.append("--independent-prior-cost-regression")
        if args.bootstrap_only_independent_prior:
            command.append("--bootstrap-only-independent-prior")
        if args.replay_action_cost_regression:
            command.append("--replay-action-cost-regression")
        for phase in ("dec", "sched", "enum", "adapt"):
            phase_epochs = getattr(args, f"independent_{phase}_prior_epochs")
            if phase_epochs is not None:
                command.extend(
                    [
                        f"--independent-{phase}-prior-epochs",
                        str(phase_epochs),
                    ]
                )
        if args.disable_runtime_replay:
            command.append("--disable-runtime-replay")
        if args.disable_bootstrap_fixed_replay:
            command.append("--disable-bootstrap-fixed-replay")
        for phase in ("dec", "sched", "enum", "adapt"):
            phase_epochs = getattr(args, f"{phase}_replay_epochs")
            if phase_epochs is not None:
                command.extend([f"--{phase}-replay-epochs", str(phase_epochs)])
            phase_importance_power = getattr(
                args, f"{phase}_replay_importance_power"
            )
            if phase_importance_power is not None:
                command.extend(
                    [
                        f"--{phase}-replay-importance-power",
                        str(phase_importance_power),
                    ]
                )
        for replay_group in fixed_replay_groups:
            command.extend(["--fixed-replay-group", replay_group])
        if args.eval_test_limit is not None:
            command.extend(["--test-limit", str(args.eval_test_limit)])
        run_command(command)

        training_dir = args.output_root / args.workload.lower() / training_id
        primary_history = load_jsonl(training_dir / "learning_curve.jsonl")
        primary_state = json.loads(
            (training_dir / "training_state.json").read_text(encoding="utf-8")
        )
        refinement_history = None
        refinement_state = None
        if args.refinement_iterations > 0:
            refinement_id = f"{training_id}-refine"
            refinement_command = list(command)
            for option, value in (
                ("--experiment-id", refinement_id),
                ("--iterations", args.refinement_iterations),
                ("--learning-rate", args.refinement_learning_rate),
                ("--ppo-epochs", args.refinement_ppo_epochs),
                ("--entropy", args.refinement_entropy),
                (
                    "--exploration-curriculum",
                    args.refinement_exploration_curriculum,
                ),
                ("--dec-replay-epochs", args.refinement_dec_replay_epochs),
                (
                    "--dec-replay-importance-power",
                    args.refinement_dec_replay_importance_power,
                ),
                ("--initial-checkpoint", primary_state["best_checkpoint"]),
            ):
                refinement_command = set_cli_option(
                    refinement_command,
                    option,
                    value,
                )
            if "--disable-bootstrap-fixed-replay" not in refinement_command:
                refinement_command.append("--disable-bootstrap-fixed-replay")
            run_command(refinement_command)
            refinement_dir = args.output_root / args.workload.lower() / refinement_id
            refinement_history = load_jsonl(
                refinement_dir / "learning_curve.jsonl"
            )
            refinement_state = json.loads(
                (refinement_dir / "training_state.json").read_text(encoding="utf-8")
            )

        combined_history, combined_state = combine_training_stages(
            primary_history,
            primary_state,
            refinement_history,
            refinement_state,
            primary_iterations=args.iterations,
        )
        return protocol, fold, combined_history, combined_state

    training_jobs = [
        (protocol, fold) for protocol in protocols for fold in folds
    ]
    if not args.skip_training and training_jobs:
        workers = min(args.training_parallelism, len(training_jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(train_fold, protocol, fold): (protocol, fold)
                for protocol, fold in training_jobs
            }
            for future in as_completed(futures):
                protocol, fold, history, state = future.result()
                training_histories[protocol][fold] = history
                training_states[protocol][fold] = state

    report: dict[str, Any] = {
        "experiment_id": matrix_id,
        "workload": args.workload,
        "experience_db": str(dataset_experience_db),
        "experience_sharing": "one live dataset store across protocols and folds",
        "baseline": str(baseline_path),
        "action_config": str(action_config_path),
        "action_config_hash": frozen_action_config["config_hash"],
        "standalone_actions": frozen_action_config.get("standalone_actions", {}),
        "protocols": {},
    }
    action_rows = [
        {
            "mechanism": mechanism,
            "query_count": entry.get("query_count"),
            "pg_total_s": (
                float(entry["pg_total_ms"]) / 1000.0
                if entry.get("pg_total_ms") is not None
                else None
            ),
            "action_total_s": (
                float(entry["action_total_ms"]) / 1000.0
                if entry.get("action_total_ms") is not None
                else None
            ),
            "workload_speedup": entry.get("workload_speedup"),
            "geometric_mean_speedup": entry.get("geometric_mean_speedup"),
            "improved_pct": entry.get("improved_pct"),
            "timeouts": entry.get("timeouts"),
            "wrong_results": entry.get("wrong_results"),
            "errors": entry.get("errors"),
            "action_config_hash": frozen_action_config["config_hash"],
        }
        for mechanism, entry in sorted(
            frozen_action_config.get("standalone_actions", {}).items()
        )
    ]
    curve_rows = []
    selected_rows = []
    for protocol in protocols:
        protocol_report: dict[str, Any] = {
            "test_multiplicity": matrix_manifest["test_multiplicity"][protocol],
            "action_config_hash": frozen_action_config["config_hash"],
        }
        if training_histories[protocol]:
            curve = aggregate_learning_curves(training_histories[protocol])
            selected = aggregate_selected_checkpoints(
                training_histories[protocol],
                training_states[protocol],
            )
            protocol_report["learning_curve"] = curve
            protocol_report["selected_checkpoints"] = selected
            for entry in curve:
                curve_rows.append({"protocol": protocol, **entry})
            selected_rows.append(
                {
                    "protocol": protocol,
                    "test_query_instances": matrix_manifest["test_multiplicity"][
                        protocol
                    ]["instances"],
                    "test_unique_queries": matrix_manifest["test_multiplicity"][
                        protocol
                    ]["unique"],
                    **{key: value for key, value in selected.items() if key != "folds"},
                }
            )
        report["protocols"][protocol] = protocol_report

    report_path = matrix_dir / "online_matrix_report.json"
    write_json_atomic(report_path, report)
    if action_rows:
        write_csv_atomic(
            matrix_dir / "standalone_actions.csv",
            action_rows,
            action_rows[0].keys(),
        )
    if curve_rows:
        write_csv_atomic(
            matrix_dir / "learning_curve.csv",
            curve_rows,
            curve_rows[0].keys(),
        )
    if selected_rows:
        write_csv_atomic(
            matrix_dir / "selected_test_results.csv",
            selected_rows,
            selected_rows[0].keys(),
        )
    print(json.dumps({"report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
