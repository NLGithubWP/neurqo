"""Shared process and runtime orchestration for benchmark workflows."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from benchmarking.utils import safe_name


ROOT = Path(__file__).resolve().parents[2]
ACTION_RUNNER_MODULE = "benchmarking.action_runner"


def cleanup_docker_exec_child(command: list[str]) -> None:
    """Terminate a detached container child left by an interrupted docker exec."""
    if command[:2] != ["docker", "exec"] or "--output" not in command:
        return
    output_index = command.index("--output")
    if output_index + 1 >= len(command):
        return
    container_index = 2
    options_with_value = {
        "-e",
        "--env",
        "--env-file",
        "-u",
        "--user",
        "-w",
        "--workdir",
    }
    while container_index < len(command):
        token = command[container_index]
        if token in options_with_value:
            container_index += 2
            continue
        if token.startswith("-"):
            container_index += 1
            continue
        break
    if container_index >= len(command):
        return
    container = command[container_index]
    pattern = command[output_index + 1]
    if not pattern:
        return
    # The bracketed first character keeps pkill from matching its own argv.
    regex = f"[{pattern[0]}]{pattern[1:]}"
    for signal_name in ("TERM", "KILL"):
        subprocess.run(
            [
                "docker",
                "exec",
                container,
                "pkill",
                f"-{signal_name}",
                "-f",
                regex,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if signal_name == "TERM":
            try:
                process_wait = subprocess.run(
                    [
                        "docker",
                        "exec",
                        container,
                        "pgrep",
                        "-f",
                        regex,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                continue
            if process_wait.returncode != 0:
                break


def run_command(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        start_new_session=True,
    )
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        cleanup_docker_exec_child(command)
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def sync_runtime(pgdb_root: Path, runtime_project: Path) -> None:
    runtime_project.mkdir(parents=True, exist_ok=True)
    for relative in (
        "src",
        "config",
        "workloads",
        "tools",
        "tests",
        "scripts/nqo_benchmark.py",
        "scripts/reproduce/common",
        "scripts/reproduce/nqo/workload_fk_center_analysis.json",
    ):
        source = ROOT / relative
        if not source.exists():
            continue
        destination = runtime_project / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            source_arg = f"{source}/"
            destination_arg = f"{destination}/"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_arg = str(source)
            destination_arg = str(destination)
        run_command(
            [
                "rsync",
                "-a",
                "--delete",
                "--no-owner",
                "--no-group",
                "--exclude=__pycache__/",
                source_arg,
                destination_arg,
            ],
            cwd=ROOT,
        )
    # Query split manifests import the immutable raw/derived workload inputs
    # through ROOT/results.  Isolated source snapshots share the already
    # staged copy without touching the default runtime's source tree.
    default_runtime = pgdb_root / ".nqo_runtime" / "nqo"
    shared_results = default_runtime / "results"
    runtime_results = runtime_project / "results"
    if (
        runtime_project != default_runtime
        and shared_results.is_dir()
        and not runtime_results.exists()
    ):
        runtime_results.symlink_to(
            os.path.relpath(shared_results, runtime_project),
            target_is_directory=True,
        )
    action_server = runtime_project / "src/runtime/action_server.py"
    if not action_server.is_file():
        raise FileNotFoundError(action_server)


def host_to_container(path: Path, pgdb_root: Path) -> str:
    relative = path.resolve().relative_to(pgdb_root.resolve())
    return str(Path("/code/pgdb-dev") / relative)


def runner_run_id(
    experiment_id: str,
    profile_label: str,
    protocol: str,
    fold: str,
    role: str,
) -> str:
    return safe_name(f"run_{experiment_id}_{profile_label}_{protocol}_{fold}_{role}")


def query_arguments(query_ids: Iterable[str]) -> list[str]:
    result: list[str] = []
    for query_id in query_ids:
        result.extend(["--query-id", query_id])
    return result


def action_runner_command(
    args: argparse.Namespace,
    *,
    experiment_id: str,
    profile: str,
    role: str,
    query_ids: Iterable[str],
    experience_db: Path,
    baseline_json: Path | None = None,
    model_path: Path | None = None,
    policy_version: str | None = None,
    inference_mode: str = "deterministic",
    temperature: float = 1.0,
    exploration_epsilon: float = 0.0,
    coverage_counts: Path | None = None,
    coverage_mix: float = 0.0,
    coverage_power: float = 0.5,
    stochastic_heads: str | None = None,
    replay_group: str | None = None,
    sampling_seed: int | None = None,
    warmups: int = 0,
    measurements: int = 1,
    execution_cache: str = "off",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        ACTION_RUNNER_MODULE,
        "--workload",
        args.workload,
        "--profiles",
        profile,
        "--protocol",
        args.protocol,
        "--fold",
        args.fold,
        "--role",
        role,
        "--experiment-id",
        experiment_id,
        "--output-root",
        str(args.output_root),
        "--experience-db",
        str(experience_db),
        "--catalog-path",
        str(args.catalog_snapshot),
        "--warmups",
        str(warmups),
        "--measurements",
        str(measurements),
        "--execution-cache",
        execution_cache,
        "--cache-match-mode",
        args.cache_match_mode,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--user",
        args.user,
        "--container",
        args.container,
        "--pgdb-root",
        str(args.pgdb_root),
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
        "--aja-conservative-rows",
        str(args.aja_conservative_rows),
        "--aja-aggressive-rows",
        str(args.aja_aggressive_rows),
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
        "--lip-min-probe-ratio",
        str(args.lip_min_probe_ratio),
        "--lip-max-filters",
        str(args.lip_max_filters),
        "--sched-alpha",
        str(args.initial_sched_alpha),
        "--correctness",
        "strict",
        *query_arguments(query_ids),
    ]
    if baseline_json is not None:
        command.extend(["--baseline-json", str(baseline_json)])
    if getattr(args, "sql_execution_lock", None) is not None:
        command.extend(["--sql-execution-lock", str(args.sql_execution_lock)])
        command.extend(
            ["--sql-execution-slots", str(args.sql_execution_slots)]
        )
    if getattr(args, "action_config", None) is not None:
        command.extend(["--action-config", str(args.action_config)])
    if replay_group is not None:
        command.extend(["--replay-group", replay_group])
    sched_alpha_sequence = getattr(args, "sched_alpha_sequence", "")
    if sched_alpha_sequence:
        command.extend(["--sched-alpha-sequence", sched_alpha_sequence])
    if model_path is not None:
        if policy_version is None:
            raise ValueError("model runs require a policy version")
        command.extend(
            [
                "--model-path",
                str(model_path),
                "--model-method",
                "standardmdp_rl",
                "--model-hidden",
                str(args.hidden),
                "--model-device",
                args.model_device,
                "--model-nqo-src",
                (
                    host_to_container(args.runtime_project, args.pgdb_root)
                    + "/src"
                ),
                "--inference-mode",
                inference_mode,
                "--temperature",
                str(temperature),
                "--exploration-epsilon",
                str(exploration_epsilon),
                "--sampling-seed",
                str(args.seed if sampling_seed is None else sampling_seed),
                "--policy-version",
                policy_version,
                "--action-ablation",
                args.action_ablation,
            ]
        )
        if coverage_counts is not None:
            command.extend(
                [
                    "--coverage-counts",
                    str(coverage_counts),
                    "--coverage-mix",
                    str(coverage_mix),
                    "--coverage-power",
                    str(coverage_power),
                ]
            )
        if stochastic_heads is not None:
            command.extend(["--stochastic-heads", stochastic_heads])
    return command
