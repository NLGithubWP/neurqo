"""Collect and train a registered dataset inside the release container."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarking.database_runner import create_catalog_snapshot
from benchmarking.orchestration import run_command
from benchmarking.utils import write_json_atomic
from benchmarking.workloads import (
    ROOT,
    SPLIT_PROTOCOLS,
    WORKLOAD_DATABASES,
    connect,
    query_sql,
    split_folds,
    workload_query_ids,
)
from optimization.actions import workload_metrics
from optimization.decomposition_eligibility import summarize_query_compatibility
from optimization.query_compatibility import classify_spj_compatibility


def selected_folds(dataset: str, protocol: str, fold: str) -> list[dict]:
    workload = dataset.upper()
    protocols = (
        SPLIT_PROTOCOLS[workload]
        if protocol == "all"
        else (protocol.replace("-", "_"),)
    )
    if any(item not in SPLIT_PROTOCOLS[workload] for item in protocols):
        raise ValueError(f"unsupported protocol for {dataset}: {protocol}")
    selected = []
    for item in protocols:
        for suffix in "abc" if fold == "all" else fold:
            spec = split_folds(workload, item)[f"{item}_{suffix}"]
            if (
                not spec["train"]
                or not spec["test"]
                or set(spec["train"]) & set(spec["test"])
            ):
                raise ValueError(
                    f"empty or overlapping train/test split: {item}/{suffix}"
                )
            selected.append({"protocol": item, "fold": suffix, **spec})
    return selected


def compatibility_manifest(workload: str, queries: dict[str, str]) -> dict:
    records = {qid: classify_spj_compatibility(sql) for qid, sql in queries.items()}
    summary = summarize_query_compatibility(workload, records)
    return {
        "schema_version": 2,
        "workloads": {
            workload: {
                "query_count": summary.query_count,
                "spj_compatible_query_count": summary.compatible_query_count,
                "spj_compatible_query_ratio": summary.compatible_query_ratio,
                "recommend_decomposition_enabled": summary.enabled,
                "queries": records,
            }
        },
    }


def common_arguments(request: dict) -> list[str]:
    workspace = Path(request["workspace"])
    result = [
        "--workload",
        request["dataset"].upper(),
        "--host",
        "127.0.0.1",
        "--port",
        "5432",
        "--user",
        "neurdb",
        "--container",
        "local",
        "--pgdb-root",
        str(workspace),
        "--experience-db",
        request["experience_db"],
        "--output-root",
        str(workspace / "reports"),
        "--ai-port",
        str(request["ai_port"]),
    ]
    if request.get("action_config"):
        result.extend(["--action-config", str(workspace / "action-config.json")])
    return result


def collector_command(
    request: dict,
    *,
    experiment: str,
    profiles: str,
    baseline: Path | None = None,
    **options,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarking.action_runner",
        *common_arguments(request),
        "--experiment-id",
        experiment,
        "--profiles",
        profiles,
        "--warmups",
        "0",
        "--measurements",
        "1",
        "--correctness",
        "strict",
    ]
    if baseline is not None:
        command.extend(["--baseline-json", str(baseline)])
    for key, value in options.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            command.extend(["--" + key.replace("_", "-"), str(item)])
    return command


def training_command(request: dict, spec: dict, baseline: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarking.iterative_training",
        *common_arguments(request),
        "--experiment-id",
        f"{spec['protocol']}-{spec['fold']}",
        "--runtime-project",
        str(ROOT),
        "--protocol",
        spec["protocol"],
        "--fold",
        spec["fold"],
        "--baseline-json",
        str(baseline),
        "--baseline-first-episodes",
        str(baseline.parent / "episodes.csv"),
        "--fixed-replay-group",
        request["replay_group"],
        "--bootstrap-replay-epochs",
        str(request["pretrain_epochs"]),
        "--replay-epochs",
        "4",
        "--iterations",
        str(request["iterations"]),
        "--eval-every",
        str(request["eval_every"]),
        "--trainer-container",
        "local",
        "--trainer-device",
        "cpu",
        "--model-device",
        "cpu",
        "--sql-execution-slots",
        "1",
        "--formal-execution-cache",
        "read-write",
    ]


def ensure_valid(summary: dict) -> None:
    for name, profile in summary.items():
        if not profile["metrics"]["valid"]:
            raise RuntimeError(
                f"invalid results in {name}; inspect the saved episodes before training"
            )


def collect_and_train(request: dict) -> None:
    workspace = Path(request["workspace"])
    workload = request["dataset"].upper()
    specs = selected_folds(workload, request["protocol"], request["fold"])
    queries = {qid: query_sql(workload, qid) for qid in workload_query_ids(workload)}
    analysis = compatibility_manifest(workload, queries)
    write_json_atomic(workspace / "compatibility.json", analysis)
    if request.get("action_config"):
        write_json_atomic(workspace / "action-config.json", request["action_config"])
    connection = connect(workload, host="127.0.0.1", port=5432, user="neurdb")
    try:
        with connection.cursor() as cursor:
            cursor.execute((ROOT / "config/setup_database.sql").read_text())
    finally:
        connection.close()
    create_catalog_snapshot(
        SimpleNamespace(workload=workload, host="127.0.0.1", port=5432, user="neurdb"),
        workspace / "catalog.json",
    )
    report_root = workspace / "reports" / workload.lower()
    baseline = report_root / "postgres" / "baseline.json"
    run_command(
        collector_command(request, experiment="postgres", profiles="pg", role="all")
    )
    profiles = ["neurqo_none", "top5", "lip_selective", "aja_conservative"]
    if analysis["workloads"][workload]["recommend_decomposition_enabled"]:
        profiles.insert(1, "query_split")
    # The no-action NeurQO trajectory supplies a comparable stop/native label.
    run_command(
        collector_command(
            request,
            experiment="actions",
            profiles=",".join(profiles),
            baseline=baseline,
            role="all",
            replay_group=request["replay_group"],
            catalog_path=workspace / "catalog.json",
            execution_cache="read-write",
        )
    )
    ensure_valid(json.loads((report_root / "actions/summary.json").read_text()))
    deployment = {
        **request,
        "database": WORKLOAD_DATABASES[workload],
        "models": {},
        "query_hashes": {
            qid: hashlib.sha256(sql.encode()).hexdigest()
            for qid, sql in queries.items()
        },
        "folds": specs,
        "baseline": str(baseline),
    }
    for spec in specs:
        run_command(training_command(request, spec, baseline))
        key = f"{spec['protocol']}/{spec['fold']}"
        state = json.loads(
            (
                report_root
                / f"{spec['protocol']}-{spec['fold']}"
                / "training_state.json"
            ).read_text()
        )
        selected = Path(state["formal_best_checkpoint"])
        model = workspace / "models" / key / "best.pt"
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, model)
        deployment["models"][key] = str(model)
        write_json_atomic(workspace / "deployment.json", deployment)


def aggregate_folds(folds: list[dict], baseline: dict) -> list[dict]:
    summaries = []
    for protocol in dict.fromkeys(fold["protocol"] for fold in folds):
        reference, measured = {}, {}
        for fold in folds:
            if fold["protocol"] != protocol:
                continue
            for qid, value in fold["queries"].items():
                key = f"{fold['fold']}:{qid}"
                reference[key], measured[key] = baseline[qid], value
        metrics = workload_metrics(measured, reference)
        summaries.append(
            {
                "protocol": protocol.replace("_", "-"),
                **metrics,
                "ws": metrics["workload_speedup"] if metrics["valid"] else None,
                "gs": metrics["geometric_mean_speedup"],
                "imp_pct": metrics["improved_pct"],
            }
        )
    return summaries


def benchmark(request: dict) -> None:
    workspace = Path(request["workspace"])
    deployment = json.loads((workspace / "deployment.json").read_text())
    specs = selected_folds(request["dataset"], request["protocol"], request["fold"])
    baseline = json.loads(Path(deployment["baseline"]).read_text())
    folds = []
    for spec in specs:
        key = f"{spec['protocol']}/{spec['fold']}"
        model = deployment["models"][key]
        ids = request.get("query") or spec["test"]
        experiment = f"test-{request['run_id']}-{spec['protocol']}-{spec['fold']}"
        run_command(
            collector_command(
                deployment,
                experiment=experiment,
                profiles="learned",
                baseline=Path(deployment["baseline"]),
                protocol=spec["protocol"],
                fold=spec["fold"],
                role="test",
                query_id=ids,
                model_path=model,
                policy_version="custom-test",
                catalog_path=workspace / "catalog.json",
                execution_cache="read-write",
            )
        )
        summary = json.loads(
            (
                workspace / "reports" / request["dataset"] / experiment / "summary.json"
            ).read_text()
        )
        ensure_valid(summary)
        queries = summary["learned@custom-test"]["queries"]
        records = []
        for qid, value in queries.items():
            records.append(
                {
                    "query_id": qid,
                    "pg": {
                        "status": "ok",
                        "wall_ms": baseline[qid]["median_charged_ms"],
                    },
                    "neurqo": {
                        "status": value["status"],
                        "wall_ms": value["official_client_wall_ms"],
                    },
                    "correct": value["status"] == "ok",
                }
            )
        folds.append(
            {
                "protocol": spec["protocol"],
                "fold": spec["fold"],
                "checkpoint": model,
                "queries": queries,
                "records": records,
            }
        )
    report = {
        "dataset": request["dataset"],
        "folds": folds,
        "summary": aggregate_folds(folds, baseline),
        "baseline_source": deployment["baseline"],
        "execution_cache": "read-write",
    }
    write_json_atomic(workspace / f"benchmark-{request['run_id']}.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def cancel(workspace: Path, run_id: str) -> None:
    path = workspace / "active.json"
    if not path.is_file():
        return
    state = json.loads(path.read_text())
    pid = int(state["pid"])
    process = Path(f"/proc/{pid}")
    if state["run_id"] != run_id or not process.exists():
        return
    if process.joinpath("stat").read_text().split()[21] != state["start_ticks"]:
        return
    os.killpg(pid, signal.SIGINT)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args(argv)
    if args.cancel:
        cancel(args.workspace, args.run_id)
        return 0
    request = json.load(sys.stdin)
    if (
        Path(request["workspace"]).resolve() != args.workspace.resolve()
        or request["run_id"] != args.run_id
    ):
        raise ValueError("request identity mismatch")
    os.environ["NEURQO_LOCAL_RUNTIME"] = "1"
    os.environ["NEURQO_WORKLOAD_CENTER_ANALYSIS"] = str(
        args.workspace / "compatibility.json"
    )
    os.chdir(ROOT)
    if os.getpgrp() != os.getpid():
        os.setsid()
    # One foreground workflow owns this dataset's writable buffer and policy port.
    lock_path = Path(request["experience_db"]).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = args.workspace / "active.json"
        write_json_atomic(
            active,
            {
                "pid": os.getpid(),
                "run_id": args.run_id,
                "start_ticks": Path("/proc/self/stat").read_text().split()[21],
            },
        )
        try:
            if request["operation"] == "train":
                collect_and_train(request)
            elif request["operation"] == "benchmark":
                benchmark(request)
            else:
                raise ValueError("unknown workflow operation")
        finally:
            active.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
