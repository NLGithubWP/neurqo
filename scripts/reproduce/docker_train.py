"""Collect and train a registered dataset using the released Docker environment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = "benchmarking.dataset_training"


def docker_python(container, code, *arguments, **kwargs):
    return subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "neurdb",
            container,
            "python",
            "-c",
            code,
            *map(str, arguments),
        ],
        check=True,
        **kwargs,
    )


def validate_dataset(dataset, root=ROOT):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset):
        raise ValueError("dataset must be a lowercase identifier")
    tree = ast.parse((root / "workloads/train_test.py").read_text())
    splits = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SPLITS"
            for target in node.targets
        )
    )
    if dataset.upper() not in splits:
        raise ValueError(f"register {dataset.upper()} in workloads/train_test.py first")
    if not (root / "workloads" / f"query_{dataset}").is_dir():
        raise ValueError(f"missing workloads/query_{dataset}")


def stage_source(container, workspace, root=ROOT):
    """Stage only Python/query/config inputs; never replace bundled services."""
    relatives = [
        "src",
        "config",
        "workloads",
        "scripts/reproduce/common",
        "scripts/reproduce/neurqo/workload_fk_center_analysis.json",
    ]
    stop_file = root / ".local/STOP_ALL_TRAINING"
    if stop_file.exists():
        raise RuntimeError(f"training is disabled by {stop_file}")
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for relative in relatives:
                tar.add(
                    root / relative,
                    arcname=relative,
                    filter=lambda info: (
                        None
                        if "__pycache__" in Path(info.name).parts
                        or info.name.endswith(".pyc")
                        else info
                    ),
                )
        archive.seek(0)
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-u",
                "neurdb",
                container,
                "python",
                "-c",
                "import sys,tarfile; from pathlib import Path; "
                "p=Path(sys.argv[1]); p.mkdir(parents=True); "
                "tarfile.open(fileobj=sys.stdin.buffer,mode='r|').extractall(p)",
                str(Path(workspace) / "source"),
            ],
            stdin=archive,
            check=True,
        )


def run_worker(container, request, output):
    source = str(Path(request["workspace"]) / "source")
    command = [
        "docker",
        "exec",
        "-i",
        "-u",
        "neurdb",
        "-e",
        f"PYTHONPATH={source}/src",
        "-w",
        source,
        container,
        "python",
        "-u",
        "-m",
        MODULE,
        "--workspace",
        request["workspace"],
        "--run-id",
        request["run_id"],
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        with (output / "progress.log").open("a") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        return process.wait()
    except KeyboardInterrupt:
        cancel_command = [item for item in command if item != "-i"] + ["--cancel"]
        subprocess.run(cancel_command, check=False, timeout=20)
        try:
            process.wait(timeout=30)
            return 130
        except subprocess.TimeoutExpired:
            # Signal only the explicitly identified workflow, never other services.
            docker_python(
                container,
                "import json,os,signal,sys; from pathlib import Path; "
                "p=Path(sys.argv[1])/'active.json'; s=json.loads(p.read_text()) if p.exists() else {}; "
                "pid=s.get('pid'); proc=Path('/proc')/str(pid); "
                "ok=s.get('run_id')==sys.argv[2] and proc.exists() and "
                "(proc/'stat').read_text().split()[21]==s.get('start_ticks'); "
                "os.killpg(pid,signal.SIGKILL) if ok else None",
                request["workspace"],
                request["run_id"],
            )
            process.wait(timeout=10)
            return 130


def copy_from(container, source, target):
    subprocess.run(["docker", "cp", f"{container}:{source}", str(target)], check=True)


def benchmark(args):
    """Custom-model branch of docker_benchmark; released-model behavior is unchanged."""
    run_dir = args.training_run.resolve()
    deployment = json.loads((run_dir / "deployment.json").read_text())
    if deployment["dataset"] != args.dataset:
        raise ValueError("--dataset does not match the training run")
    protocols = (
        list(dict.fromkeys(spec["protocol"] for spec in deployment["folds"]))
        if args.protocol == "all"
        else [args.protocol.replace("-", "_")]
    )
    selected = []
    for protocol in protocols:
        for fold in "abc" if args.fold == "all" else args.fold:
            key = f"{protocol}/{fold}"
            if key not in deployment["models"]:
                raise ValueError(f"no trained checkpoint for {key}")
            spec = next(
                item
                for item in deployment["folds"]
                if item["protocol"] == protocol and item["fold"] == fold
            )
            ids = args.query or spec["test"]
            for qid in ids:
                if Path(qid).name != qid or qid not in deployment["query_hashes"]:
                    raise ValueError(f"unknown query: {qid}")
                sql = (
                    ROOT / "workloads" / f"query_{args.dataset}" / f"{qid}.sql"
                ).read_text()
                sql = re.sub(r"/\*\+.*?\*/", "", sql, flags=re.DOTALL)
                if (
                    hashlib.sha256(sql.encode()).hexdigest()
                    != deployment["query_hashes"][qid]
                ):
                    raise ValueError(f"SQL changed since collection: {qid}")
            selected.append({"protocol": protocol, "fold": fold, "query_ids": ids})
    if args.list:
        print(json.dumps(selected, indent=2))
        return 0
    output = args.output or ROOT / "results" / (
        "benchmark_neurqo_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    request = {
        **deployment,
        "operation": "benchmark",
        "run_id": uuid.uuid4().hex,
        "protocol": args.protocol,
        "fold": args.fold,
        "query": args.query,
    }
    code = run_worker(args.container, request, output)
    if code == 0:
        copy_from(
            args.container,
            f"{request['workspace']}/benchmark-{request['run_id']}.json",
            output / "results.json",
        )
        from scripts.reproduce.docker_benchmark import write_outputs

        report = json.loads((output / "results.json").read_text())
        write_outputs(output, report)
        for summary in report["summary"]:
            print(f"{args.dataset}/{summary['protocol']}: WS={summary['ws']:.3f}x")
    print(f"Saved: {output.resolve()}")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="neurqo")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--protocol",
        choices=("all", "random", "base-query", "leave-one-out"),
        default="all",
    )
    parser.add_argument("--fold", choices=("all", "a", "b", "c"), default="all")
    parser.add_argument("--pretrain-epochs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--ai-port", type=int, default=18140)
    parser.add_argument("--action-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume a previously created training output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request without contacting Docker",
    )
    args = parser.parse_args(argv)
    try:
        validate_dataset(args.dataset)
        if min(args.iterations, args.pretrain_epochs, args.eval_every) < 1:
            raise ValueError(
                "epochs, iterations and evaluation interval must be positive"
            )
        if not 1024 <= args.ai_port <= 65535:
            raise ValueError("invalid policy port")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (
        args.resume or args.output or ROOT / "results/training" / args.dataset / stamp
    )
    if args.resume:
        request = json.loads((output / "run.json").read_text())
        if request["dataset"] != args.dataset or request["container"] != args.container:
            parser.error("resume requires the original dataset and container")
        request["iterations"] = max(request["iterations"], args.iterations)
        request["run_id"] = uuid.uuid4().hex
    else:
        run_id = uuid.uuid4().hex
        request = {
            "operation": "train",
            "container": args.container,
            "dataset": args.dataset,
            "workspace": f"/opt/neurqo/runtime/training/{args.dataset}/{run_id}",
            "run_id": run_id,
            "experience_db": f"/opt/neurqo/runtime/training/experience/{args.dataset}.sqlite",
            "replay_group": f"{args.dataset}:{run_id}:fixed",
            "protocol": args.protocol,
            "fold": args.fold,
            "pretrain_epochs": args.pretrain_epochs,
            "iterations": args.iterations,
            "eval_every": args.eval_every,
            "ai_port": args.ai_port,
            "action_config": (
                json.loads(args.action_config.read_text())
                if args.action_config
                else None
            ),
        }
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    if (ROOT / ".local/STOP_ALL_TRAINING").exists():
        parser.error("training is disabled by .local/STOP_ALL_TRAINING")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    if not args.resume:
        stage_source(args.container, request["workspace"])
    (output / "run.json").write_text(json.dumps(request, indent=2) + "\n")
    code = run_worker(args.container, request, output)
    # Raw logs/checkpoints remain in the container, including on interruption.
    if code == 0:
        for name in ("deployment.json", "catalog.json", "compatibility.json"):
            copy_from(args.container, f"{request['workspace']}/{name}", output / name)
        for name in ("models", "reports"):
            (output / name).mkdir(exist_ok=True)
            copy_from(args.container, f"{request['workspace']}/{name}/.", output / name)
    print(f"Saved: {output.resolve()}\nContainer workspace: {request['workspace']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
