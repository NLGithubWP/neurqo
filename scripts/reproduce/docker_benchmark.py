"""Send repository SQL to the self-contained NeurQO Docker release."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = {
    "job": ("base-query", "leave-one-out", "random"),
    "stack": ("base-query", "leave-one-out", "random"),
    "tpch": ("random",),
}
WORKER = "/opt/neurqo/docker_worker.py"
EVENT_PREFIX = "NEURQO_EVENT "


def build_request(args, root=ROOT):
    split_file = root / "workloads/train_test.py"
    tree = ast.parse(split_file.read_text())
    splits = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SPLITS" for t in node.targets)
    )
    protocols = PROTOCOLS[args.dataset] if args.protocol == "all" else (args.protocol,)
    if any(p not in PROTOCOLS[args.dataset] for p in protocols):
        raise ValueError(f"Unsupported protocol for {args.dataset}: {args.protocol}")
    folds = "abc" if args.fold == "all" else args.fold
    selected, queries = [], {}
    for protocol in protocols:
        for fold in folds:
            key = f"{protocol.replace('-', '_')}_{fold}"
            ids = args.query or splits[args.dataset.upper()][key]["test"]
            ids = [name[:-4] if name.endswith(".sql") else name for name in ids]
            if len(set(ids)) != len(ids):
                raise ValueError("Repeated query in a fold")
            for query_id in ids:
                if (
                    not query_id
                    or Path(query_id).name != query_id
                    or query_id in (".", "..")
                ):
                    raise ValueError(f"Invalid query ID: {query_id!r}")
                path = root / "workloads" / f"query_{args.dataset}" / f"{query_id}.sql"
                raw = path.read_bytes()
                queries[query_id] = {
                    "sql": raw.decode(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            selected.append({"protocol": protocol, "fold": fold, "query_ids": ids})
    return {
        "version": 1,
        "dataset": args.dataset,
        "folds": selected,
        "queries": queries,
        "scope": "queries" if args.query else "test",
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
    }


def write_outputs(output, report):
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps(report.get("summary", []), indent=2) + "\n"
    )
    rows = []
    for fold in report.get("folds", []):
        for record in fold.get("records", []):
            pg, nq = record["pg"], record.get("neurqo", {})
            charged = nq.get("wall_ms")
            if nq.get("status") == "timeout" and pg["status"] == "ok":
                charged = min(360000.0, 5 * pg["wall_ms"])
            rows.append(
                {
                    "dataset": report["dataset"],
                    "protocol": fold["protocol"],
                    "fold": fold["fold"],
                    "query_id": record["query_id"],
                    "pg_status": pg["status"],
                    "pg_first_ms": pg["wall_ms"],
                    "neurqo_status": nq.get("status"),
                    "neurqo_wall_ms": nq.get("wall_ms"),
                    "neurqo_charged_ms": charged,
                    "correct": record.get("correct"),
                    "checkpoint": fold["checkpoint"],
                }
            )
    pg_rows = [
        dict(query_id=q, **r)
        for q, runs in report.get("pg_runs", {}).items()
        for r in runs
    ]
    for name, values in (("per-query.csv", rows), ("pg-executions.csv", pg_rows)):
        if values:
            fields = list(dict.fromkeys(k for row in values for k in row))
            with (output / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)


def cancel(container, run_id):
    subprocess.run(
        ["docker", "exec", container, "python", WORKER, "--cancel", run_id],
        check=False,
        timeout=15,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="neurqo")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--training-run",
        type=Path,
        help="output directory from docker_train.py for a custom model",
    )
    parser.add_argument(
        "--protocol",
        choices=("base-query", "leave-one-out", "random", "all"),
        default="random",
    )
    parser.add_argument("--fold", choices=("a", "b", "c", "all"), default="a")
    parser.add_argument(
        "--query",
        action="append",
        help="one canonical query ID; repeat to select several",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list selected local SQL files without contacting Docker",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.training_run is not None:
        sys.path.insert(0, str(ROOT))
        from scripts.reproduce.docker_train import benchmark

        try:
            return benchmark(args)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    if args.dataset not in PROTOCOLS:
        parser.error("custom datasets require --training-run from docker_train.py")
    try:
        request = build_request(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.list:
        print(json.dumps(request["folds"], indent=2))
        return 0
    run_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or ROOT / "results" / f"benchmark_neurqo_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    command = [
        "docker",
        "exec",
        "-i",
        args.container,
        "python",
        "-u",
        WORKER,
        "--run-id",
        run_id,
    ]
    report = None
    print(f"Results: {output.resolve()}", flush=True)
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
        process.stdin = None
        with (output / "progress.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(EVENT_PREFIX):
                    event = json.loads(line[len(EVENT_PREFIX) :])
                    if event["type"] == "report":
                        report = event["report"]
                        write_outputs(output, report)
                    else:
                        print(event.get("message", json.dumps(event)), flush=True)
                else:
                    print(line, end="", flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        print("Cancelling this benchmark inside Docker...", file=sys.stderr)
        cancel(args.container, run_id)
        try:
            remaining, _ = process.communicate(timeout=15)
            for line in remaining.splitlines():
                if line.startswith(EVENT_PREFIX):
                    event = json.loads(line[len(EVENT_PREFIX) :])
                    if event["type"] == "report":
                        write_outputs(output, event["report"])
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        return 130
    if report is None:
        print(
            "No report received. Check container readiness and use the updated release image.",
            file=sys.stderr,
        )
        return code or 1
    for summary in report.get("summary", []):
        if summary["ws"] is not None:
            print(
                f"{args.dataset}/{summary['protocol']}: WS={summary['ws']:.3f}x GS={summary['gs']:.3f} Imp={summary['imp_pct']:.2f}%"
            )
        else:
            print(
                f"{args.dataset}/{summary['protocol']}: incomplete or invalid comparison"
            )
    print(f"Saved: {output.resolve()}")
    return code


if __name__ == "__main__":
    sys.exit(main())
