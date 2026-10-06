"""Container-side worker for docker_benchmark.py; never starts training."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

EVENT_PREFIX = "NEURQO_EVENT "
RUNTIME = Path("/opt/neurqo/runtime")


def emit(kind, **values):
    print(EVENT_PREFIX + json.dumps({"type": kind, **values}, default=str), flush=True)


def natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
    )


def validate(request, backend):
    dataset = request["dataset"]
    if request["version"] != 1 or dataset not in backend.PROTOCOLS:
        raise ValueError("Unsupported benchmark request")
    if request["scope"] not in ("queries", "test"):
        raise ValueError("Unsupported query selection")
    split = backend.ROOT / "source/workloads/train_test.py"
    if request["split_sha256"] != hashlib.sha256(split.read_bytes()).hexdigest():
        raise ValueError("Local and image train/test splits differ")
    expected = {
        (f["protocol"], f["fold"]): f for f in backend.plan() if f["dataset"] == dataset
    }
    seen, selected = set(), set()
    for fold in request["folds"]:
        key = fold["protocol"], fold["fold"]
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate fold")
        seen.add(key)
        ids = fold["query_ids"]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty fold or repeated query")
        if request["scope"] == "test" and ids != expected[key]["query_ids"]:
            raise ValueError("Local and image test query lists differ")
        selected.update(ids)
    if not seen or selected != set(request["queries"]):
        raise ValueError("Query payload differs from selected folds")
    for query_id, query in request["queries"].items():
        if not query_id or Path(query_id).name != query_id or query_id in (".", ".."):
            raise ValueError("Unsafe query ID")
        bundled = backend.ROOT / "workloads" / f"query_{dataset}" / f"{query_id}.sql"
        if query["sql"].encode() != bundled.read_bytes():
            raise ValueError(f"Local SQL differs from the release: {query_id}")
        if hashlib.sha256(query["sql"].encode()).hexdigest() != query["sha256"]:
            raise ValueError("SQL checksum mismatch")
    return expected


def same_result(pg, nq, query_id, dataset, backend, output):
    pair = {"query_id": query_id, "pg": pg, "neurqo": nq}
    equal = backend.compare(pair)
    if equal is False:
        backend.certify_known_limit_case(pair, dataset, output)
        equal = pair.get("correct", False)
    return equal, {
        k: pair[k]
        for k in ("correctness_method", "correctness_proof", "hash_equal")
        if k in pair
    }


def aggregate(report, backend):
    summaries = []
    for protocol in dict.fromkeys(f["protocol"] for f in report["folds"]):
        folds = [f for f in report["folds"] if f["protocol"] == protocol]
        records = [r for f in folds for r in f["records"]]
        summaries.append(
            {
                "protocol": protocol,
                "folds": [f["fold"] for f in folds],
                "complete_test_protocol": report["scope"] == "test"
                and {f["fold"] for f in folds} == set("abc"),
                **backend.summarize(records, sum(len(f["query_ids"]) for f in folds)),
            }
        )
    return summaries


def measure_pg(request, report, backend, output, save):
    dataset, database = report["dataset"], report["database"]
    ids = sorted(request["queries"], key=natural_key)
    for index, query_id in enumerate(ids, 1):
        runs = report["pg_runs"].setdefault(query_id, [])
        for repetition in range(3):
            result = backend.run_once(
                request["queries"][query_id]["sql"],
                dataset,
                database,
                False,
                output / "unused.db.jsonl",
            )
            result["repetition"] = repetition
            runs.append(result)
            if repetition and result["status"] == "ok":
                correct, _ = same_result(
                    runs[0], result, query_id, dataset, backend, output
                )
                if correct is False:
                    save()
                    raise ValueError(f"PG repetitions disagree: {query_id}")
            save()
            emit(
                "progress",
                message=f"PG [{index}/{len(ids)}] {query_id} run {repetition+1}/3: {result['wall_ms']/1000:.3f}s {result['status']}",
            )
            if result["status"] != "ok":
                break


def run(request, run_id, backend):
    expected = validate(request, backend)
    if not (backend.RUNTIME / "ready.json").is_file():
        raise ValueError("Services are not ready; inspect docker logs")
    dataset = request["dataset"]
    output = backend.RUNTIME / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "pid.txt").write_text(str(os.getpid()))
    report = {
        "version": 1,
        "run_id": run_id,
        "dataset": dataset,
        "database": backend.DATABASES[dataset][0],
        "scope": request["scope"],
        "phase": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pg_repetitions": 3,
        "pg_reported_repetition": 0,
        "neurqo_repetitions": 1,
        "timer": backend.TIMER,
        "statement_timeout_ms": 60000,
        "timeout_charge": "min(360000ms, 5 * pg_first_wall_ms)",
        "no_training": True,
        "no_experience_replay": True,
        "container_output": str(output),
        "pg_runs": {},
        "folds": [],
        "inference_environment": json.loads(
            (backend.RUNTIME / "ready.json").read_text()
        ).get("inference"),
        "sql_sha256": {q: value["sha256"] for q, value in request["queries"].items()},
    }
    for selection in request["folds"]:
        fold = copy.deepcopy(expected[selection["protocol"], selection["fold"]])
        fold.update(query_ids=selection["query_ids"], records=[])
        report["folds"].append(fold)

    def save():
        report["summary"] = aggregate(report, backend)
        backend.write_json(output / "results.json", report)

    code = 0
    try:
        if (backend.RUNTIME / "results" / f"{run_id}.cancel").exists():
            raise KeyboardInterrupt("Cancellation requested before startup completed")
        with ExitStack() as stack:
            for name in ("paper-benchmark.lock", f"{dataset}.lock"):
                lock = stack.enter_context((backend.RUNTIME / "locks" / name).open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            connection = backend.psycopg2.connect(
                **backend.connection_options(report["database"])
            )
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SHOW ALL")
                    report["pg_settings"] = {row[0]: row[1] for row in cursor}
            finally:
                connection.close()
            for key, value in {
                "shared_buffers": "32GB",
                "effective_cache_size": "96GB",
                "work_mem": "128MB",
                "max_parallel_workers_per_gather": "0",
                "jit": "off",
                "geqo": "off",
            }.items():
                if report["pg_settings"][key] != value:
                    raise ValueError(f"Unexpected benchmark setting: {key}")
            report["hash_function_sha256"] = backend.verify_hash_implementation()
            report["action_settings"] = backend.dataset_settings(dataset)
            report["storage_preparation"] = backend.prepare_database(report["database"])
            save()
            measure_pg(request, report, backend, output, save)
            policy_log = backend.RUNTIME / "logs" / f"{dataset}.policy.jsonl"
            for fold in report["folds"]:
                checkpoint = Path(fold["checkpoint"])
                backend.reload_model(dataset, report["database"], checkpoint)
                directory = output / fold["protocol"] / fold["fold"]
                directory.mkdir(parents=True)
                for index, query_id in enumerate(fold["query_ids"], 1):
                    pg = copy.deepcopy(report["pg_runs"][query_id][0])
                    record = {"query_id": query_id, "pg": pg}
                    fold["records"].append(record)
                    if pg["status"] != "ok":
                        record.update(
                            neurqo={"status": "skipped_pg_failure"}, correct=None
                        )
                        save()
                        code = 1
                        continue
                    db_log = directory / f"{query_id}.db.jsonl"
                    offset = policy_log.stat().st_size if policy_log.exists() else 0
                    result = backend.run_once(
                        request["queries"][query_id]["sql"],
                        dataset,
                        report["database"],
                        True,
                        db_log,
                    )
                    record["neurqo"] = result
                    correct, proof = same_result(
                        pg, result, query_id, dataset, backend, output
                    )
                    record.update(correct=correct, **proof)
                    record["trace"] = backend.collect_trace(
                        db_log, policy_log, offset, result, checkpoint
                    )
                    save()
                    emit(
                        "progress",
                        message=f"NeurQO {fold['protocol']}/{fold['fold']} [{index}/{len(fold['query_ids'])}] {query_id}: {result['wall_ms']/1000:.3f}s {result['status']} correct={correct}",
                    )
                    if correct is False or result["status"] == "error":
                        raise ValueError(
                            f"Execution or correctness failure: {query_id}: {result.get('error')}"
                        )
            report["phase"] = "complete"
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        report.update(phase="interrupted" if code == 130 else "failed", error=str(exc))
        emit("error", message=f"Benchmark stopped: {exc}")
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        (output / "pid.txt").unlink(missing_ok=True)
        (backend.RUNTIME / "results" / f"{run_id}.cancel").unlink(missing_ok=True)
        emit("report", report=report)
    return code


def cancel(run_id):
    # Cover cancellation while the worker is still validating its request.
    (RUNTIME / "results" / f"{run_id}.cancel").touch()
    pid_file = RUNTIME / "results" / run_id / "pid.txt"
    if not pid_file.is_file():
        return
    pid = int(pid_file.read_text())
    command = Path(f"/proc/{pid}/cmdline")
    if command.exists():
        argv = command.read_bytes().split(b"\0")
        if run_id.encode() in argv and b"/opt/neurqo/docker_worker.py" in argv:
            os.kill(pid, signal.SIGINT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id")
    group.add_argument("--cancel")
    args = parser.parse_args()
    run_id = args.run_id or args.cancel
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        parser.error("Invalid run ID")
    if args.cancel:
        cancel(run_id)
        return 0
    import paper_benchmark as backend

    try:
        return run(json.load(sys.stdin), run_id, backend)
    except Exception as exc:
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
