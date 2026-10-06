import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.reproduce import docker_benchmark as client
from scripts.reproduce import docker_worker as worker


def args(dataset="job", protocol="all", fold="all", query=None):
    return argparse.Namespace(
        dataset=dataset, protocol=protocol, fold=fold, query=query
    )


@pytest.mark.parametrize(
    "dataset,folds,unique,occurrences",
    [("job", 9, 113, 341), ("stack", 9, 112, 336), ("tpch", 3, 22, 22)],
)
def test_real_splits_and_sql_payload(dataset, folds, unique, occurrences):
    request = client.build_request(args(dataset))
    assert len(request["folds"]) == folds
    assert len(request["queries"]) == unique
    assert sum(len(f["query_ids"]) for f in request["folds"]) == occurrences
    for query in request["queries"].values():
        assert hashlib.sha256(query["sql"].encode()).hexdigest() == query["sha256"]
    if dataset == "tpch":
        assert [len(f["query_ids"]) for f in request["folds"]] == [7, 7, 8]


@pytest.mark.parametrize("query", ["../29a", "..", "/29a"])
def test_reject_query_path_traversal(query):
    with pytest.raises(ValueError, match="query ID"):
        client.build_request(args(protocol="random", fold="a", query=[query]))


def test_tpch_rejects_unsupported_protocol():
    with pytest.raises(ValueError, match="Unsupported protocol"):
        client.build_request(args("tpch", "base-query"))


def test_one_query_reads_workload_file():
    request = client.build_request(
        args(protocol="base-query", fold="a", query=["27a.sql"])
    )
    assert request["scope"] == "queries"
    assert (
        request["queries"]["27a"]["sql"]
        == (client.ROOT / "workloads/query_job/27a.sql").read_text()
    )


def backend_for_request(tmp_path, request):
    root = tmp_path / "image"
    (root / "source/workloads").mkdir(parents=True)
    (root / "source/workloads/train_test.py").write_bytes(
        (client.ROOT / "workloads/train_test.py").read_bytes()
    )
    directory = root / "workloads" / f"query_{request['dataset']}"
    directory.mkdir(parents=True)
    for query_id, query in request["queries"].items():
        (directory / f"{query_id}.sql").write_text(query["sql"])
    return SimpleNamespace(
        ROOT=root,
        PROTOCOLS=client.PROTOCOLS,
        plan=lambda: [dict(dataset=request["dataset"], **f) for f in request["folds"]],
    )


def test_worker_checks_payload_and_image_compatibility(tmp_path):
    request = client.build_request(args(protocol="base-query", fold="a", query=["29a"]))
    backend = backend_for_request(tmp_path, request)
    assert worker.validate(request, backend)
    request["queries"]["29a"]["sql"] += "\n-- a local edit"
    with pytest.raises(ValueError, match="Local SQL differs"):
        worker.validate(request, backend)


def test_test_fold_membership_is_not_trusted(tmp_path):
    request = client.build_request(args("tpch", "random", "a"))
    backend = backend_for_request(tmp_path, request)
    expected = [dict(dataset=request["dataset"], **f) for f in request["folds"]]
    backend.plan = lambda: expected
    request["folds"] = [
        {**request["folds"][0], "query_ids": request["folds"][0]["query_ids"][:-1]}
    ]
    with pytest.raises(ValueError, match="test query lists differ"):
        worker.validate(request, backend)


def fake_result(ms=1, status="ok"):
    return {
        "status": status,
        "wall_ms": ms,
        "columns": ["n"],
        "result_hash": "hash",
        "result_rows": 1,
    }


def test_pg_order_and_timeout_no_retry(tmp_path):
    calls = []

    def execute(sql, dataset, database, enabled, log):
        assert not enabled
        calls.append(sql)
        return fake_result(status="timeout" if sql == "10" else "ok")

    backend = SimpleNamespace(run_once=execute, compare=lambda pair: True)
    request = {"queries": {q: {"sql": q} for q in ("10", "2")}}
    report = {"dataset": "tpch", "database": "tpch", "pg_runs": {}}
    worker.measure_pg(request, report, backend, tmp_path, lambda: None)
    assert calls == ["2", "2", "2", "10"]
    assert [r["repetition"] for r in report["pg_runs"]["2"]] == [0, 1, 2]
    assert len(report["pg_runs"]["10"]) == 1


def test_aggregate_combines_records_not_fold_ratios():
    counts = []

    def summarize(records, expected):
        counts.append((len(records), expected))
        return {"ws": sum(r["pg"] for r in records) / sum(r["neurqo"] for r in records)}

    report = {
        "scope": "test",
        "folds": [
            {
                "protocol": "random",
                "fold": fold,
                "query_ids": [fold],
                "records": [{"pg": pg, "neurqo": nq}],
            }
            for fold, pg, nq in (("a", 100, 10), ("b", 200, 400), ("c", 100, 100))
        ],
    }
    summary = worker.aggregate(report, SimpleNamespace(summarize=summarize))[0]
    assert counts == [(3, 3)]
    assert summary["ws"] == pytest.approx(400 / 510)
    assert summary["complete_test_protocol"] is True


def test_client_preserves_all_pg_runs_and_timeout_charge(tmp_path):
    report = {
        "dataset": "job",
        "pg_runs": {
            "29a": [
                {**fake_result(ms), "repetition": i} for i, ms in enumerate((100, 2, 1))
            ]
        },
        "folds": [
            {
                "protocol": "random",
                "fold": "a",
                "checkpoint": "best.pt",
                "records": [
                    {
                        "query_id": "29a",
                        "pg": fake_result(100),
                        "neurqo": fake_result(60000, "timeout"),
                        "correct": None,
                    }
                ],
            }
        ],
    }
    client.write_outputs(tmp_path, report)
    assert (
        len(json.loads((tmp_path / "results.json").read_text())["pg_runs"]["29a"]) == 3
    )
    with (tmp_path / "per-query.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["neurqo_charged_ms"]) == 500.0


@pytest.fixture
def mock_client_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "ROOT", tmp_path)
    monkeypatch.setattr(client, "build_request", lambda args: {"dataset": args.dataset})
    report = {"dataset": "job", "summary": [], "folds": [], "pg_runs": {}}
    event = client.EVENT_PREFIX + json.dumps({"type": "report", "report": report})
    monkeypatch.setattr(
        client.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(
            stdin=io.StringIO(), stdout=io.StringIO(event + "\n"), wait=lambda: 0
        ),
    )


def test_client_creates_timestamped_folder_per_run(
    tmp_path, monkeypatch, mock_client_execution
):
    times = iter(
        datetime(2026, 10, 5, 12, 30, 45, microsecond, tzinfo=timezone.utc)
        for microsecond in (123456, 123457)
    )
    monkeypatch.setattr(client, "datetime", SimpleNamespace(now=lambda tz: next(times)))
    for suffix in ("123456", "123457"):
        assert client.main(["--dataset", "job"]) == 0
        output = tmp_path / "results" / f"benchmark_neurqo_20261005_123045_{suffix}"
        assert (output / "request.json").is_file()
        assert (output / "results.json").is_file()
    assert len(list((tmp_path / "results").iterdir())) == 2


def test_client_custom_output_never_overwrites(tmp_path, mock_client_execution):
    output = tmp_path / "custom-results"
    argv = ["--dataset", "job", "--output", str(output)]
    assert client.main(argv) == 0
    previous = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        client.main(argv)
    assert (output / "results.json").read_bytes() == previous


def test_early_cancel_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "RUNTIME", tmp_path)
    (tmp_path / "results").mkdir()
    worker.cancel("a" * 32)
    assert (tmp_path / "results" / ("a" * 32 + ".cancel")).exists()
