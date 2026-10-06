import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from benchmarking import dataset_training as workflow
from benchmarking import local_runtime, run_environment
from scripts.reproduce import docker_benchmark, docker_train


@pytest.fixture
def workflow_request(tmp_path):
    return {
        "dataset": "example",
        "workspace": str(tmp_path),
        "operation": "train",
        "experience_db": str(tmp_path / "experience/example.sqlite"),
        "ai_port": 18140,
        "replay_group": "example-fixed",
        "pretrain_epochs": 16,
        "iterations": 32,
        "eval_every": 4,
        "action_config": None,
        "protocol": "random",
        "fold": "a",
    }


def option(command, name):
    return command[command.index(name) + 1]


def test_native_transport_is_opt_in(monkeypatch):
    monkeypatch.delenv("NEURQO_LOCAL_RUNTIME", raising=False)
    command = ["docker", "exec", "old", "python3", "-m", "runtime.action_server"]
    assert local_runtime.command(command) == command
    monkeypatch.setenv("NEURQO_LOCAL_RUNTIME", "1")
    assert local_runtime.command(command) == [
        sys.executable,
        "-m",
        "runtime.action_server",
    ]
    assert local_runtime.command(
        [
            "docker",
            "exec",
            "-e",
            "A=B",
            "-w",
            os.getcwd(),
            "local",
            "python",
            "-c",
            "pass",
        ]
    ) == ["env", "A=B", sys.executable, "-c", "pass"]
    with pytest.raises(ValueError, match="working directory"):
        local_runtime.command(
            ["docker", "exec", "-w", "/not-this-directory", "local", "python"]
        )


def test_native_shared_buffer_uses_unique_policy_and_episode_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURQO_LOCAL_RUNTIME", "1")
    first = local_runtime.scoped_id("random-a-policy-0000", tmp_path / "run-1")
    second = local_runtime.scoped_id("random-a-policy-0000", tmp_path / "run-2")
    assert first != second
    assert local_runtime.scoped_id("random-a-policy-0000", tmp_path / "run-1") == first


def test_native_staging_does_not_use_rsync_or_git(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURQO_LOCAL_RUNTIME", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("native staging must not invoke development tools")

    monkeypatch.setattr(run_environment, "command_output", forbidden)
    assert (
        run_environment.stage_policy_runtime(tmp_path) == run_environment.ROOT / "src"
    )
    (tmp_path / "src").mkdir()
    code = tmp_path / "src/example.py"
    code.write_text("x = 1\n")
    before = run_environment.repository_version(tmp_path)
    code.write_text("x = 2\n")
    assert (
        before["implementation_hash"]
        != run_environment.repository_version(tmp_path)["implementation_hash"]
    )


def test_training_plan_uses_baseline_pretraining_and_fixed_iterations(workflow_request):
    request = workflow_request
    spec = {"protocol": "random", "fold": "a", "train": ["1", "2"], "test": ["3"]}
    baseline = Path(request["workspace"]) / "postgres/baseline.json"
    command = workflow.training_command(request, spec, baseline)
    assert option(command, "--baseline-json") == str(baseline)
    assert option(command, "--baseline-first-episodes") == str(
        baseline.parent / "episodes.csv"
    )
    assert option(command, "--fixed-replay-group") == "example-fixed"
    assert option(command, "--bootstrap-replay-epochs") == "16"
    assert option(command, "--replay-epochs") == "4"
    assert option(command, "--iterations") == "32"
    assert option(command, "--eval-every") == "4"
    assert option(command, "--formal-execution-cache") == "read-write"
    assert option(command, "--trainer-device") == "cpu"
    assert "--fixed-sched-alpha" not in command
    assert "--validation-ratio" not in command


def test_collection_is_one_execution_and_shared_buffer(workflow_request):
    request = workflow_request
    command = workflow.collector_command(
        request, experiment="fixed", profiles="top5", baseline=Path("baseline.json")
    )
    assert option(command, "--warmups") == "0"
    assert option(command, "--measurements") == "1"
    assert option(command, "--experience-db") == request["experience_db"]
    assert option(command, "--port") == "5432"
    assert option(command, "--user") == "neurdb"


def test_fixed_service_uses_the_registered_workload(tmp_path):
    from benchmarking.policy_server import DockerFixedPolicyServer
    from optimization.actions import ActionProfile

    server = DockerFixedPolicyServer(
        container="unused",
        port=18140,
        profile=ActionProfile(name="top5"),
        workload="JOB50",
        runtime_host_dir=tmp_path,
        runtime_container_dir=str(tmp_path),
    )
    assert option(server.model_arguments(), "--workload") == "job50"


def test_new_workload_compatibility_preserves_spj_aggregates(tmp_path):
    from optimization.decomposition_eligibility import decomposition_eligibility

    payload = workflow.compatibility_manifest(
        "EXAMPLE", {"1": "SELECT MIN(t.title) FROM title t;"}
    )
    path = tmp_path / "compatibility.json"
    path.write_text(json.dumps(payload))
    assert decomposition_eligibility("example", analysis_path=path).enabled


def test_fold_aggregation_sums_runtimes_not_speedups():
    baseline = {"q": {"median_charged_ms": 100}, "r": {"median_charged_ms": 2}}
    folds = [
        {
            "protocol": "random",
            "fold": "a",
            "queries": {"q": {"median_charged_ms": 50}},
        },
        {"protocol": "random", "fold": "b", "queries": {"r": {"median_charged_ms": 4}}},
        {
            "protocol": "random",
            "fold": "c",
            "queries": {"q": {"median_charged_ms": 25}},
        },
    ]
    summary = workflow.aggregate_folds(folds, baseline)[0]
    assert summary["ws"] == pytest.approx(202 / 79)
    assert summary["query_count"] == 3


def test_fold_selection_accepts_registered_dataset(monkeypatch):
    monkeypatch.setitem(workflow.SPLIT_PROTOCOLS, "EXAMPLE", ("random",))
    monkeypatch.setattr(
        workflow,
        "split_folds",
        lambda *args: {
            f"random_{fold}": {"train": ["train"], "test": [f"test_{fold}"]}
            for fold in "abc"
        },
    )
    assert len(workflow.selected_folds("example", "all", "all")) == 3
    with pytest.raises(ValueError, match="unsupported protocol"):
        workflow.selected_folds("example", "base-query", "a")


def test_training_workflow_orders_collection_then_training(
    monkeypatch, workflow_request, tmp_path
):
    request = workflow_request
    spec = {"protocol": "random", "fold": "a", "train": ["1"], "test": ["2"]}
    monkeypatch.setattr(workflow, "selected_folds", lambda *args: [spec])
    monkeypatch.setattr(workflow, "workload_query_ids", lambda *args: ["1", "2"])
    monkeypatch.setattr(
        workflow, "query_sql", lambda *args: "SELECT MIN(t.id) FROM title t"
    )
    monkeypatch.setitem(workflow.WORKLOAD_DATABASES, "EXAMPLE", "example_db")

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            assert "CREATE EXTENSION" in sql

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    monkeypatch.setattr(workflow, "connect", lambda *args, **kwargs: Connection())
    monkeypatch.setattr(workflow, "create_catalog_snapshot", lambda *args: None)
    calls = []

    def run(command):
        calls.append(command)
        experiment = option(command, "--experiment-id")
        report = tmp_path / "reports/example" / experiment
        report.mkdir(parents=True)
        if experiment == "actions":
            (report / "summary.json").write_text(
                json.dumps({"query_split": {"metrics": {"valid": True}}})
            )
        elif experiment == "random-a":
            checkpoint = tmp_path / "selected.pt"
            checkpoint.write_bytes(b"test checkpoint")
            (report / "training_state.json").write_text(
                json.dumps({"formal_best_checkpoint": str(checkpoint)})
            )

    monkeypatch.setattr(workflow, "run_command", run)
    workflow.collect_and_train(request)
    assert [option(command, "--experiment-id") for command in calls] == [
        "postgres",
        "actions",
        "random-a",
    ]
    assert option(calls[0], "--profiles") == "pg"
    assert set(option(calls[1], "--profiles").split(",")) == {
        "neurqo_none",
        "query_split",
        "top5",
        "lip_selective",
        "aja_conservative",
    }
    saved = json.loads((tmp_path / "deployment.json").read_text())
    assert Path(saved["models"]["random/a"]).read_bytes() == b"test checkpoint"
    assert saved["database"] == "example_db"


def test_benchmark_dispatch_preserves_released_path(monkeypatch):
    called = []
    monkeypatch.setattr(
        docker_train, "benchmark", lambda args: called.append(args) or 0
    )
    assert (
        docker_benchmark.main(
            [
                "--dataset",
                "example",
                "--training-run",
                "trained",
                "--protocol",
                "all",
                "--fold",
                "all",
            ]
        )
        == 0
    )
    assert called[0].training_run == Path("trained")
    with pytest.raises(SystemExit):
        docker_benchmark.main(["--dataset", "example"])


def test_dry_run_does_not_contact_docker(monkeypatch, capsys):
    monkeypatch.setattr(docker_train, "validate_dataset", lambda *args: None)
    monkeypatch.setattr(
        docker_train,
        "stage_source",
        lambda *args: pytest.fail("unexpected Docker call"),
    )
    assert docker_train.main(["--dataset", "example", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experience_db"].endswith("/example.sqlite")
    assert payload["pretrain_epochs"] == 16 and payload["iterations"] == 32


def test_invalid_correctness_prevents_training():
    with pytest.raises(RuntimeError, match="invalid results"):
        workflow.ensure_valid({"top5": {"metrics": {"valid": False}}})


def test_cancel_does_not_signal_another_run(monkeypatch, tmp_path):
    (tmp_path / "active.json").write_text(
        json.dumps({"pid": os.getpid(), "run_id": "another"})
    )
    monkeypatch.setattr(
        os, "killpg", lambda *args: pytest.fail("unrelated process signalled")
    )
    workflow.cancel(tmp_path, "ours")


def test_iterative_driver_pretrains_updates_and_resumes_without_pg_rerun(
    monkeypatch, tmp_path
):
    from benchmarking import iterative_training as trainer

    monkeypatch.setattr(trainer, "TRAINING_STOP_FILE", tmp_path / "absent-stop")
    monkeypatch.setattr(trainer, "sync_runtime", lambda *args: None)
    monkeypatch.setattr(trainer, "workload_supports_decomposition", lambda *args: True)
    monkeypatch.setattr(
        trainer,
        "split_folds",
        lambda *args: {"random_a": {"train": ["1a"], "test": ["2a"]}},
    )
    monkeypatch.setattr(trainer, "workload_query_ids", lambda *args: ["1a", "2a"])
    monkeypatch.setattr(
        trainer.psycopg2,
        "connect",
        lambda **kwargs: argparse.Namespace(close=lambda: None),
    )
    monkeypatch.setattr(trainer, "read_postgres_catalog", lambda *args, **kwargs: {})

    def catalog(path, value):
        path.write_text("{}")
        return "catalog-hash"

    monkeypatch.setattr(trainer, "write_catalog_snapshot", catalog)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({qid: {"median_charged_ms": 100.0} for qid in ["1a", "2a"]})
    )
    (tmp_path / "episodes.csv").write_text(
        "profile,repetition,query_id,charged_wall_ms\npg,0,1a,100\npg,0,2a,100\n"
    )
    experience = tmp_path / "example.sqlite"
    experience.touch()
    calls = []

    def fake_run(command):
        calls.append(command)
        if "training.experience_trainer" in command:
            remote_output = Path(option(command, "--output"))
            output = tmp_path / remote_output.relative_to("/code/pgdb-dev")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mock checkpoint")

    monkeypatch.setattr(trainer, "run_command", fake_run)

    def evaluate(args, **kwargs):
        return {
            "iteration": kwargs["iteration"],
            "checkpoint": str(kwargs["checkpoint"]),
            "policy_version": kwargs["policy_version"],
            "runtime_semantics": trainer.FIRST_RUNTIME_SEMANTICS,
            "test": {
                "workload_speedup": 1.0 + kwargs["iteration"],
                "runtime_semantics": trainer.FIRST_RUNTIME_SEMANTICS,
            },
        }

    monkeypatch.setattr(trainer, "evaluate_checkpoint", evaluate)
    arguments = [
        "--workload",
        "JOB",
        "--protocol",
        "random",
        "--fold",
        "a",
        "--experiment-id",
        "smoke",
        "--pgdb-root",
        str(tmp_path),
        "--output-root",
        str(tmp_path / "reports"),
        "--baseline-json",
        str(baseline),
        "--experience-db",
        str(experience),
        "--iterations",
        "1",
        "--eval-every",
        "1",
        "--fixed-replay-group",
        "fixed",
        "--replay-epochs",
        "4",
        "--bootstrap-replay-epochs",
        "16",
        "--formal-execution-cache",
        "read-write",
    ]
    assert trainer.main(arguments) == 0
    updates = [command for command in calls if "training.experience_trainer" in command]
    assert len(updates) == 2
    assert (
        "--replay-only" in updates[0] and option(updates[0], "--replay-epochs") == "16"
    )
    assert (
        "--source-policy-version" in updates[1]
        and option(updates[1], "--replay-epochs") == "4"
    )
    assert all(option(command, "--replay-query-id") == "1a" for command in updates)
    assert all("2a" not in command for command in updates)
    assert all(
        option(command, "--profiles") != "pg"
        for command in calls
        if "--profiles" in command
    )
    calls.clear()
    assert trainer.main(arguments) == 0
    assert not calls
