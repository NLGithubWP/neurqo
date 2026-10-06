import argparse
import csv
import json
import subprocess
from pathlib import Path

import pytest

from benchmarking import replay_runtime as runtime
from benchmarking.execution_cache import read_jsonl_since
from benchmarking.policy_server import DockerFixedPolicyServer
from benchmarking.run_environment import stage_policy_runtime
from experience.store import canonical_trajectory
from optimization.actions import ActionProfile, semantic_policy_trajectory
from scripts.reproduce import docker_replay
from scripts.reproduce.neurqo import run


@pytest.mark.parametrize("phase", ["dec", "sched", "enum", "adapt"])
def test_index_only_actions_remain_distinct(phase):
    for index in (0, 1):
        records = canonical_trajectory(
            [{"phase": phase, "state": {}, "action": {"action_index": index}}]
        )
        assert records[0]["action"] == {"action_index": index}


@pytest.fixture
def release(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURQO_REPLAY_CONTAINER", "release-test")
    monkeypatch.setenv("NEURQO_REPLAY_ROOT", str(tmp_path))
    monkeypatch.setenv("NEURQO_REPLAY_REMOTE_ROOT", "/opt/neurqo/runtime/replay/test")
    return tmp_path


def test_release_paths_do_not_require_a_mount(release):
    assert runtime.container_path(release / "model.pt", release) == (
        "/opt/neurqo/runtime/replay/test/model.pt"
    )
    with pytest.raises(ValueError):
        runtime.container_path(release.parent / "outside", release)
    assert stage_policy_runtime(release) == Path("/opt/neurqo/source/src")
    assert not (release / ".neurqo_runtime").exists()


def test_development_paths_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("NEURQO_REPLAY_CONTAINER", raising=False)
    assert runtime.container_path(tmp_path / "a", tmp_path) == "/code/pgdb-dev/a"
    assert runtime.runtime_python() == "python3"


def test_release_refuses_fresh_collection(release):
    with pytest.raises(ValueError, match="no SQL collection"):
        runtime.configure_args(argparse.Namespace(cache_miss="execute"))


def test_release_connections(release):
    args = argparse.Namespace(
        cache_miss="error",
        container="old",
        user="pgdb",
        pg_port=15432,
        server_action_host=None,
    )
    runtime.configure_args(args)
    assert (args.container, args.user, args.pg_port) == ("release-test", "neurdb", 5432)
    assert args.server_action_host == "127.0.0.1"


def test_remote_log_reads_offsets_without_a_shared_file(release, monkeypatch):
    remote = release / "policy.jsonl"
    payload = '{"state":"example"}\n'
    calls = []

    def fake_remote(code, path, offset):
        calls.append((path, offset))
        return json.dumps({"data": payload, "offset": 17 + len(payload)})

    monkeypatch.setattr(runtime, "remote_python", fake_remote)
    events, offset = read_jsonl_since(remote, 17)
    assert not remote.exists()
    assert events == [{"state": "example"}]
    assert offset == 17 + len(payload)
    assert calls == [("/opt/neurqo/runtime/replay/test/policy.jsonl", "17")]


def test_remote_log_byte_offsets_and_partial_lines(release, monkeypatch):
    path = release / "log"
    raw = '{"value":"\u00e9"}\n'.encode()
    path.write_bytes(raw + b'{"partial":')

    def execute_code(code, remote, offset):
        return subprocess.check_output(
            ["python3", "-c", code, str(path), offset], text=True
        )

    monkeypatch.setattr(runtime, "remote_python", execute_code)
    events, offset = runtime.read_remote_jsonl(path, 0)
    assert events == [{"value": "\u00e9"}]
    assert offset == len(raw)
    assert runtime.read_remote_jsonl(path, offset) == ([], offset)


def test_release_health_check_does_not_require_curl(release, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append(command))
    server = DockerFixedPolicyServer(
        container="release-test",
        port=19000,
        profile=ActionProfile(name="test"),
        runtime_host_dir=release,
        runtime_container_dir="/opt/neurqo/runtime/test",
    )
    server._curl("/")
    assert calls[0][3:5] == ["python", "-c"]
    assert "curl" not in calls[0]


def test_release_does_not_accept_another_checkpoint_service(release):
    server = DockerFixedPolicyServer(
        container="release-test",
        port=19000,
        profile=ActionProfile(name="test"),
        runtime_host_dir=release,
        runtime_container_dir="/opt/neurqo/runtime/test",
    )
    server.model_container_path = "/opt/neurqo/runtime/test/model.pt"
    other = subprocess.CompletedProcess(
        [], 0, stdout="model_source=checkpoint:/other.pt\n"
    )
    own = subprocess.CompletedProcess(
        [], 0, stdout=f"model_source=checkpoint:{server.model_container_path}\n"
    )
    assert not server._is_owned_health(other)
    assert server._is_owned_health(own)


def test_launcher_preserves_explicit_experiment_options():
    args = docker_replay.prepare_arguments(
        "learning", ["--workers=2", "--devices", "cpu"]
    )
    assert args.count("--devices") == 1
    assert "--workers" not in args
    assert args[-2:] == ["--cache-miss", "error"]


def test_scale_requires_matching_pg_reference():
    with pytest.raises(ValueError, match="database scale"):
        docker_replay.prepare_arguments("run", ["--database", "so_scale_50"])


def test_launcher_requires_separate_output():
    with pytest.raises(SystemExit) as result:
        docker_replay.main(["run", "--dataset", "job"])
    assert result.value.code == 2


def test_reference_seeding_preserves_times_and_detects_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "DEFAULT_RUNTIME_DIR", tmp_path / "runtime")
    source, output = tmp_path / "reference.csv", tmp_path / "new.csv"
    row = {field: "" for field in run.CSV_FIELDS}
    row.update(
        dataset="JOB",
        method="PostgreSQL",
        sql_path="workloads/query_job/1a.sql",
        runtime_ms="123.450000",
        status="ok",
    )
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    original = source.read_bytes()
    run.seed_postgres_reference(source, output, "JOB")
    run.seed_postgres_reference(source, output, "JOB")
    assert len(run.ResultCsv(output).rows) == 1
    assert next(iter(run.ResultCsv(output).rows.values()))["runtime_ms"] == "123.450000"
    assert source.read_bytes() == original
    output.write_text(output.read_text().replace("123.450000", "456.000000"))
    with pytest.raises(ValueError, match="conflicting PostgreSQL"):
        run.seed_postgres_reference(source, output, "JOB")


def test_reference_cannot_be_output(tmp_path):
    with pytest.raises(ValueError, match="different files"):
        run.seed_postgres_reference(tmp_path / "x.csv", tmp_path / "x.csv", "JOB")


def test_legacy_low_action_matches_live_semantics_without_mutating_buffer():
    stored = {
        "phase": "low",
        "state": {"request_type": "low"},
        "action": {"execution_action": "conservative", "lip_action": "selective"},
    }
    before = json.dumps(stored, sort_keys=True)
    cached = canonical_trajectory([stored])[0]
    live = semantic_policy_trajectory(
        [
            {
                "phase": "policy_decision",
                "state": {"request_type": "adapt"},
                "action": {
                    "ajoin_action": "conservative",
                    "filter_action": "selective",
                    "adapt_action": "filter_selective+ajoin_conservative",
                },
            }
        ]
    )[0]
    assert cached["action"] == live["action"]
    assert json.dumps(stored, sort_keys=True) == before


def test_standalone_profile_accepts_released_pre_vocabulary_rename_hash():
    assert "36b15fae0f83c801a7b6aeee3fed3b26620fd8d4693dec7a68c172f33cff8949" in (
        run.profile_for("JOB", "query_split").compatible_hashes()
    )
