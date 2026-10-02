"""Opt-in old-checkpoint smoke through the local server and NeurQO kernel."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import psycopg2
import pytest
from psycopg2 import sql

from optimization.actions import ActionProfile

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("NEURQO_TEST_DB") != "1"
    or not os.environ.get("NEURQO_TEST_MODEL")
    or not os.environ.get("NEURQO_TEST_CATALOG"),
    reason="provide NEURQO_TEST_DB, NEURQO_TEST_MODEL and NEURQO_TEST_CATALOG",
)


@pytest.fixture(scope="module")
def model_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("neurqo-model")
    model = Path(os.environ["NEURQO_TEST_MODEL"])
    before = hashlib.sha256(model.read_bytes()).hexdigest()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NEURQO_", "NQO_"))
    }
    env["PYTHONPATH"] = str(ROOT / "src")
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (directory / "server.log").open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "runtime.action_server",
                "--port",
                str(port),
                "--model-path",
                str(model),
                "--catalog-path",
                os.environ["NEURQO_TEST_CATALOG"],
                "--workload",
                "job",
                "--require-model",
                "--trajectory-log",
                str(directory / "policy.jsonl"),
            ],
            env=env,
            cwd="/tmp",
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail((directory / "server.log").read_text())
                try:
                    with urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                        assert "model_source=checkpoint:" in response.read().decode()
                    break
                except URLError:
                    time.sleep(0.05)
            yield f"http://127.0.0.1:{port}/action", directory
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            assert hashlib.sha256(model.read_bytes()).hexdigest() == before


def execute(query, *, enabled, server_url, log_path):
    config = json.loads(
        (ROOT / "scripts/reproduce/neurqo/action_config.json").read_text()
    )
    profile = ActionProfile.from_mapping(
        name="smoke", values=config["datasets"]["JOB"]["base_profile"]
    )
    connection = psycopg2.connect(application_name="neurqo-model-smoke")
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            settings = {
                "neurqo": "off",
                "statement_timeout": 60_000,
                **profile.guc_settings(),
                "neurqo.server_url": server_url,
                "neurqo.trajectory_log": str(log_path),
            }
            for key, value in settings.items():
                value = ("on" if value else "off") if isinstance(value, bool) else value
                cursor.execute(
                    sql.SQL("SET {} TO %s").format(sql.Identifier(key)), (value,)
                )
            if enabled:
                cursor.execute("SET neurqo = on")
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize("query_id", ["2a", "6a", "29a"])
def test_old_model_matches_postgres(query_id, model_server):
    server_url, directory = model_server
    query = (ROOT / "workloads/query_job" / f"{query_id}.sql").read_text()
    log = directory / f"{query_id}.db.jsonl"
    expected = execute(query, enabled=False, server_url=server_url, log_path=log)
    actual = execute(query, enabled=True, server_url=server_url, log_path=log)
    assert actual == expected
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[-1]["status"] == "ok"
    decisions = [
        json.loads(line)
        for line in (directory / "policy.jsonl").read_text().splitlines()
    ]
    assert decisions
    assert all(
        item["action"]["model_source"].startswith("checkpoint:") for item in decisions
    )
