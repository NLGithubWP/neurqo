"""Opt-in native service/model smoke; no benchmark SQL or training is executed."""

import hashlib
import json
import os
import socket
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from benchmarking.policy_server import (
    DockerFixedPolicyServer,
    DockerLearnedPolicyServer,
)
from optimization.actions import ActionProfile

pytestmark = pytest.mark.skipif(
    os.environ.get("NEURQO_TEST_NATIVE") != "1",
    reason="set NEURQO_TEST_NATIVE=1 with model/catalog paths inside the container",
)


@pytest.mark.parametrize("learned", [False, True])
def test_native_policy_service_and_cleanup(monkeypatch, tmp_path, learned):
    monkeypatch.setenv("NEURQO_LOCAL_RUNTIME", "1")
    monkeypatch.delenv("NEURQO_REPLAY_CONTAINER", raising=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    arguments = dict(
        container="unused",
        port=port,
        profile=ActionProfile(name="native-smoke"),
        runtime_host_dir=tmp_path,
        runtime_container_dir=str(tmp_path),
        startup_timeout=30,
    )
    if learned:
        model = Path(os.environ["NEURQO_TEST_MODEL"])
        before = hashlib.sha256(model.read_bytes()).hexdigest()
        arguments.update(
            model_container_path=str(model),
            model_method="standardmdp_rl",
            model_hidden=128,
            workload="tpch",
            catalog_container_path=os.environ["NEURQO_TEST_CATALOG"],
            model_device="cpu",
            neurqo_src=str(Path(__file__).resolve().parents[1] / "src"),
            inference_mode="deterministic",
            temperature=1.0,
            exploration_epsilon=0.0,
            coverage_counts_container_path=None,
            coverage_mix=0.0,
            coverage_power=0.5,
            stochastic_heads=None,
            sampling_seed=42,
            policy_version="native-smoke",
            action_ablation="none",
        )
        server = DockerLearnedPolicyServer(**arguments)
    else:
        from benchmarking.dataset_training import compatibility_manifest

        analysis = tmp_path / "compatibility.json"
        analysis.write_text(
            json.dumps(
                compatibility_manifest(
                    "JOB50", {"example": "SELECT MIN(r.r_regionkey) FROM region r"}
                )
            )
        )
        monkeypatch.setenv("NEURQO_WORKLOAD_CENTER_ANALYSIS", str(analysis))
        arguments["workload"] = "job50"
        server = DockerFixedPolicyServer(**arguments)
    with server:
        health = server._curl("/")
        assert health.returncode == 0
        if learned:
            assert f"model_source=checkpoint:{model}" in health.stdout
        payload = {
            "request_type": "dec",
            "sql": "SELECT MIN(r.r_regionkey) FROM region r",
            "round": 0,
            "max_split_rounds": 16,
        }
        with urlopen(
            Request(
                server.action_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=10,
        ) as response:
            assert "dec_action=" in response.read().decode()
        assert server.policy_log_host.is_file()
    assert server.process.poll() is not None
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    if learned:
        assert hashlib.sha256(model.read_bytes()).hexdigest() == before
