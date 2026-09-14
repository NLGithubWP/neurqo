"""Container-managed policy servers used by benchmark workflows."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from optimization.actions import ActionProfile


class DockerFixedPolicyServer:
    def __init__(
        self,
        *,
        container: str,
        port: int,
        profile: ActionProfile,
        runtime_host_dir: Path,
        runtime_container_dir: str,
        runtime_source_container: str = (
            "/code/pgdb-dev/.nqo_runtime/nqo/src"
        ),
        run_label: Optional[str] = None,
        listen_host: str = "127.0.0.1",
        action_host: str = "127.0.0.1",
        startup_timeout: float = 15.0,
    ) -> None:
        self.container = container
        self.port = port
        self.profile = profile
        self.run_label = run_label or profile.name
        self.listen_host = listen_host
        self.action_host = action_host
        self.startup_timeout = float(startup_timeout)
        if self.startup_timeout <= 0.0:
            raise ValueError("startup_timeout must be positive")
        self.runtime_host_dir = runtime_host_dir
        self.runtime_container_dir = runtime_container_dir
        self.runtime_source_container = runtime_source_container
        self.policy_log_host = runtime_host_dir / f"{self.run_label}.policy.jsonl"
        self.policy_log_container = (
            f"{runtime_container_dir}/{self.run_label}.policy.jsonl"
        )
        self.server_log_host = runtime_host_dir / f"{self.run_label}.server.log"
        self.process: Optional[subprocess.Popen[Any]] = None
        self.log_handle = None

    @property
    def action_url(self) -> str:
        return f"http://{self.action_host}:{self.port}/action"

    def _curl(self, path: str, method: str = "GET") -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "docker",
                "exec",
                self.container,
                "curl",
                "-fsS",
                "-X",
                method,
                f"http://127.0.0.1:{self.port}{path}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def policy_environment(self) -> dict[str, str]:
        return self.profile.policy_environment()

    def model_arguments(self) -> list[str]:
        return [
            "--model-module",
            "runtime.policies.fixed:predict",
            "--policy-version",
            f"fixed:{self.profile.name}",
        ]

    def _kill_owned_container_server(self) -> None:
        subprocess.run(
            [
                "docker",
                "exec",
                self.container,
                "pkill",
                "-f",
                self.policy_log_container,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _cleanup(self) -> None:
        self._curl("/shutdown", method="POST")
        if self.process is not None:
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._kill_owned_container_server()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        if self._curl("/").returncode == 0:
            self._kill_owned_container_server()
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __enter__(self) -> "DockerFixedPolicyServer":
        if self._curl("/").returncode == 0:
            raise RuntimeError(f"policy server port {self.port} is already in use")
        self.runtime_host_dir.mkdir(parents=True, exist_ok=True)
        self.policy_log_host.unlink(missing_ok=True)
        self.log_handle = self.server_log_host.open("ab")
        command = ["docker", "exec"]
        command.extend(["-e", f"PYTHONPATH={self.runtime_source_container}"])
        for key, value in self.policy_environment().items():
            command.extend(["-e", f"{key}={value}"])
        command.extend(
            [
                self.container,
                "python3",
                "-m",
                "runtime.action_server",
                "--host",
                self.listen_host,
                "--port",
                str(self.port),
                *self.model_arguments(),
                "--trajectory-log",
                self.policy_log_container,
                "--require-model",
            ]
        )
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"policy server exited; inspect {self.server_log_host}"
                    )
                if self._curl("/").returncode == 0:
                    return self
                time.sleep(0.1)
            raise TimeoutError(f"policy server did not start on port {self.port}")
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, *_exc: Any) -> None:
        self._cleanup()


class DockerLearnedPolicyServer(DockerFixedPolicyServer):
    def __init__(
        self,
        *,
        model_container_path: str,
        model_method: str,
        model_hidden: int,
        workload: str,
        catalog_container_path: str,
        model_device: str,
        nqo_src: str,
        inference_mode: str,
        temperature: float,
        exploration_epsilon: float,
        coverage_counts_container_path: str | None,
        coverage_mix: float,
        coverage_power: float,
        stochastic_heads: str | None,
        sampling_seed: int,
        policy_version: str,
        action_ablation: str,
        fixed_sched_alpha: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime_source_container=nqo_src, **kwargs)
        self.model_container_path = model_container_path
        self.model_method = model_method
        self.model_hidden = model_hidden
        self.workload = workload
        self.catalog_container_path = catalog_container_path
        self.model_device = model_device
        self.nqo_src = nqo_src
        self.inference_mode = inference_mode
        self.temperature = temperature
        self.exploration_epsilon = exploration_epsilon
        self.coverage_counts_container_path = coverage_counts_container_path
        self.coverage_mix = coverage_mix
        self.coverage_power = coverage_power
        self.stochastic_heads = stochastic_heads
        self.sampling_seed = sampling_seed
        self.learned_policy_version = policy_version
        self.action_ablation = action_ablation
        self.fixed_sched_alpha = fixed_sched_alpha

    def policy_environment(self) -> dict[str, str]:
        return {}

    def model_arguments(self) -> list[str]:
        arguments = [
            "--model-path",
            self.model_container_path,
            "--model-method",
            self.model_method,
            "--model-hidden",
            str(self.model_hidden),
            "--workload",
            self.workload.lower(),
            "--catalog-path",
            self.catalog_container_path,
            "--device",
            self.model_device,
            "--nqo-src",
            self.nqo_src,
            "--inference-mode",
            self.inference_mode,
            "--temperature",
            str(self.temperature),
            "--exploration-epsilon",
            str(self.exploration_epsilon),
            "--coverage-mix",
            str(self.coverage_mix),
            "--coverage-power",
            str(self.coverage_power),
            "--sampling-seed",
            str(self.sampling_seed),
            "--policy-version",
            self.learned_policy_version,
            "--action-ablation",
            self.action_ablation,
        ]
        if self.coverage_counts_container_path is not None:
            arguments.extend(
                ["--coverage-counts-path", self.coverage_counts_container_path]
            )
        if self.stochastic_heads is not None:
            arguments.extend(["--stochastic-heads", self.stochastic_heads])
        if self.fixed_sched_alpha is not None:
            arguments.extend(
                ["--fixed-sched-alpha", str(self.fixed_sched_alpha)]
            )
        return arguments
