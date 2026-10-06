"""Container-managed policy servers used by benchmark workflows."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from benchmarking.local_runtime import command as local_command
from benchmarking.local_runtime import enabled as local_runtime
from benchmarking.replay_runtime import (
    release_container,
    runtime_python,
    stage_directory,
)
from optimization.actions import ActionProfile


class DockerFixedPolicyServer:
    def __init__(
        self,
        *,
        container: str,
        port: int,
        profile: ActionProfile,
        workload: str = "job",
        runtime_host_dir: Path,
        runtime_container_dir: str,
        runtime_source_container: str = ("/code/pgdb-dev/.neurqo_runtime/neurqo/src"),
        run_label: Optional[str] = None,
        listen_host: str = "127.0.0.1",
        action_host: str = "127.0.0.1",
        startup_timeout: float = 15.0,
    ) -> None:
        self.container = container
        self.port = port
        self.profile = profile
        self.workload = workload
        self.run_label = run_label or profile.name
        self.listen_host = listen_host
        self.action_host = action_host
        self.startup_timeout = float(startup_timeout)
        if self.startup_timeout <= 0.0:
            raise ValueError("startup_timeout must be positive")
        self.runtime_host_dir = runtime_host_dir
        self.runtime_container_dir = runtime_container_dir
        self.runtime_source_container = runtime_source_container
        if local_runtime():
            self.runtime_source_container = str(Path(__file__).resolve().parents[1])
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
        if release_container() or local_runtime():
            return subprocess.run(
                local_command(
                    [
                        "docker",
                        "exec",
                        self.container,
                        runtime_python(),
                        "-c",
                        "import sys,urllib.request;"
                        "r=urllib.request.Request(sys.argv[1],method=sys.argv[2]);"
                        "sys.stdout.buffer.write(urllib.request.urlopen(r,timeout=5).read())",
                        f"http://127.0.0.1:{self.port}{path}",
                        method,
                    ]
                ),
                capture_output=True,
                text=True,
                timeout=15,
            )
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

    def _is_owned_health(self, result: subprocess.CompletedProcess) -> bool:
        if result.returncode != 0:
            return False
        if (release_container() or local_runtime()) and hasattr(
            self, "model_container_path"
        ):
            return (
                f"model_source=checkpoint:{self.model_container_path}\n"
                in result.stdout
            )
        return True

    def model_arguments(self) -> list[str]:
        return [
            "--model-module",
            "runtime.policies.fixed:predict",
            "--policy-version",
            f"fixed:{self.profile.name}",
            "--workload",
            self.workload.lower(),
        ]

    def _kill_owned_container_server(self) -> None:
        if local_runtime():
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return
        if release_container():
            subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container,
                    runtime_python(),
                    "-c",
                    "import os,signal,sys;from pathlib import Path;"
                    "marker=sys.argv[1].encode();"
                    "[(os.kill(int(p.name),signal.SIGTERM)) for p in Path('/proc').iterdir() "
                    "if p.name.isdigit() and int(p.name)!=os.getpid() "
                    "and (p/'cmdline').exists() "
                    "and marker in (p/'cmdline').read_bytes().split(b'\\0') "
                    "and b'runtime.action_server' in (p/'cmdline').read_bytes().split(b'\\0')]",
                    self.policy_log_container,
                ],
                capture_output=True,
                timeout=15,
                check=False,
            )
            return
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
        if not (release_container() or local_runtime()) or self._is_owned_health(
            self._curl("/")
        ):
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
        if release_container():
            stage_directory(
                self.runtime_host_dir,
                self.runtime_container_dir,
                self.policy_log_container,
            )
        self.log_handle = self.server_log_host.open("ab")
        command = ["docker", "exec"]
        command.extend(["-e", f"PYTHONPATH={self.runtime_source_container}"])
        for key, value in self.policy_environment().items():
            command.extend(["-e", f"{key}={value}"])
        command.extend(
            [
                self.container,
                runtime_python(),
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
                local_command(command),
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"policy server exited; inspect {self.server_log_host}"
                    )
                if self._is_owned_health(self._curl("/")):
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
        neurqo_src: str,
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
        super().__init__(
            runtime_source_container=neurqo_src, workload=workload, **kwargs
        )
        self.model_container_path = model_container_path
        self.model_method = model_method
        self.model_hidden = model_hidden
        self.workload = workload
        self.catalog_container_path = catalog_container_path
        self.model_device = model_device
        self.neurqo_src = neurqo_src
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
            "--neurqo-src",
            self.neurqo_src,
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
            arguments.extend(["--fixed-sched-alpha", str(self.fixed_sched_alpha)])
        return arguments
