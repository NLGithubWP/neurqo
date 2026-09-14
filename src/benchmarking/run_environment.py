"""Runtime staging and reproducibility contracts for benchmark runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from experience.store import canonical_json


ROOT = Path(__file__).resolve().parents[2]


def command_output(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def portable_path(path: Path) -> str:
    """Represent a host path relative to the repository root."""
    relative = Path(os.path.relpath(path.resolve(), ROOT)).as_posix()
    if relative == "." or relative.startswith(".."):
        return relative
    return f"./{relative}"


def stage_policy_runtime(pgdb_root: Path) -> Path:
    """Mirror the NQO Python modules into the database container mount."""
    source = ROOT / "src"
    runtime_source = pgdb_root / ".nqo_runtime" / "nqo" / "src"
    center_analysis_source = (
        ROOT
        / "scripts"
        / "reproduce"
        / "nqo"
        / "workload_fk_center_analysis.json"
    )
    center_analysis_destination = (
        runtime_source.parent
        / "scripts"
        / "reproduce"
        / "nqo"
        / center_analysis_source.name
    )
    destination = runtime_source
    lock_path = pgdb_root / ".nqo_runtime" / "nqo" / ".sync.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        subprocess.run(
            [
                "rsync",
                "-a",
                "--delete",
                "--delete-excluded",
                "--no-owner",
                "--no-group",
                "--exclude=__pycache__/",
                "--exclude=*.egg-info/",
                f"{source}/",
                f"{destination}/",
            ],
            check=True,
        )
        center_analysis_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(center_analysis_source, center_analysis_destination)
    return runtime_source


def repository_version(path: Path) -> dict[str, Any]:
    head = command_output(["git", "rev-parse", "HEAD"], path)
    status = command_output(["git", "status", "--porcelain"], path)
    if path.resolve() == ROOT.resolve():
        pathspecs = (
            "src/database",
            "src/experience",
            "src/model",
            "src/optimization",
            "src/runtime",
            "src/training",
            "src/benchmarking",
            "scripts/nqo_benchmark.py",
            "scripts/reproduce/nqo/workload_fk_center_analysis.json",
            "workloads/train_test.py",
        )
    else:
        pathspecs = (
            "dbengine/src/backend/parser/query_split.c",
            "dbengine/src/backend/executor/nodeNqoAdaptiveJoin.c",
            "dbengine/src/backend/utils/misc/guc_tables.c",
            "dbengine/src/include/executor/nodeNqoAdaptiveJoin.h",
            "dbengine/src/include/parser/query_split.h",
        )
    files = command_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
        path,
    ).splitlines()
    digest = hashlib.sha256()
    for relative in sorted(files):
        source = path / relative
        if not source.is_file():
            continue
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "path": portable_path(path),
        "head": head,
        "dirty": str(bool(status)),
        "implementation_hash": digest.hexdigest(),
        "implementation_files": files,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_manifest(
    manifest_path: Path,
    resume_contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not manifest_path.is_file():
        episodes_path = manifest_path.parent / "episodes.csv"
        if episodes_path.is_file() and episodes_path.stat().st_size:
            raise RuntimeError(
                f"refusing to resume without a manifest: {episodes_path}"
            )
        return None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_contract = existing.get("resume_contract")
    if isinstance(previous_contract, dict) and canonical_json(
        previous_contract
    ) == canonical_json(resume_contract):
        return existing
    if not isinstance(previous_contract, dict):
        differences = ["resume_contract"]
    else:
        differences = sorted(
            key
            for key in set(previous_contract) | set(resume_contract)
            if canonical_json(previous_contract.get(key))
            != canonical_json(resume_contract.get(key))
        )
    raise RuntimeError(
        "refusing an incompatible experiment resume; use a new "
        f"--experiment-id (changed: {', '.join(differences)})"
    )
