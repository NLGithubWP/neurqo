"""Mount-free transport for replay against the released Docker image."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def release_container() -> str | None:
    return os.environ.get("NEURQO_REPLAY_CONTAINER") or None


def container_path(path: Path, legacy_root: Path) -> str:
    relative = path.resolve().relative_to(legacy_root.resolve())
    if release_container():
        root = os.environ["NEURQO_REPLAY_REMOTE_ROOT"]
    else:
        root = "/code/pgdb-dev"
    return f"{root}/{relative.as_posix()}"


def policy_source() -> str:
    if release_container():
        return "/opt/neurqo/source/src"
    return "/code/pgdb-dev/.neurqo_runtime/neurqo/src"


def runtime_python() -> str:
    return "python" if release_container() else "python3"


def remote_python(code: str, *arguments: str) -> str:
    container = release_container()
    if not container:
        raise RuntimeError("release replay is not configured")
    return subprocess.run(
        [
            "docker",
            "exec",
            "-e",
            f"PYTHONPATH={policy_source()}",
            container,
            runtime_python(),
            "-c",
            code,
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


def read_catalog(database: str) -> dict[str, Any]:
    # Only metadata is read; benchmark SQL is never executed by this adapter.
    return json.loads(
        remote_python(
            "import json,sys,psycopg2;"
            "from database.catalog import read_postgres_catalog;"
            "c=psycopg2.connect(host='127.0.0.1',port=5432,user='neurdb',dbname=sys.argv[1]);"
            "c.set_session(readonly=True);"
            "print(json.dumps(read_postgres_catalog(c,schema='public')));c.close()",
            database,
        )
    )


def stage_directory(local: Path, remote: str, policy_log: str) -> None:
    remote_python(
        "from pathlib import Path;import sys;"
        "Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)",
        remote,
    )
    subprocess.run(
        ["docker", "cp", f"{local}/.", f"{release_container()}:{remote}/"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_python(
        "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(b'')",
        policy_log,
    )


def read_remote_jsonl(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    remote = container_path(path, Path(os.environ["NEURQO_REPLAY_ROOT"]))
    payload = json.loads(
        remote_python(
            "import json,sys;from pathlib import Path;"
            "p=Path(sys.argv[1]);offset=int(sys.argv[2]);"
            "f=p.open('rb');size=f.seek(0,2);"
            "assert 0<=offset<=size,'policy log was truncated';"
            "f.seek(offset);data=f.read();f.close();"
            "data=data[:data.rfind(b'\\n')+1];"
            "print(json.dumps({'data':data.decode('utf-8'),'offset':offset+len(data)}))",
            remote,
            str(offset),
        )
    )
    records = [
        json.loads(line) for line in payload["data"].splitlines() if line.strip()
    ]
    return [record for record in records if isinstance(record, dict)], payload["offset"]


def configure_args(args: Any) -> None:
    """Select the release connection without changing policy or cache matching."""
    container = release_container()
    if not container:
        return
    if getattr(args, "cache_miss", "error") != "error":
        raise ValueError(
            "release replay requires --cache-miss error; no SQL collection"
        )
    args.container = container
    for key, value in (
        ("database_container", container),
        ("host", "127.0.0.1"),
        ("pg_port", 5432),
        ("user", "neurdb"),
        ("server_action_host", "127.0.0.1"),
        ("action_host", "127.0.0.1"),
    ):
        if hasattr(args, key):
            setattr(args, key, value)
