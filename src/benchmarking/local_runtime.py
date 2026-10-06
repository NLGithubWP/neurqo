"""Run orchestration inside one container without Docker-in-Docker."""

import os
import sys
from pathlib import Path


def enabled() -> bool:
    return os.environ.get("NEURQO_LOCAL_RUNTIME") == "1"


def scoped_id(value: str, workspace: Path) -> str:
    """Keep run/policy identities distinct across shared-buffer training runs."""
    return f"{workspace.resolve().name}:{value}" if enabled() else value


def command(arguments: list[str]) -> list[str]:
    if not enabled() or arguments[:2] != ["docker", "exec"]:
        return arguments
    index = 2
    environment = []
    while arguments[index].startswith("-"):
        option = arguments[index]
        if option in ("-e", "--env"):
            environment.append(arguments[index + 1])
            index += 2
        elif option in ("-w", "--workdir"):
            # Native subprocesses inherit the workflow's source directory.
            if Path(arguments[index + 1]).resolve() != Path.cwd().resolve():
                raise ValueError(
                    "local runtime requires the workflow working directory"
                )
            index += 2
        elif option == "-i":
            index += 1
        else:
            raise ValueError(f"unsupported local exec option: {option}")
    result = arguments[index + 1 :]
    if result[0] in ("python", "python3"):
        result = [sys.executable, *result[1:]]
    return ["env", *environment, *result] if environment else result
