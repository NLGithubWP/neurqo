"""Run the original buffer-replay experiments against the release container."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "run": "run.py",
    "learning": "run_learning_trace.py",
    "transfer": "run_transfer.py",
    "abl-action": "run_abl_action.py",
    "abl-rl": "run_abl_rl.py",
    "abl-state": "run_abl_state.py",
    "alpha": "run_alpha_sensitivity.py",
}


def has_option(arguments: list[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def prepare_arguments(experiment: str, arguments: list[str]) -> list[str]:
    arguments = list(arguments)
    defaults = {"--cache-miss": "error"}
    if experiment in ("run", "alpha"):
        defaults["--model-device"] = "cpu"
    else:
        defaults["--devices"] = "cpu"
    if experiment != "run":
        defaults["--workers"] = "1"
    if experiment == "run" and not has_option(arguments, "--pg-reference"):
        if has_option(arguments, "--database"):
            raise ValueError(
                "--database requires --pg-reference from that database scale"
            )
        defaults["--pg-reference"] = str(ROOT / "results/benchmark/nqo/nqo_runs.csv")
    for key, value in defaults.items():
        if not has_option(arguments, key):
            arguments.extend((key, value))
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="neurqo")
    parser.add_argument("experiment", choices=RUNNERS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    helping = "--help" in args.arguments or "-h" in args.arguments
    if not helping and not has_option(args.arguments, "--output"):
        parser.error(
            "specify --output for the original runner; released CSVs are read-only inputs"
        )
    try:
        arguments = prepare_arguments(args.experiment, args.arguments)
    except ValueError as exc:
        parser.error(str(exc))

    run_id = uuid.uuid4().hex
    os.environ["NEURQO_REPLAY_CONTAINER"] = args.container
    os.environ["NEURQO_REPLAY_ROOT"] = str(ROOT / ".neurqo_runtime" / "replay" / run_id)
    os.environ["NEURQO_REPLAY_REMOTE_ROOT"] = f"/opt/neurqo/runtime/replay/{run_id}"
    directory = ROOT / "scripts/reproduce/neurqo"
    for path in (ROOT, ROOT / "src", directory):
        sys.path.insert(0, str(path))
    script = directory / RUNNERS[args.experiment]
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
