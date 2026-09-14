"""Unified command-line interface for NQO benchmark workflows."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from benchmarking import (
    action_runner,
    action_tuning,
    iterative_training,
    matrix,
)


Command = tuple[Callable[[list[str] | None], int], str]

COMMANDS: dict[str, Command] = {
    "run": (
        action_runner.main,
        "execute PostgreSQL, fixed-action, or learned-policy benchmarks",
    ),
    "tune": (
        action_tuning.main,
        "calibrate and freeze dataset-level action parameters",
    ),
    "train": (
        iterative_training.main,
        "alternate experience collection, policy training, and evaluation",
    ),
    "matrix": (
        matrix.main,
        "run and aggregate workload, protocol, and fold combinations",
    ),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {description}"
        for name, (_entrypoint, description) in COMMANDS.items()
    )
    return (
        "usage: nqo-benchmark <command> [options]\n\n"
        "commands:\n"
        f"{commands}\n\n"
        "Run 'nqo-benchmark <command> --help' for command-specific options."
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage())
        return 0

    name = arguments.pop(0)
    command = COMMANDS.get(name)
    if command is None:
        print(f"nqo-benchmark: unknown command: {name}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    entrypoint, _description = command
    return entrypoint(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
