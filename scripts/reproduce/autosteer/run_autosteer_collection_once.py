#!/usr/bin/env python3
"""Collect AutoSteer query-hint runtimes once, sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.reproduce.autosteer import run_autosteer_tpch as autosteer
from benchmarking.workloads import load_postgres_times


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("JOB", "STACK", "TPCH"), required=True)
    parser.add_argument("--postgres-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--postgres-time-column", default="run1_charged_s")
    args = parser.parse_args()

    autosteer.WORKLOAD = args.workload
    autosteer.AUTOSTEER_ROOT = (
        ROOT
        / "thrid_party"
        / "genjoin"
        / "OtherMethods_Repository"
        / "autosteer"
    )
    autosteer.KNOB_PATH = autosteer.AUTOSTEER_ROOT / "knobs" / "postgres.txt"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pg_times = load_postgres_times(
        args.postgres_csv.resolve(),
        time_column=args.postgres_time_column,
    )
    autosteer.collect_all(args, output_dir, pg_times)


if __name__ == "__main__":
    main()
