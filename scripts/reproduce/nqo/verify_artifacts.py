#!/usr/bin/env python3
"""Audit the retained NQO paper artifacts and their reproduction entry points."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
NQO_RESULTS = REPO / "results" / "benchmark" / "nqo"
PRODUCERS = {
    "nqo_runs.csv": ("run.py",),
    "nqo_independent_action_runs.csv": ("run_independent_actions.py",),
    "nqo_abl_action_run.csv": ("run_abl_action.py",),
    "nqo_abl_rl_run.csv": ("run_abl_rl.py",),
    "nqo_abl_sate_run.csv": ("run_abl_state.py",),
    "nqo_alpha_sensitivity.csv": ("run_alpha_sensitivity.py",),
    "nqo_decomposition_depth.csv": ("run_decomposition_depth.py",),
    "nqo_job_scale_25_runs.csv": ("run_data_scale.sh",),
    "nqo_job_scale_50_runs.csv": ("run_data_scale.sh",),
    "nqo_job_scale_75_runs.csv": ("run_data_scale.sh",),
    "nqo_stack_scale_50_runs.csv": ("run_stack_data_scale.sh",),
    "nqo_learning_trace.csv": ("run_learning_trace.py",),
    "nqo_learning_time.csv": ("export_learning_time.py",),
    "nqo_transfer_run.csv": ("run_transfer.py",),
}
SUMMARY_RESULTS = ("nqo_training_time_comparison.csv",)
SCALE_BUFFERS = {
    "nqo_job_scale_25_runs.csv": "job_scale_25.sql",
    "nqo_job_scale_50_runs.csv": "job_scale_50.sql",
    "nqo_job_scale_75_runs.csv": "job_scale_75.sql",
    "nqo_stack_scale_50_runs.csv": "stack_scale_50.sql",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_cache_ids(path: Path) -> set[str]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return {
            str(row[0])
            for row in connection.execute("SELECT cache_id FROM replay_cache")
        }
    finally:
        connection.close()


def dataset_for(row: dict[str, str]) -> str:
    return str(row.get("dataset") or row.get("target_dataset") or "").upper()


def normalized_checkpoint(value: str) -> Path:
    return REPO / (value[2:] if value.startswith("./") else value)


def verify_learning_inventory(errors: list[str]) -> None:
    trace = {
        (
            row["dataset"].lower(),
            row["protocol"],
            row["fold"],
            int(row["iteration"]),
            row["checkpoint"],
        )
        for row in read_csv(NQO_RESULTS / "nqo_learning_trace.csv")
        if int(row["iteration"]) >= 0
    }
    inventory_path = Path(__file__).with_name("learning_trace_checkpoints.csv")
    inventory = {
        (
            row["workload"],
            row["protocol"],
            row["fold"],
            int(row["iteration"]),
            row["checkpoint"],
        )
        for row in read_csv(inventory_path)
    }
    if trace != inventory:
        errors.append("learning-trace checkpoint inventory differs from released CSV")


def main() -> int:
    errors: list[str] = []
    script_root = Path(__file__).resolve().parent
    rows_by_file: dict[str, list[dict[str, str]]] = {}
    for filename, producers in PRODUCERS.items():
        result_path = NQO_RESULTS / filename
        if not result_path.is_file():
            errors.append(f"missing result: {result_path}")
            continue
        rows = read_csv(result_path)
        rows_by_file[filename] = rows
        if not rows:
            errors.append(f"empty result: {result_path}")
        for producer in producers:
            if not (script_root / producer).is_file():
                errors.append(f"missing producer for {filename}: {producer}")
    for filename in SUMMARY_RESULTS:
        result_path = NQO_RESULTS / filename
        if not result_path.is_file() or not read_csv(result_path):
            errors.append(f"missing or empty summary: {result_path}")

    buffer_ids = {
        dataset: existing_cache_ids(
            REPO / "results" / "buffers" / f"{dataset.lower()}_light.sql"
        )
        for dataset in ("JOB", "STACK", "TPCH")
    }
    for filename, rows in rows_by_file.items():
        scale_buffer = SCALE_BUFFERS.get(filename)
        scale_ids = (
            existing_cache_ids(REPO / "results" / "buffers" / scale_buffer)
            if scale_buffer
            else None
        )
        for row in rows:
            checkpoint = row.get("checkpoint") or ""
            if checkpoint and not normalized_checkpoint(checkpoint).is_file():
                errors.append(f"missing checkpoint referenced by {filename}: {checkpoint}")
            cache_id = row.get("cache_id") or ""
            if not cache_id:
                continue
            if scale_ids is not None:
                if cache_id not in scale_ids:
                    errors.append(f"missing scale cache_id in {filename}: {cache_id}")
                continue
            dataset = dataset_for(row)
            if dataset in buffer_ids and cache_id not in buffer_ids[dataset]:
                errors.append(f"missing {dataset} cache_id in {filename}: {cache_id}")

    verify_learning_inventory(errors)
    if errors:
        print("NQO artifact audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "NQO artifact audit passed: "
        f"{len(PRODUCERS)} reproduced CSVs and {len(SUMMARY_RESULTS)} summary "
        "CSVs, with all checkpoints and cache IDs present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
