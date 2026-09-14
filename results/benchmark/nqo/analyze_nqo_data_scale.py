#!/usr/bin/env python3
"""Summarize JOB data-scale transfer measurements."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    (
        "JOB",
        (
            ("Full", ROOT / "nqo_runs.csv"),
            ("Title 75%", ROOT / "nqo_job_scale_75_runs.csv"),
            ("Title 50%", ROOT / "nqo_job_scale_50_runs.csv"),
            ("Title 25%", ROOT / "nqo_job_scale_25_runs.csv"),
        ),
    ),
    (
        "STACK",
        (
            ("Full", ROOT / "nqo_runs.csv"),
            ("Thread 50%", ROOT / "nqo_stack_scale_50_runs.csv"),
        ),
    ),
)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(
    rows: list[dict[str, str]], *, dataset: str, without_inference: bool
) -> tuple[float, float, int, int, float, float, float, int, int]:
    pg = {
        row["sql_path"]: row
        for row in rows
        if row["dataset"] == dataset and row["method"] == "PostgreSQL"
    }
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["method"] == "NQO"
        and row["protocol"] == "random"
    ]
    pairs = []
    end_to_end_sum = 0.0
    inference_sum = 0.0
    timeouts = 0
    excluded_pg_timeouts = 0
    for row in selected:
        baseline = pg[row["sql_path"]]
        if baseline["status"] != "ok":
            excluded_pg_timeouts += 1
            continue
        runtime = float(row["runtime_ms"])
        end_to_end_sum += runtime
        inference_sum += float(row["inference_ms"] or 0.0)
        if without_inference and row["status"] == "ok":
            runtime -= float(row["inference_ms"] or 0.0)
        pairs.append((float(baseline["runtime_ms"]), runtime))
        timeouts += row["status"] == "timeout"
    if not pairs:
        raise RuntimeError(f"no {dataset} Random NQO rows")
    pg_sum = sum(base for base, _ in pairs)
    nqo_sum = sum(value for _, value in pairs)
    ws = pg_sum / nqo_sum
    gs = math.exp(statistics.fmean(math.log(base / value) for base, value in pairs))
    improved = sum(value < base for base, value in pairs)
    return (
        ws,
        gs,
        improved,
        len(pairs),
        pg_sum,
        nqo_sum,
        100.0 * inference_sum / end_to_end_sum,
        timeouts,
        excluded_pg_timeouts,
    )


def table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(
            value if i == len(row) - 1 else value.ljust(widths[i])
            for i, value in enumerate(row)
        )

    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def main() -> None:
    print(
        "PG baseline timeouts are excluded from paired metrics; "
        "NQO timeouts retain their charged runtime.\n"
        "Inference overhead is sum(inference_ms) / sum(NQO end-to-end runtime_ms) "
        "over the paired queries.\n"
    )
    for dataset_index, (dataset, inputs) in enumerate(EXPERIMENTS):
        if dataset_index:
            print()
        loaded = [(label, load(path)) for label, path in inputs]
        print(dataset)
        modes = (
            (False, "NQO end-to-end"),
            (True, "NQO without inference time"),
        )
        for mode_index, (without_inference, title) in enumerate(modes):
            if mode_index:
                print()
            print(title)
            output = []
            for label, rows in loaded:
                (
                    ws,
                    gs,
                    improved,
                    total,
                    pg_sum,
                    nqo_sum,
                    inference_share,
                    timeouts,
                    excluded_pg_timeouts,
                ) = summarize(
                    rows, dataset=dataset, without_inference=without_inference
                )
                values = (
                    label,
                    f"{ws:.6f}",
                    f"{gs:.6f}",
                    f"{improved}/{total} ({100.0 * improved / total:.2f}%)",
                    f"{pg_sum / 1000.0:.3f}",
                    f"{nqo_sum / 1000.0:.3f}",
                )
                if not without_inference:
                    values += (f"{inference_share:.2f}",)
                output.append(
                    values + (str(timeouts), str(excluded_pg_timeouts))
                )
            headers = ("Data", "WS", "GS", "Imp", "PG (s)", "NQO (s)")
            if not without_inference:
                headers += ("Inf. overhead (%)",)
            table(
                headers + ("NQO timeouts", "PG excluded"), output
            )


if __name__ == "__main__":
    main()
