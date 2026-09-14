#!/usr/bin/env python3
"""Reproduce the materialization-footprint and runtime-overhead tables."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "nqo_runs.csv"
INDEPENDENT_ACTION_RUNS = HERE / "nqo_independent_action_runs.csv"
BUFFER_FILES = {
    dataset: REPO / "results/buffers" / f"{dataset.lower()}_light.sql"
    for dataset in ("JOB", "STACK", "TPCH")
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decode_json(encoding: str, payload: bytes) -> object:
    raw = bytes(payload)
    if encoding == "zlib":
        raw = zlib.decompress(raw)
    elif encoding not in {"json", "plain", "raw"}:
        raise ValueError(f"unsupported cache encoding: {encoding}")
    return json.loads(raw.decode("utf-8"))


def print_table(headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> None:
    rows = list(rows)
    widths = [len(value) for value in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index < 2 else value.rjust(widths[index])
            for index, value in enumerate(row)
        ).rstrip()

    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def query_split_materialization(dataset: str) -> dict[str, float]:
    rows = [
        row
        for row in read_csv(INDEPENDENT_ACTION_RUNS)
        if row["dataset"] == dataset
        and row["profile"] == "query_split"
        and row["status"] == "ok"
    ]
    query_ids = {row["query_id"] for row in rows}
    if len(query_ids) != len(rows):
        raise ValueError(f"duplicate QuerySplit measurements for {dataset}")
    materializations = sum(int(row["materializations"] or 0) for row in rows)
    materialized_bytes = sum(int(row["materialized_bytes"] or 0) for row in rows)
    return materialization_summary(len(query_ids), materializations, materialized_bytes)


def selected_nqo_rows(
    runs: list[dict[str, str]], dataset: str
) -> list[dict[str, str]]:
    return [
        row
        for row in runs
        if row["dataset"] == dataset
        and row["method"] == "NQO"
        and row["protocol"] == "random"
    ]


def cached_events(
    dataset: str, rows: list[dict[str, str]]
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    connection = sqlite3.connect(
        f"{BUFFER_FILES[dataset].resolve().as_uri()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        for row in rows:
            cache_id = row["cache_id"]
            record = connection.execute(
                """
                SELECT db_events_encoding, db_events_payload
                FROM replay_cache WHERE cache_id=?
                """,
                (cache_id,),
            ).fetchone()
            if record is None:
                raise KeyError(f"missing {dataset} cache_id {cache_id}")
            events = decode_json(
                str(record["db_events_encoding"]), record["db_events_payload"]
            )
            if not isinstance(events, list):
                raise TypeError(f"cache_id {cache_id} has non-list DB events")
            result[cache_id] = events
    finally:
        connection.close()
    return result


def materialization_summary(
    queries: int, materializations: int, materialized_bytes: int
) -> dict[str, float]:
    mib = materialized_bytes / (1024.0 * 1024.0)
    return {
        "queries": float(queries),
        "materializations": float(materializations),
        "bytes": float(materialized_bytes),
        "mat_per_query": materializations / queries,
        "mib_per_query": mib / queries,
    }


def nqo_materialization(
    rows: list[dict[str, str]], events_by_cache: dict[str, list[dict[str, object]]]
) -> dict[str, float]:
    materializations = 0
    materialized_bytes = 0
    for row in rows:
        for event in events_by_cache[row["cache_id"]]:
            if event.get("phase") != "split":
                continue
            materializations += 1
            materialized = event.get("materialized") or {}
            if not isinstance(materialized, dict):
                raise TypeError("invalid materialized event payload")
            materialized_bytes += int(materialized.get("bytes") or 0)
    return materialization_summary(len(rows), materializations, materialized_bytes)


def runtime_overhead(
    rows: list[dict[str, str]], events_by_cache: dict[str, list[dict[str, object]]]
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    totals["end_to_end"] = sum(float(row["runtime_ms"]) for row in rows)
    totals["inference"] = sum(float(row["inference_ms"] or 0.0) for row in rows)
    for row in rows:
        for event in events_by_cache[row["cache_id"]]:
            timing = event.get("timing_ms") or {}
            if not isinstance(timing, dict):
                raise TypeError("invalid timing event payload")
            for name in (
                "lip_build",
                "aja_build",
                "search",
                "analyze",
                "residual_rewrite",
            ):
                totals[name] += float(timing.get(name) or 0.0)
    totals["decomposition"] = totals["analyze"] + totals["residual_rewrite"]
    totals["measured"] = sum(
        totals[name]
        for name in ("inference", "lip_build", "aja_build", "search", "decomposition")
    )
    return dict(totals)


def percentage(value: float, total: float) -> str:
    return f"{100.0 * value / total:.2f}"


def relative_reduction(baseline: float, value: float) -> str:
    return f"{100.0 * (baseline - value) / baseline:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()

    runs = read_csv(args.runs)
    nqo_rows: dict[str, list[dict[str, str]]] = {}
    nqo_events: dict[str, dict[str, list[dict[str, object]]]] = {}
    for dataset in ("JOB", "STACK", "TPCH"):
        nqo_rows[dataset] = selected_nqo_rows(runs, dataset)
        nqo_events[dataset] = cached_events(dataset, nqo_rows[dataset])

    print("NQO resource and runtime overhead\n")
    print("Inputs")
    print(f"  NQO runs: {args.runs.resolve().relative_to(REPO)}")
    print(
        "  Independent Actions: "
        f"{INDEPENDENT_ACTION_RUNS.relative_to(REPO)}"
    )
    for dataset, path in BUFFER_FILES.items():
        print(f"  NQO {dataset} events: {path.relative_to(REPO)}")
    print()

    print("(a) Materialization footprint (lower is better)\n")
    panel_a = []
    for dataset in ("JOB", "STACK"):
        query_split = query_split_materialization(dataset)
        nqo = nqo_materialization(nqo_rows[dataset], nqo_events[dataset])
        panel_a.extend(
            [
                (
                    dataset,
                    "QuerySplit",
                    f"{query_split['mat_per_query']:.2f}",
                    f"{query_split['mib_per_query']:.2f}",
                    "—",
                ),
                (
                    dataset,
                    "NQO",
                    f"{nqo['mat_per_query']:.2f}",
                    f"{nqo['mib_per_query']:.2f}",
                    relative_reduction(
                        query_split["mib_per_query"], nqo["mib_per_query"]
                    ),
                ),
            ]
        )
    print_table(
        (
            "Workload",
            "Method",
            "Avg materializations/query (#)",
            "Avg materialized data/query (MiB)",
            "Materialized-data reduction (%)",
        ),
        panel_a,
    )
    print("\nTPC-H is omitted because Query Split is not enabled for that workload.\n")

    print("(b) Directly measured NQO-specific runtime overhead (% of end-to-end time)\n")
    panel_b = []
    for dataset in ("JOB", "STACK", "TPCH"):
        overhead = runtime_overhead(nqo_rows[dataset], nqo_events[dataset])
        total = overhead["end_to_end"]
        panel_b.append(
            (
                dataset,
                str(len(nqo_rows[dataset])),
                percentage(overhead["inference"], total),
                percentage(overhead["lip_build"], total),
                percentage(overhead["aja_build"], total),
                percentage(overhead["search"], total),
                percentage(overhead["decomposition"], total),
                percentage(overhead["measured"], total),
            )
        )
    print_table(
        (
            "Workload",
            "#Q",
            "Inference",
            "LIP build",
            "AJA build",
            "Search",
            "Decomp. bookkeeping",
            "Total measured",
        ),
        panel_b,
    )
    print(
        "\nMaterialization execution and regular PostgreSQL planning/execution are "
        "query-processing work and are not counted as NQO-specific overhead."
    )


if __name__ == "__main__":
    main()
