#!/usr/bin/env python3
"""Recompute corrected-fold TPC-H metrics with the July 26 collection PG."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
CORRECTED_DIR = ROOT / "results" / "revision" / "tpch" / "corrected_splits"
AUTOSTEER_COLLECTION = (
    ROOT / "results" / "revision" / "tpch" / "autosteer" / "collection.csv"
)
OUTPUT_TAG = "pg_20260726_collection_default_normalized"

sys.path.insert(0, str(SCRIPT_DIR))

from scripts.reproduce.common import summarize_tpch_corrected_splits as base  # noqa: E402


def collection_postgres_times() -> dict[str, float]:
    default_rows = [
        row
        for row in base.read_csv(AUTOSTEER_COLLECTION)
        if row["config"] == "None"
    ]
    expected = {str(value) for value in range(1, 23)}
    query_ids = [row["query_id"] for row in default_rows]
    if len(query_ids) != 22 or set(query_ids) != expected:
        raise RuntimeError("July 26 collection PG does not cover Q1--Q22 exactly once")
    if any(row["error"] for row in default_rows):
        raise RuntimeError("July 26 collection PG contains failed executions")
    return {row["query_id"]: float(row["charged_runtime_s"]) for row in default_rows}


def default_query_ids(path: Path, column: str, value: str) -> set[str]:
    return {
        row["query_id"]
        for row in base.read_csv(path)
        if row[column] == value
    }


def corrected_records() -> list[dict]:
    rows = base.per_query_records()
    pg = collection_postgres_times()
    fastgres_defaults = default_query_ids(
        CORRECTED_DIR / "fastgres" / "resolved_predictions.csv",
        "predicted_hint",
        "63",
    )
    autosteer_defaults = default_query_ids(
        CORRECTED_DIR / "autosteer" / "selected_experience.csv",
        "predicted_config",
        "None",
    )

    for row in rows:
        query_id = row["query_id"]
        row["pg_runtime_s"] = pg[query_id]
        is_default = (
            row["method"] == "FASTgres" and query_id in fastgres_defaults
        ) or (
            row["method"] == "AutoSteer" and query_id in autosteer_defaults
        )
        if row["fallback"] or is_default:
            row["method_runtime_s"] = pg[query_id]
            row["runtime_source"] = (
                "july26_collection_postgres_fallback"
                if row["fallback"]
                else "july26_collection_postgres_default_normalization"
            )
        row["speedup"] = row["pg_runtime_s"] / row["method_runtime_s"]
        row["improved"] = row["method_runtime_s"] < row["pg_runtime_s"]
    return rows


def main() -> None:
    rows = corrected_records()
    per_query_path = CORRECTED_DIR / f"resolved_per_query_{OUTPUT_TAG}.csv"
    summary_path = CORRECTED_DIR / f"summary_two_cases_{OUTPUT_TAG}.csv"
    base.write_csv(per_query_path, rows)
    summaries = [
        *base.summarize(rows, "all_22_with_pg_fallback", set()),
        *base.summarize(
            rows,
            "exclude_q7_q8_q9_q13_q15_q22",
            base.EXCLUDED_QUERIES,
        ),
    ]
    base.write_csv(summary_path, summaries)
    for row in summaries:
        print(
            "{case} {method}: n={query_count} WS={WS:.6f} GS={GS:.6f} "
            "Imp={Imp_pct:.3f}%".format(**row)
        )


if __name__ == "__main__":
    main()
