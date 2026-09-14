"""Stable CSV schemas and result materialization for benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from benchmarking.trajectory import EpisodeCsv


PER_QUERY_FIELDS = (
    "profile",
    "query_id",
    "run_1_ms",
    "run_2_ms",
    "run_3_ms",
    "official_repetition",
    "query_wall_ms",
    "charged_wall_ms",
    "pg_baseline_ms",
    "speedup",
    "status",
    "correct",
    "correctness_validation",
    "result_hash",
    "result_rows",
    "materialized_rows",
    "materialized_bytes",
    "split_applied",
    "search_applied",
    "lip_filters",
    "aja_decided",
    "action_applied",
    "cache_hit",
    "cache_source",
    "timeout_actual_wall_ms",
    "timeout_charged_ms",
)
ACTION_SUMMARY_FIELDS = (
    "profile",
    "query_count",
    "pg_total_ms",
    "action_total_ms",
    "workload_speedup",
    "geometric_mean_speedup",
    "improved_queries",
    "improved_pct",
    "regressed_queries",
    "timeouts",
    "wrong_results",
    "errors",
    "split_applied_queries",
    "split_application_pct",
    "search_applied_queries",
    "search_application_pct",
    "lip_applied_queries",
    "lip_application_pct",
    "aja_applied_queries",
    "aja_application_pct",
    "action_applied_queries",
    "action_application_pct",
    "coverage_complete",
    "valid",
)


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_result_tables(
    output_dir: Path,
    episodes: EpisodeCsv,
    summaries: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    per_query_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for profile, payload in sorted(summaries.items()):
        records = episodes.profile_records(profile)
        records_by_query: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            records_by_query.setdefault(str(record["query_id"]), []).append(record)
        for query_id, query in sorted(payload.get("queries", {}).items()):
            query_records = records_by_query.get(query_id, [])
            runs = {
                int(record["repetition"]): float(record["client_wall_ms"])
                for record in query_records
            }
            official_record = max(
                (record for record in query_records if not record.get("is_warmup")),
                key=lambda record: int(record.get("repetition") or 0),
                default={},
            )
            pg_ms = float(
                baseline.get(query_id, {}).get(
                    "median_charged_ms",
                    query["official_charged_ms"],
                )
            )
            charged_ms = float(query["official_charged_ms"])
            status = str(query["status"])
            per_query_rows.append(
                {
                    "profile": profile,
                    "query_id": query_id,
                    "run_1_ms": runs.get(0, ""),
                    "run_2_ms": runs.get(1, ""),
                    "run_3_ms": runs.get(2, ""),
                    "official_repetition": int(query["official_repetition"]) + 1,
                    "query_wall_ms": query["official_client_wall_ms"],
                    "charged_wall_ms": charged_ms,
                    "pg_baseline_ms": pg_ms,
                    "speedup": pg_ms / charged_ms if charged_ms > 0.0 else "",
                    "status": status,
                    "correct": status == "ok",
                    "correctness_validation": query.get("correctness_validation", ""),
                    "result_hash": query.get("result_hash") or "",
                    "result_rows": query.get("result_rows") or 0,
                    "materialized_rows": query.get("materialized_rows") or 0,
                    "materialized_bytes": query.get("materialized_bytes") or 0,
                    "split_applied": bool(query.get("split_applied", False)),
                    "search_applied": bool(query.get("search_applied", False)),
                    "lip_filters": int(query.get("lip_filters") or 0),
                    "aja_decided": int(query.get("aja_decided") or 0),
                    "action_applied": bool(query.get("action_applied", False)),
                    "cache_hit": bool(official_record.get("cache_hit", False)),
                    "cache_source": official_record.get("cache_source", ""),
                    "timeout_actual_wall_ms": (
                        query["official_client_wall_ms"] if status == "timeout" else ""
                    ),
                    "timeout_charged_ms": (charged_ms if status == "timeout" else ""),
                }
            )
        metrics = payload.get("metrics", {})
        action_rows.append(
            {
                field: profile if field == "profile" else metrics.get(field, "")
                for field in ACTION_SUMMARY_FIELDS
            }
        )
    write_csv_atomic(
        output_dir / "per_query_results.csv",
        per_query_rows,
        PER_QUERY_FIELDS,
    )
    write_csv_atomic(
        output_dir / "action_summary.csv",
        action_rows,
        ACTION_SUMMARY_FIELDS,
    )
