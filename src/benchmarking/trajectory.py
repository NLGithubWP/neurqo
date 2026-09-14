"""Trajectory assembly and append-only episode result storage."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from optimization.actions import ingest_trajectory, semantic_policy_trajectory


EPISODE_FIELDS = (
    "result_key",
    "experiment_id",
    "run_id",
    "profile",
    "workload",
    "protocol",
    "fold",
    "role",
    "query_id",
    "repetition",
    "is_warmup",
    "status",
    "client_wall_ms",
    "charged_wall_ms",
    "db_total_ms",
    "pg_baseline_ms",
    "speedup",
    "timeout_limit_ms",
    "result_hash",
    "result_rows",
    "correctness_validation",
    "rounds",
    "decisions",
    "materialized_rows",
    "materialized_bytes",
    "search_applied",
    "lip_filters",
    "aja_decided",
    "cache_hit",
    "cache_source",
    "cache_saved_wall_ms",
    "error",
)


def ingest_benchmark_trajectory(
    *,
    episode_id: str,
    environment_hash: str,
    implementation_version: str,
    db_events: list[dict[str, Any]],
    policy_events: list[dict[str, Any]],
    timeout_limit_ms: int,
    timeout_charged_ms: float,
    episode_status: str,
    cache_hit: bool,
) -> dict[str, Any]:
    """Build one sampled rollout identically for live and cached execution.

    A cache hit replaces only the database execution.  The policy was still
    sampled for the current checkpoint, so its action probability, log
    probability, value estimate, and the cached runtime reward must form the
    same transition that a live execution would have produced.
    """
    counts = ingest_trajectory(
        episode_id=episode_id,
        environment_hash=environment_hash,
        implementation_version=implementation_version,
        db_events=db_events,
        policy_events=policy_events,
        timeout_limit_ms=timeout_limit_ms,
        timeout_charged_ms=timeout_charged_ms,
        episode_status=episode_status,
        runtime_source="cache" if cache_hit else "physical",
    )
    if cache_hit and semantic_policy_trajectory(policy_events) and not counts[
        "decisions"
    ]:
        raise RuntimeError(
            "cached policy rollout did not produce training decisions"
        )
    return counts


class EpisodeCsv:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
                    raise RuntimeError(f"incompatible episodes.csv schema: {self.path}")
                for row in reader:
                    self.records[row["result_key"]] = row

    def append(self, record: dict[str, Any]) -> None:
        unexpected = set(record) - set(EPISODE_FIELDS)
        if unexpected:
            raise ValueError(f"unexpected result columns: {sorted(unexpected)}")
        write_header = not self.path.is_file() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
            handle.flush()
            os.fsync(handle.fileno())
        self.records[str(record["result_key"])] = record

    def profile_records(self, profile: str) -> list[dict[str, Any]]:
        rows = []
        for record in self.records.values():
            if record["profile"] != profile:
                continue
            converted = dict(record)
            for key in ("is_warmup", "cache_hit", "search_applied"):
                converted[key] = str(record.get(key, False)).lower() in {"1", "true"}
            for key in (
                "client_wall_ms",
                "charged_wall_ms",
                "db_total_ms",
                "pg_baseline_ms",
            ):
                if record.get(key) not in ("", None):
                    converted[key] = float(record[key])
            rows.append(converted)
        return rows
