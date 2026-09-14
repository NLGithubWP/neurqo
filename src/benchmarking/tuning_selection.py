"""Pure candidate validation and parameter selection for action tuning."""

from __future__ import annotations

import copy
from typing import Any


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    allow_no_application: bool = False,
) -> dict[str, Any]:
    valid = [
        candidate
        for candidate in candidates
        if candidate["workload_speedup"] is not None and candidate_is_valid(candidate)
    ]
    selection_status = "applied_on_tuning_sample"
    if not valid and allow_no_application:
        valid = [
            candidate
            for candidate in candidates
            if candidate["workload_speedup"] is not None
            and candidate_is_safe(candidate)
        ]
        selection_status = "no_application_on_tuning_sample"
    if not valid:
        raise RuntimeError("no correct candidate completed")
    selected = copy.deepcopy(
        max(valid, key=lambda item: float(item["workload_speedup"]))
    )
    selected["tuning_selection_status"] = selection_status
    return selected


def candidate_is_safe(candidate: dict[str, Any]) -> bool:
    query_count = max(int(candidate.get("query_count") or 1), 1)
    timeout_rate = float(candidate.get("timeouts") or 0) / query_count
    return (
        candidate.get("wrong_results", 0) == 0
        and candidate.get("errors", 0) == 0
        and timeout_rate <= 0.20
    )


def candidate_is_valid(candidate: dict[str, Any]) -> bool:
    profile = str(candidate.get("profile") or "")
    if profile in {"split_search", "top5", "top10"}:
        applied_queries = int(candidate.get("search_applied_queries") or 0)
    elif profile in {"lip_full", "lip_selective"}:
        applied_queries = int(candidate.get("lip_applied_queries") or 0)
    elif profile in {"aja_conservative", "aja_aggressive"}:
        applied_queries = int(candidate.get("aja_applied_queries") or 0)
    else:
        applied_queries = 1
    return candidate_is_safe(candidate) and applied_queries >= 1
