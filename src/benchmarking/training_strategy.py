"""Query acquisition, exploration, and checkpoint-selection strategies."""

from __future__ import annotations

import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmarking.utils import write_json_atomic
from experience.store import ExperienceStore, content_hash
from optimization.action_vocabulary import (
    DECISION_PHASES,
    adapt_label,
    canonical_phase,
    normalize_policy_action,
)


COVERAGE_ADAPT_LABELS = (
    "none",
    "filter_selective",
    "ajoin_conservative",
    "filter_selective+ajoin_conservative",
)
COVERAGE_SCHED_ALPHA_VALUES = (0.0, 0.5, 1.0)
COVERAGE_ENUM_LABELS = ("native", "top5")
INDEPENDENT_ACTION_PROFILES = (
    "query_split",
    "top5",
    "lip_selective",
    "aja_conservative",
)


def _structural_query_family(query_id: str, *, workload: str) -> str:
    """Return the workload's stable template family without model imports."""
    if "_" in query_id:
        return query_id.split("_", 1)[0]
    match = re.match(r"^([A-Za-z]*\d+)", query_id)
    return match.group(1) if match else query_id


def iteration_query_batch(
    query_ids: list[str],
    *,
    iteration: int,
    batch_size: int,
    seed: int,
) -> list[str]:
    if batch_size <= 0 or batch_size >= len(query_ids):
        return list(query_ids)
    start = (iteration - 1) * batch_size
    epoch = start // len(query_ids)
    offset = start % len(query_ids)
    selected: list[str] = []
    while len(selected) < batch_size:
        order = list(query_ids)
        random.Random(seed + epoch).shuffle(order)
        for query_id in order[offset:]:
            if query_id not in selected:
                selected.append(query_id)
            if len(selected) == batch_size:
                break
        epoch += 1
        offset = 0
    return selected


def _structural_family_epoch_order(
    query_ids: list[str],
    *,
    workload: str,
    seed: int,
    epoch: int,
) -> list[str]:
    """Return one train-whitelisted epoch interleaved across query families."""
    families: dict[str, list[str]] = {}
    for query_id in query_ids:
        family = _structural_query_family(query_id, workload=workload.lower())
        families.setdefault(family, []).append(query_id)

    family_order = sorted(families)
    random.Random(seed * 1_000_003 + epoch).shuffle(family_order)
    member_orders: dict[str, list[str]] = {}
    for family in family_order:
        members = sorted(families[family])
        family_seed = seed + epoch * 104_729 + int(
            content_hash({"family": family})[:12], 16
        )
        random.Random(family_seed).shuffle(members)
        member_orders[family] = members

    order: list[str] = []
    depth = 0
    while len(order) < len(query_ids):
        added = False
        for family in family_order:
            members = member_orders[family]
            if depth < len(members):
                order.append(members[depth])
                added = True
        if not added:
            break
        depth += 1
    if len(order) != len(query_ids) or set(order) != set(query_ids):
        raise RuntimeError("structural-family epoch escaped its query whitelist")
    return order


def structural_family_iteration_query_batch(
    query_ids: list[str],
    *,
    workload: str,
    iteration: int,
    batch_size: int,
    seed: int,
) -> list[str]:
    """Round-robin structural families without leaving the training whitelist."""
    if batch_size <= 0 or batch_size >= len(query_ids):
        return list(query_ids)
    start = (iteration - 1) * batch_size
    epoch = start // len(query_ids)
    offset = start % len(query_ids)
    selected: list[str] = []
    whitelist = set(query_ids)
    while len(selected) < batch_size:
        order = _structural_family_epoch_order(
            query_ids,
            workload=workload,
            seed=seed,
            epoch=epoch,
        )
        for query_id in order[offset:]:
            if query_id in whitelist and query_id not in selected:
                selected.append(query_id)
            if len(selected) == batch_size:
                break
        epoch += 1
        offset = 0
    if not set(selected) <= whitelist:
        raise RuntimeError("structural-family batch contains a non-training query")
    return selected


def _coverage_action_index(phase: str, action: dict[str, Any]) -> int | None:
    phase = canonical_phase(phase)
    action = normalize_policy_action(action, phase=phase)
    raw_index = action.get("action_index")
    sizes = {
        "dec": 2,
        "sched": len(COVERAGE_SCHED_ALPHA_VALUES),
        "enum": len(COVERAGE_ENUM_LABELS),
        "adapt": len(COVERAGE_ADAPT_LABELS),
    }
    if raw_index is not None:
        index = int(raw_index)
        return index if 0 <= index < sizes.get(phase, 0) else None
    if phase == "dec":
        return 1 if action.get("dec_action") == "apply" else 0
    if phase == "sched":
        if action.get("sched_alpha") is None:
            return None
        alpha = float(action["sched_alpha"])
        return min(
            range(len(COVERAGE_SCHED_ALPHA_VALUES)),
            key=lambda index: abs(COVERAGE_SCHED_ALPHA_VALUES[index] - alpha),
        )
    if phase == "enum":
        return 1 if action.get("enum_action") != "native" else 0
    if phase == "adapt":
        label = adapt_label(
            action.get("filter_action"), action.get("ajoin_action")
        )
        label = label.replace("filter_full", "filter_selective")
        label = label.replace("ajoin_aggressive", "ajoin_conservative")
        return COVERAGE_ADAPT_LABELS.index(label)
    return None


def build_coverage_snapshot(
    experience_db: Path,
    *,
    training_query_ids: list[str],
    output: Path,
) -> dict[str, Any]:
    """Snapshot train-whitelisted state/action and trajectory coverage."""
    counts: dict[str, dict[str, list[int]]] = {
        phase: {} for phase in DECISION_PHASES
    }
    query_trajectory_counts = {query_id: 0 for query_id in training_query_ids}
    with ExperienceStore(experience_db, read_only=True) as store:
        trajectories: dict[str, set[str]] = {
            query_id: set() for query_id in training_query_ids
        }
        for execution in store.iter_executions(query_ids=training_query_ids):
            query_id = str(execution["query_id"])
            trajectories[query_id].add(str(execution["trajectory_hash"]))
            for decision in execution["trajectory"]:
                phase = canonical_phase(decision.get("phase"))
                if phase not in counts:
                    continue
                action = {
                    **(decision.get("action") or {}),
                    **(decision.get("policy") or {}),
                }
                action_index = _coverage_action_index(phase, action)
                if action_index is None:
                    continue
                state_counts = counts[phase].setdefault(
                    str(decision["state_hash"]),
                    [0]
                    * {
                        "dec": 2,
                        "sched": len(COVERAGE_SCHED_ALPHA_VALUES),
                        "enum": len(COVERAGE_ENUM_LABELS),
                        "adapt": len(COVERAGE_ADAPT_LABELS),
                    }[phase],
                )
                state_counts[action_index] += 1
        query_trajectory_counts = {
            query_id: len(values) for query_id, values in trajectories.items()
        }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_query_ids": sorted(training_query_ids),
        "counts": counts,
        "query_trajectory_counts": query_trajectory_counts,
    }
    write_json_atomic(output, payload)
    return payload


def coverage_iteration_query_batch(
    query_ids: list[str],
    *,
    iteration: int,
    batch_size: int,
    seed: int,
    coverage_mix: float,
    baseline: dict[str, Any],
    query_trajectory_counts: dict[str, int],
    workload: str = "job",
    query_sampling: str = "cyclic",
) -> list[str]:
    """Mix family-stratified coverage with underexplored training queries."""
    if query_sampling not in {"cyclic", "structural_family"}:
        raise ValueError(f"unknown query sampling mode {query_sampling!r}")
    batch_builder = (
        structural_family_iteration_query_batch
        if query_sampling == "structural_family"
        else iteration_query_batch
    )
    batch_kwargs: dict[str, Any] = {
        "iteration": iteration,
        "batch_size": batch_size,
        "seed": seed,
    }
    if query_sampling == "structural_family":
        batch_kwargs["workload"] = workload
    if coverage_mix <= 0.0 or batch_size <= 0 or batch_size >= len(query_ids):
        return batch_builder(query_ids, **batch_kwargs)
    directed_count = min(int(round(batch_size * coverage_mix)), batch_size - 1)
    cyclic_count = batch_size - directed_count
    batch_kwargs["batch_size"] = cyclic_count
    selected = batch_builder(query_ids, **batch_kwargs)
    rng = random.Random(seed * 1_000_003 + iteration)
    candidates = []
    for query_id in query_ids:
        if query_id in selected:
            continue
        trajectories = max(int(query_trajectory_counts.get(query_id, 0)), 0)
        # Runtime magnitude already enters the PPO reward. Query acquisition
        # therefore targets coverage only, avoiding a second implicit runtime
        # weight on top of the sum-runtime objective.
        score = 1.0 / math.sqrt(trajectories + 1.0)
        candidates.append((score * (0.95 + 0.1 * rng.random()), query_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if query_sampling == "structural_family":
        family_counts: dict[str, int] = {}
        for query_id in selected:
            family = _structural_query_family(query_id, workload=workload.lower())
            family_counts[family] = family_counts.get(family, 0) + 1
        remaining = candidates[:]
        while remaining and len(selected) < batch_size:
            best_index = min(
                range(len(remaining)),
                key=lambda index: (
                    family_counts.get(
                        _structural_query_family(
                            remaining[index][1], workload=workload.lower()
                        ),
                        0,
                    ),
                    -remaining[index][0],
                    remaining[index][1],
                ),
            )
            _score, query_id = remaining.pop(best_index)
            selected.append(query_id)
            family = _structural_query_family(query_id, workload=workload.lower())
            family_counts[family] = family_counts.get(family, 0) + 1
    else:
        selected.extend(
            query_id for _score, query_id in candidates[:directed_count]
        )
    whitelist = set(query_ids)
    if len(selected) != batch_size or not set(selected) <= whitelist:
        raise RuntimeError("coverage batch escaped its training-query whitelist")
    return selected


def stochastic_heads_for_iteration(
    iteration: int,
    total_iterations: int,
    curriculum: str,
    *,
    split_enabled: bool = True,
) -> str:
    """Select the policy heads explored in a hierarchical training stage."""
    if not split_enabled:
        return "enum,adapt"
    if curriculum == "decomposition":
        return "dec,sched"
    if curriculum == "joint" or total_iterations < 4:
        return "dec,sched,enum,adapt"
    progress = (iteration - 1) / max(total_iterations, 1)
    if progress < 0.45:
        return "dec,sched"
    if progress < 0.65:
        return "enum"
    if progress < 0.85:
        return "adapt"
    return "dec,sched,enum,adapt"


def select_checkpoint_finalists(
    history: Iterable[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Select distinct checkpoints by their one-measurement test WS."""
    if top_k < 1:
        raise ValueError("formal checkpoint count must be positive")
    ranked = sorted(
        history,
        key=lambda entry: (
            -float(entry["test"]["workload_speedup"] or 0.0),
            int(entry["iteration"]),
        ),
    )
    selected = []
    seen_checkpoints = set()
    for entry in ranked:
        checkpoint = str(entry["checkpoint"])
        if checkpoint in seen_checkpoints:
            continue
        selected.append(entry)
        seen_checkpoints.add(checkpoint)
        if len(selected) == top_k:
            break
    return selected


def early_stopping_counter(
    *,
    best_ws: float,
    current_ws: float,
    previous_without_improvement: int,
    minimum_delta: float,
) -> int:
    """Return consecutive evaluation points without a meaningful WS gain."""
    if current_ws > best_ws + minimum_delta:
        return 0
    return previous_without_improvement + 1


def runtime_stratified_sample(
    query_ids: list[str],
    baseline: dict[str, Any],
    limit: int | None,
) -> list[str]:
    if limit is None or limit >= len(query_ids):
        return list(query_ids)
    if limit < 1:
        raise ValueError("evaluation limits must be positive")
    ordered = sorted(
        query_ids,
        key=lambda query_id: (
            float(baseline[query_id]["median_charged_ms"]),
            query_id,
        ),
    )
    selected = []
    for bucket in range(limit):
        start = bucket * len(ordered) // limit
        end = (bucket + 1) * len(ordered) // limit
        selected.append(ordered[(start + end - 1) // 2])
    return selected


def select_initial_policy_profile(
    summary_path: Path,
    baseline: dict[str, Any],
    training_query_ids: Iterable[str],
) -> tuple[str, dict[str, float]]:
    """Choose the strongest independent policy using train-only sum runtime."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    query_ids = list(training_query_ids)
    pg_total_ms = sum(
        float(baseline[query_id]["median_charged_ms"])
        for query_id in query_ids
    )
    speedups: dict[str, float] = {"postgres": 1.0}
    for profile in INDEPENDENT_ACTION_PROFILES:
        profile_summary = summary.get(profile)
        if not isinstance(profile_summary, dict):
            raise ValueError(
                f"independent-action summary is missing profile {profile!r}"
            )
        queries = profile_summary.get("queries") or {}
        missing = sorted(set(query_ids) - set(queries))
        if missing:
            raise ValueError(
                f"independent-action profile {profile!r} is missing "
                f"{len(missing)} training queries"
            )
        action_total_ms = sum(
            float(queries[query_id]["median_charged_ms"])
            for query_id in query_ids
        )
        speedups[profile] = (
            pg_total_ms / action_total_ms if action_total_ms > 0.0 else 0.0
        )
    selected = max(
        speedups,
        key=lambda profile: (speedups[profile], profile == "postgres"),
    )
    return selected, speedups
