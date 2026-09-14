"""Result loading and aggregation for benchmark matrices."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from benchmarking.workloads import split_folds


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def combine_training_stages(
    primary_history: list[dict[str, Any]],
    primary_state: dict[str, Any],
    refinement_history: list[dict[str, Any]] | None,
    refinement_state: dict[str, Any] | None,
    *,
    primary_iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge a primary trajectory and a best-checkpoint refinement branch."""

    def tagged(
        entries: list[dict[str, Any]],
        *,
        stage: str,
        offset: int,
        elapsed_offset_s: float = 0.0,
    ) -> list[dict[str, Any]]:
        result = []
        for entry in entries:
            stage_iteration = int(entry["iteration"])
            tagged_entry = {
                **entry,
                "iteration": offset + stage_iteration,
                "stage": stage,
                "stage_iteration": stage_iteration,
            }
            if "elapsed_active_s" in entry:
                tagged_entry["elapsed_active_s"] = (
                    elapsed_offset_s + float(entry["elapsed_active_s"])
                )
            result.append(tagged_entry)
        return result

    combined = tagged(primary_history, stage="primary", offset=0)
    states = {"primary": primary_state}
    if refinement_history is not None and refinement_state is not None:
        # refinement iteration 0 re-evaluates the selected primary checkpoint.
        # It intentionally replaces the primary endpoint in aggregate curves,
        # while both raw checkpoints remain eligible for final selection.
        combined.extend(
            tagged(
                refinement_history,
                stage="refinement",
                offset=primary_iterations,
                elapsed_offset_s=max(
                    (
                        float(entry.get("elapsed_active_s", 0.0))
                        for entry in primary_history
                    ),
                    default=0.0,
                ),
            )
        )
        states["refinement"] = refinement_state

    best_stage, best_state = max(
        states.items(),
        key=lambda item: float(item[1]["best_test_ws"]),
    )
    best_stage_iteration = int(best_state["best_iteration"])
    merged_state = {
        **best_state,
        "best_iteration": (
            best_stage_iteration
            + (primary_iterations if best_stage == "refinement" else 0)
        ),
        "best_stage": best_stage,
        "best_stage_iteration": best_stage_iteration,
        "stage_states": states,
    }
    return combined, merged_state


def aggregate_learning_curves(
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate independently trained folds at their common checkpoints."""
    by_fold = {
        fold: {int(entry["iteration"]): entry for entry in entries}
        for fold, entries in histories.items()
    }
    common_iterations = set.intersection(
        *(set(entries) for entries in by_fold.values())
    )
    result = []
    for iteration in sorted(common_iterations):
        entries = [by_fold[fold][iteration] for fold in sorted(by_fold)]
        test_pg_ms = sum(float(entry["test"]["pg_total_ms"]) for entry in entries)
        test_learned_ms = sum(
            float(entry["test"]["learned_total_ms"]) for entry in entries
        )
        result.append(
            {
                "iteration": iteration,
                "fold_count": len(entries),
                "test_query_instances": sum(
                    int(entry["test"]["query_count"]) for entry in entries
                ),
                "test_pg_total_s": test_pg_ms / 1000.0,
                "test_learned_total_s": test_learned_ms / 1000.0,
                "test_ws": (
                    test_pg_ms / test_learned_ms if test_learned_ms > 0.0 else None
                ),
                "normalized_runtime": (
                    test_learned_ms / test_pg_ms if test_pg_ms > 0.0 else None
                ),
                "unique_subquery_action_pairs": sum(
                    int(
                        entry.get("training", {}).get("unique_subquery_action_pairs", 0)
                    )
                    for entry in entries
                ),
                "training_wall_s": sum(
                    float(entry.get("training", {}).get("measured_wall_ms", 0.0))
                    for entry in entries
                )
                / 1000.0,
                "elapsed_training_s": sum(
                    float(entry.get("elapsed_active_s", 0.0))
                    for entry in entries
                ),
                "training_timeouts": sum(
                    int(entry.get("training", {}).get("timeouts", 0))
                    for entry in entries
                ),
                "test_timeouts": sum(
                    int(entry["test"]["timeouts"]) for entry in entries
                ),
                "test_wrong_results": sum(
                    int(entry["test"]["wrong_results"]) for entry in entries
                ),
                "test_errors": sum(
                    int(entry["test"].get("errors", 0)) for entry in entries
                ),
            }
        )
    return result


def aggregate_selected_checkpoints(
    histories: dict[str, list[dict[str, Any]]],
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = []
    official = []
    for fold in sorted(histories):
        best_iteration = int(states[fold]["best_iteration"])
        best_stage = states[fold].get("best_stage")
        if best_stage is None:
            matching = [
                entry
                for entry in histories[fold]
                if int(entry["iteration"]) == best_iteration
            ]
        else:
            best_stage_iteration = int(states[fold]["best_stage_iteration"])
            matching = [
                entry
                for entry in histories[fold]
                if entry.get("stage") == best_stage
                and int(entry.get("stage_iteration", -1)) == best_stage_iteration
            ]
        if len(matching) != 1:
            raise RuntimeError(
                f"fold {fold} has no unique evaluation for best iteration "
                f"{best_iteration}"
            )
        selected.append(matching[0])
        official.append(states[fold].get("formal_test") or matching[0]["test"])
    pg_total_ms = sum(float(entry["pg_total_ms"]) for entry in official)
    learned_total_ms = sum(float(entry["learned_total_ms"]) for entry in official)
    return {
        "folds": {
            fold: {
                "best_iteration": int(states[fold]["best_iteration"]),
                "best_stage": states[fold].get("best_stage", "primary"),
                "best_stage_iteration": int(
                    states[fold].get(
                        "best_stage_iteration",
                        states[fold]["best_iteration"],
                    )
                ),
                "best_test_ws": float(states[fold]["best_test_ws"]),
                "test_ws": float(
                    (
                        states[fold].get("formal_test")
                        or next(
                            entry["test"]
                            for entry in histories[fold]
                            if (
                                (
                                    states[fold].get("best_stage") is None
                                    and int(entry["iteration"])
                                    == int(states[fold]["best_iteration"])
                                )
                                or (
                                    states[fold].get("best_stage") is not None
                                    and entry.get("stage") == states[fold]["best_stage"]
                                    and int(entry.get("stage_iteration", -1))
                                    == int(states[fold]["best_stage_iteration"])
                                )
                            )
                        )
                    )["workload_speedup"]
                ),
            }
            for fold in sorted(histories)
        },
        "test_pg_total_s": pg_total_ms / 1000.0,
        "test_learned_total_s": learned_total_ms / 1000.0,
        "test_ws": (pg_total_ms / learned_total_ms if learned_total_ms > 0.0 else None),
        "test_timeouts": sum(int(entry["timeouts"]) for entry in official),
        "test_wrong_results": sum(int(entry["wrong_results"]) for entry in official),
        "test_errors": sum(int(entry.get("errors", 0)) for entry in official),
    }




def test_query_multiplicity(
    workload: str,
    protocol: str,
    folds: list[str],
) -> dict[str, Any]:
    fold_specs = split_folds(workload, protocol)
    query_ids = [
        query_id
        for fold in folds
        for query_id in fold_specs[f"{protocol}_{fold}"]["test"]
    ]
    counts = Counter(query_ids)
    return {
        "instances": len(query_ids),
        "unique": len(counts),
        "duplicates": {
            query_id: count for query_id, count in sorted(counts.items()) if count > 1
        },
    }
