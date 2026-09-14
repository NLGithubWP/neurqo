from __future__ import annotations

import csv
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import benchmarking.matrix as benchmark_matrix
from benchmarking.action_tuning import (
    candidate_run_parameters,
)
from benchmarking.matrix import (
    set_cli_option,
)
from benchmarking.matrix_reports import (
    aggregate_learning_curves,
    aggregate_selected_checkpoints,
    combine_training_stages,
    test_query_multiplicity as query_multiplicity,
)
from benchmarking.orchestration import action_runner_command
from benchmarking.iterative_training import (
    baseline_with_first_runtimes,
    bootstrap_fixed_replay_enabled,
    load_first_pg_runtimes,
    load_profile_summary,
)
from experience.store import ExperienceStore
from benchmarking.training_strategy import (
    build_coverage_snapshot,
    coverage_iteration_query_batch,
    early_stopping_counter,
    iteration_query_batch,
    runtime_stratified_sample as training_runtime_stratified_sample,
    select_checkpoint_finalists,
    select_initial_policy_profile,
    stochastic_heads_for_iteration,
    structural_family_iteration_query_batch,
)
from benchmarking.tuning_selection import (
    candidate_is_valid,
    select_candidate,
)

runtime_stratified_sample = training_runtime_stratified_sample


def evaluation(
    iteration: int,
    *,
    pg_ms: float,
    learned_ms: float,
    unique_pairs: int,
    elapsed_active_s: float | None = None,
) -> dict:
    result = {
        "query_count": 2,
        "pg_total_ms": pg_ms,
        "learned_total_ms": learned_ms,
        "workload_speedup": pg_ms / learned_ms,
        "timeouts": 0,
        "wrong_results": 0,
    }
    entry = {
        "iteration": iteration,
        "test": dict(result),
        "training": {
            "unique_subquery_action_pairs": unique_pairs,
            "measured_wall_ms": learned_ms,
            "timeouts": 0,
        },
    }
    if elapsed_active_s is not None:
        entry["elapsed_active_s"] = elapsed_active_s
    return entry


def test_first_runtime_baseline_and_checkpoint_metrics(tmp_path) -> None:
    episodes_path = tmp_path / "episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("profile", "query_id", "repetition", "charged_wall_ms"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "profile": "pg",
                    "query_id": "q1",
                    "repetition": 0,
                    "charged_wall_ms": 100.0,
                },
                {
                    "profile": "pg",
                    "query_id": "q2",
                    "repetition": 0,
                    "charged_wall_ms": 400.0,
                },
            ]
        )
    first = load_first_pg_runtimes(
        episodes_path,
        expected_query_ids=("q1", "q2"),
        workload="stack",
    )
    baseline = baseline_with_first_runtimes(
        {
            "q1": {"median_charged_ms": 80.0, "result_hash": "one"},
            "q2": {"median_charged_ms": 300.0, "result_hash": "two"},
        },
        first,
    )
    output_root = tmp_path / "output"
    summary_dir = output_root / "stack" / "eval"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "learned@policy": {
                    "queries": {
                        "q1": {
                            "median_charged_ms": 50.0,
                            "timeouts": 0,
                            "wrong_results": 0,
                            "errors": 0,
                        },
                        "q2": {
                            "median_charged_ms": 200.0,
                            "timeouts": 0,
                            "wrong_results": 0,
                            "errors": 0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary = load_profile_summary(
        SimpleNamespace(output_root=output_root, workload="STACK"),
        experiment_id="eval",
        policy_version="policy",
        baseline=baseline,
    )

    assert baseline["q1"]["result_hash"] == "one"
    assert baseline["q1"]["median_charged_ms"] == 100.0
    assert summary["workload_speedup_first"] == 2.0
    assert summary["geometric_mean_speedup_first"] == 2.0
    assert summary["improved_pct_first"] == 100.0


def test_bootstrap_replay_requires_at_least_one_enabled_phase() -> None:
    args = SimpleNamespace(
        fixed_replay_group=["fixed"],
        replay_epochs=0,
        dec_replay_epochs=0,
        sched_replay_epochs=0,
        enum_replay_epochs=0,
        adapt_replay_epochs=0,
        disable_runtime_replay=False,
        disable_bootstrap_fixed_replay=False,
    )
    assert not bootstrap_fixed_replay_enabled(args)
    args.dec_replay_epochs = 1
    assert bootstrap_fixed_replay_enabled(args)


def test_model_collection_has_no_confidence_or_split_threshold_flags() -> None:
    source = inspect.getsource(action_runner_command)
    assert "--high-split-threshold" not in source
    assert "--high-residual-split-threshold" not in source
    assert "--schedule-min-confidence" not in source
    assert "--search-min-confidence" not in source
    assert "--low-min-confidence" not in source


def test_initial_policy_profile_uses_train_only_sum_runtime(tmp_path: Path) -> None:
    summary = {
        profile: {
            "queries": {
                "q1": {"median_charged_ms": runtime},
                "q2": {"median_charged_ms": runtime},
                # This query must not influence the train-only choice.
                "test": {"median_charged_ms": 1 if profile == "top5" else 1000},
            }
        }
        for profile, runtime in {
            "query_split": 20,
            "top5": 80,
            "lip_selective": 90,
            "aja_conservative": 70,
        }.items()
    }
    path = tmp_path / "independent.json"
    path.write_text(__import__("json").dumps(summary), encoding="utf-8")
    baseline = {
        query_id: {"median_charged_ms": 100}
        for query_id in ("q1", "q2", "test")
    }

    selected, speedups = select_initial_policy_profile(
        path,
        baseline,
        ["q1", "q2"],
    )

    assert selected == "query_split"
    assert speedups["query_split"] == 5.0


def test_early_stopping_counter_resets_only_on_required_gain() -> None:
    assert early_stopping_counter(
        best_ws=1.5,
        current_ws=1.51,
        previous_without_improvement=3,
        minimum_delta=0.0,
    ) == 0
    assert early_stopping_counter(
        best_ws=1.5,
        current_ws=1.505,
        previous_without_improvement=3,
        minimum_delta=0.01,
    ) == 4


def test_matrix_uses_one_live_experience_store_across_folds() -> None:
    source = inspect.getsource(benchmark_matrix.main)
    assert "protocol-snapshot.sqlite" not in source
    assert "delta.sqlite" not in source
    assert "configured_db" not in source
    assert "str(dataset_experience_db)" in source
    assert '"--formal-top-k"' in source
    assert "ThreadPoolExecutor" in source
    assert '"--sql-execution-slots"' in inspect.getsource(
        benchmark_matrix.training_runtime_arguments
    )


def test_learning_curve_uses_ratio_of_fold_runtime_sums() -> None:
    histories = {
        "a": [
            evaluation(
                0,
                pg_ms=100.0,
                learned_ms=50.0,
                unique_pairs=3,
                elapsed_active_s=11.0,
            )
        ],
        "b": [
            evaluation(
                0,
                pg_ms=900.0,
                learned_ms=600.0,
                unique_pairs=7,
                elapsed_active_s=29.0,
            )
        ],
    }
    row = aggregate_learning_curves(histories)[0]
    assert row["test_ws"] == 1000.0 / 650.0
    assert row["normalized_runtime"] == 650.0 / 1000.0
    assert row["unique_subquery_action_pairs"] == 10
    assert row["test_query_instances"] == 4
    assert row["elapsed_training_s"] == 40.0


def test_selected_checkpoint_is_fold_local_test_choice() -> None:
    histories = {
        "a": [
            evaluation(0, pg_ms=100.0, learned_ms=100.0, unique_pairs=1),
            evaluation(2, pg_ms=100.0, learned_ms=50.0, unique_pairs=2),
        ],
        "b": [
            evaluation(0, pg_ms=300.0, learned_ms=150.0, unique_pairs=1),
            evaluation(2, pg_ms=300.0, learned_ms=300.0, unique_pairs=2),
        ],
    }
    states = {
        "a": {
            "best_iteration": 2,
            "best_test_ws": 2.0,
        },
        "b": {
            "best_iteration": 0,
            "best_test_ws": 2.0,
        },
    }
    selected = aggregate_selected_checkpoints(histories, states)
    assert selected["test_ws"] == 400.0 / 200.0
    assert selected["folds"]["a"]["best_iteration"] == 2
    assert selected["folds"]["b"]["best_iteration"] == 0


def test_refinement_branch_keeps_best_checkpoint_across_stages() -> None:
    primary = [
        evaluation(
            0,
            pg_ms=100.0,
            learned_ms=100.0,
            unique_pairs=1,
            elapsed_active_s=10.0,
        ),
        evaluation(
            2,
            pg_ms=100.0,
            learned_ms=50.0,
            unique_pairs=2,
            elapsed_active_s=30.0,
        ),
    ]
    refinement = [
        evaluation(
            0,
            pg_ms=100.0,
            learned_ms=55.0,
            unique_pairs=2,
            elapsed_active_s=5.0,
        ),
        evaluation(
            2,
            pg_ms=100.0,
            learned_ms=40.0,
            unique_pairs=3,
            elapsed_active_s=15.0,
        ),
    ]
    history, state = combine_training_stages(
        primary,
        {"best_iteration": 2, "best_test_ws": 2.0},
        refinement,
        {"best_iteration": 2, "best_test_ws": 2.5},
        primary_iterations=2,
    )
    assert state["best_stage"] == "refinement"
    assert state["best_stage_iteration"] == 2
    assert state["best_iteration"] == 4
    assert history[-1]["elapsed_active_s"] == 45.0
    selected = aggregate_selected_checkpoints(
        {"a": history},
        {"a": state},
    )
    assert selected["test_ws"] == 2.5
    assert selected["folds"]["a"]["best_stage"] == "refinement"


def test_refinement_branch_does_not_replace_better_primary() -> None:
    primary = [evaluation(0, pg_ms=100.0, learned_ms=50.0, unique_pairs=1)]
    refinement = [evaluation(0, pg_ms=100.0, learned_ms=60.0, unique_pairs=1)]
    history, state = combine_training_stages(
        primary,
        {"best_iteration": 0, "best_test_ws": 2.0},
        refinement,
        {"best_iteration": 0, "best_test_ws": 5.0 / 3.0},
        primary_iterations=20,
    )
    assert state["best_stage"] == "primary"
    selected = aggregate_selected_checkpoints(
        {"a": history},
        {"a": state},
    )
    assert selected["test_ws"] == 2.0


def test_set_cli_option_replaces_or_appends_once() -> None:
    command = ["python", "train.py", "--iterations", "20"]
    assert set_cli_option(command, "--iterations", 8) == [
        "python",
        "train.py",
        "--iterations",
        "8",
    ]
    assert set_cli_option(command, "--lr", 0.1)[-2:] == ["--lr", "0.1"]




def test_job_leave_one_out_preserves_genjoin_duplicates() -> None:
    summary = query_multiplicity(
        "JOB",
        "leave_one_out",
        ["a", "b", "c"],
    )
    assert summary["instances"] == 115
    assert summary["unique"] == 113
    assert summary["duplicates"] == {"24a": 2, "32a": 2}


def test_runtime_stratified_tuning_sample_spans_fast_and_slow_queries() -> None:
    query_ids = [f"q{index}" for index in range(12)]
    baseline = {
        query_id: {"median_charged_ms": float(index + 1)}
        for index, query_id in enumerate(query_ids)
    }
    sample = runtime_stratified_sample(query_ids, baseline, 4)
    runtimes = [baseline[query_id]["median_charged_ms"] for query_id in sample]
    assert runtimes == [2.0, 5.0, 8.0, 11.0]


def test_runtime_stratified_evaluation_sample_spans_distribution() -> None:
    query_ids = [f"q{index}" for index in range(12)]
    baseline = {
        query_id: {"median_charged_ms": float(index + 1)}
        for index, query_id in enumerate(query_ids)
    }
    sample = training_runtime_stratified_sample(query_ids, baseline, 3)
    runtimes = [baseline[query_id]["median_charged_ms"] for query_id in sample]
    assert runtimes == [2.0, 6.0, 10.0]


def test_iteration_batches_do_not_repeat_queries_across_epoch_boundary() -> None:
    query_ids = [f"q{index}" for index in range(65)]
    for iteration in range(1, 12):
        batch = iteration_query_batch(
            query_ids,
            iteration=iteration,
            batch_size=24,
            seed=42,
        )
        assert len(batch) == 24
        assert len(set(batch)) == 24


def test_structural_family_batches_are_diverse_and_train_whitelisted() -> None:
    training = [
        *(f"{family}{variant}" for family in range(1, 9) for variant in "abc"),
    ]
    held_out = {"9a", "9b", "10a"}
    batch = structural_family_iteration_query_batch(
        training,
        workload="JOB",
        iteration=1,
        batch_size=8,
        seed=42,
    )
    assert len(batch) == 8
    assert len(set(batch)) == 8
    assert set(batch) <= set(training)
    assert not set(batch) & held_out
    assert len({query_id[:-1] for query_id in batch}) == 8


def test_coverage_structural_batch_never_uses_held_out_queries() -> None:
    training = [f"q{family}_train-{variant:03d}" for family in range(1, 9) for variant in range(3)]
    held_out = {f"q{family}_test-999" for family in range(1, 9)}
    baseline = {
        query_id: {"median_charged_ms": float(index + 1)}
        for index, query_id in enumerate(training)
    }
    batch = coverage_iteration_query_batch(
        training,
        iteration=2,
        batch_size=12,
        seed=42,
        coverage_mix=0.3,
        baseline=baseline,
        query_trajectory_counts={query_id: 0 for query_id in training},
        workload="STACK",
        query_sampling="structural_family",
    )
    assert len(batch) == 12
    assert len(set(batch)) == 12
    assert set(batch) <= set(training)
    assert not set(batch) & held_out


def test_coverage_snapshot_excludes_held_out_query_and_derived_states() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        database = Path(tmp) / "experience.sqlite"
        with ExperienceStore(database) as store:
            for query_id in ("1a", "held_out"):
                store.append_execution(
                    query_id=query_id,
                    sql_hash=f"sql-{query_id}",
                    trajectory=[
                        {
                            "round_index": 1,
                            "phase": "dec",
                            "state": {
                                "request_type": "dec",
                                "sql": f"SELECT '{query_id}'",
                                "round": 1,
                            },
                            "state_hash": f"state-{query_id}",
                            "action": {"dec_action": "apply"},
                            "policy": {"action_index": 1},
                        }
                    ],
                    db_events=[],
                    status="ok",
                    first_runtime_ms=10.0,
                    charged_runtime_ms=10.0,
                    timeout_limit_ms=60_000,
                    action_config_hash="config",
                    source_episode_id=f"episode-{query_id}",
                )
        snapshot = build_coverage_snapshot(
            database,
            training_query_ids=["1a"],
            output=Path(tmp) / "coverage.json",
        )
        assert snapshot["training_query_ids"] == ["1a"]
        assert snapshot["query_trajectory_counts"] == {"1a": 1}
        assert sum(sum(values) for values in snapshot["counts"]["dec"].values()) == 1


def test_staged_curriculum_isolates_hierarchical_heads() -> None:
    assert stochastic_heads_for_iteration(1, 20, "staged") == "dec,sched"
    assert stochastic_heads_for_iteration(10, 20, "staged") == "enum"
    assert stochastic_heads_for_iteration(14, 20, "staged") == "adapt"
    assert stochastic_heads_for_iteration(19, 20, "staged") == "dec,sched,enum,adapt"
    assert stochastic_heads_for_iteration(1, 20, "joint") == "dec,sched,enum,adapt"
    assert stochastic_heads_for_iteration(1, 20, "decomposition") == "dec,sched"
    assert (
        stochastic_heads_for_iteration(
            1,
            20,
            "staged",
            split_enabled=False,
        )
        == "enum,adapt"
    )


def test_formal_finalists_use_one_measurement_ws_and_distinct_checkpoints() -> None:
    history = [
        {
            "iteration": 0,
            "checkpoint": "/tmp/iter-0.pt",
            "policy_version": "policy-0",
            "test": {"workload_speedup": 1.0},
        },
        {
            "iteration": 4,
            "checkpoint": "/tmp/iter-4.pt",
            "policy_version": "policy-4",
            "test": {"workload_speedup": 1.2},
        },
        {
            "iteration": 8,
            "checkpoint": "/tmp/iter-8.pt",
            "policy_version": "policy-8",
            "test": {"workload_speedup": 1.1},
        },
    ]
    finalists = select_checkpoint_finalists(history, 2)
    assert [entry["iteration"] for entry in finalists] == [4, 8]










def test_action_tuning_rejects_low_application_coverage() -> None:
    candidate = {
        "profile": "top5",
        "query_count": 100,
        "timeouts": 0,
        "wrong_results": 0,
        "errors": 0,
        "search_applied_queries": 0,
    }
    assert candidate_is_valid(candidate) is False
    candidate["search_applied_queries"] = 1
    assert candidate_is_valid(candidate) is True


def test_action_tuning_can_probe_a_safe_zero_coverage_candidate() -> None:
    candidate = {
        "profile": "aja_conservative",
        "query_count": 12,
        "timeouts": 0,
        "wrong_results": 0,
        "errors": 0,
        "aja_applied_queries": 0,
        "workload_speedup": 0.99,
    }

    selected = select_candidate(
        [candidate],
        allow_no_application=True,
    )

    assert selected["tuning_selection_status"] == ("no_application_on_tuning_sample")


def test_action_tuning_replays_each_variant_with_its_own_parameters() -> None:
    candidate = {
        "parameters": {
            "search_max_rels": 6,
            "lip_max_build_relation_rows": None,
        }
    }

    assert candidate_run_parameters(candidate) == {
        "search_max_rels": 6,
    }
