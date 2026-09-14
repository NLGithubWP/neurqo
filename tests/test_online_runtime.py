from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from model.policy.action_space import coverage_state_hash
from experience.store import ExperienceStore, content_hash
from training.state_builder import ExecutionStateBuilder
from model.encoders.query_graph import CatalogInfo
from optimization.actions import (
    ACTION_CONFIG_SCHEMA_VERSION,
    ActionProfile,
    _semantic_action,
    action_config_hash,
    apply_action_config,
    builtin_profiles,
    dataset_timeout_cap_ms,
    dynamic_timeout_ms,
    global_experience_path,
    hash_result_rows,
    ingest_trajectory,
    load_action_config,
    query_runtime_summary,
    stable_state,
    timeout_charged_runtime_ms,
    validate_policy_state_contract,
    workload_metrics,
    workload_speedup,
)


def test_global_experience_path_is_shared_per_dataset() -> None:
    root = Path("/tmp/pgdb")
    assert global_experience_path(root, "JOB") == (
        root / ".nqo_runtime" / "experience" / "job_light.sql"
    )
    assert global_experience_path(root, "job") == global_experience_path(root, "JOB")
    assert global_experience_path(root, "STACK") != global_experience_path(root, "JOB")


def test_coverage_hash_matches_persisted_semantic_state_hash() -> None:
    state = {
        "request_type": "dec",
        "sql": "select * from temp91 join temp12 using (id)",
        "pid": 123,
        "run_id": 456,
        "remaining_splits": 0,
        "cumulative_cost_ms": 999.0,
    }
    assert coverage_state_hash(state) == content_hash(stable_state(state))


def test_builtin_profiles_isolate_actions() -> None:
    profiles = builtin_profiles()
    assert profiles["nqo_none"].nqo_enabled
    assert profiles["nqo_none"].dec == "skip"
    assert profiles["nqo_none"].enum == "native"
    assert profiles["nqo_none"].filter == "none"
    assert profiles["nqo_none"].ajoin == "off"
    assert profiles["query_split"].dec == "apply"
    assert profiles["query_split"].enum == "native"
    assert profiles["split_search"].dec == "skip"
    assert profiles["split_search"].enum == "top1"
    assert profiles["lip_full"].filter == "full"
    assert profiles["lip_full"].ajoin == "off"
    assert profiles["aja_aggressive"].filter == "none"
    assert profiles["aja_aggressive"].ajoin == "aggressive"
    assert profiles["top5"].search_exact_cardinality is False
    assert profiles["top5"].guc_settings()["nqo.search_exact_cardinality"] is False
    assert "nqo.search_min_cost_improvement_pct" not in (
        profiles["top5"].guc_settings()
    )
    assert profiles["top5_lip_selective"].enum == "top5"
    assert profiles["top5_lip_selective"].filter == "selective"
    assert profiles["top5_aja_conservative"].enum == "top5"
    assert profiles["top5_aja_conservative"].ajoin == "conservative"
    combined = profiles["top5_lip_selective_aja_conservative"]
    assert combined.enum == "top5"
    assert combined.filter == "selective"
    assert combined.ajoin == "conservative"


def test_fixed_profile_exports_per_round_alpha_sequence() -> None:
    profile = ActionProfile(
        name="prefix",
        dec="apply",
        sched_alpha=0.5,
        sched_alpha_sequence=(0.75, 0.25),
    )
    environment = profile.policy_environment()
    assert environment["NQO_FIXED_SCHED_ALPHA"] == "0.5"
    assert environment["NQO_FIXED_SCHED_ALPHA_SEQUENCE"] == "0.75,0.25"


def test_dynamic_timeout_is_bounded() -> None:
    assert dynamic_timeout_ms(100.0) == 5_000
    assert dynamic_timeout_ms(10_000.0) == 22_000
    assert dynamic_timeout_ms(1_000_000.0) == 60_000
    assert dataset_timeout_cap_ms("JOB") == 60_000
    assert dataset_timeout_cap_ms("stack") == 60_000


def test_timeout_charge_uses_pg_runtime_and_cap() -> None:
    assert timeout_charged_runtime_ms(100.0) == 500.0
    assert timeout_charged_runtime_ms(100_000.0) == 360_000.0


def test_first_runtime_timeout_is_five_times_pg_first_capped_at_60s() -> None:
    from optimization.actions import first_runtime_timeout_ms

    assert first_runtime_timeout_ms(123.4) == 617
    assert first_runtime_timeout_ms(12_000.0) == 60_000
    assert first_runtime_timeout_ms(100_000.0) == 60_000
    assert timeout_charged_runtime_ms(100_000.0) == 360_000.0


def test_query_runtime_summary_counts_materialized_split() -> None:
    summary = query_runtime_summary(
        [
            {
                "query_id": "1a",
                "repetition": 2,
                "is_warmup": False,
                "charged_wall_ms": 10.0,
                "client_wall_ms": 10.0,
                "status": "ok",
                "materialized_rows": 0,
                "materialized_bytes": 8192,
                "search_applied": False,
                "lip_filters": 0,
                "aja_decided": 0,
            }
        ]
    )

    assert summary["1a"]["split_applied"] is True
    assert summary["1a"]["action_applied"] is True


def test_result_hash_ignores_unspecified_row_order() -> None:
    first, first_count = hash_result_rows([(1, "a"), (2, "b"), (1, "a")])
    second, second_count = hash_result_rows([(1, "a"), (1, "a"), (2, "b")])

    assert first == second
    assert first_count == second_count == 3


def test_result_hash_preserves_duplicate_multiplicity_and_types() -> None:
    duplicated, _ = hash_result_rows([(1,), (1,), (2,)])
    distinct, _ = hash_result_rows([(1,), (2,)])
    string_value, _ = hash_result_rows([("1",), (2,)])

    assert duplicated != distinct
    assert distinct != string_value


def test_frozen_action_config_validates_workload_and_hash() -> None:
    parameters = {
        "sched_alpha": 0.75,
        "max_rounds": 16,
        "search_max_rels": 12,
        "search_exact_cardinality": False,
        "aja_conservative_rows": 1000,
        "aja_aggressive_rows": 10000,
        "aja_max_nestloop_cost_ratio_pct": 150,
        "aja_aggressive_max_nestloop_cost_ratio_pct": 125,
        "lip_max_build_relation_rows": 100000,
        "lip_selective_plan_rows": 10000,
        "lip_max_build_selectivity_pct": 10,
        "lip_min_probe_ratio": 2,
        "lip_max_filters": 4,
    }
    config = {
        "schema_version": ACTION_CONFIG_SCHEMA_VERSION,
        "status": "frozen",
        "workload": "JOB",
        "parameters": parameters,
        "execution_protocol": {
            "warmups": 2,
            "measurements": 1,
            "timeout_max_ms": 60000,
        },
    }
    config["config_hash"] = action_config_hash(config)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "job.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_action_config(path, workload="job")

    assert loaded["parameters"]["sched_alpha"] == 0.75


def test_action_config_applies_profile_specific_parameters() -> None:
    config = {
        "config_hash": "test",
        "parameters": {
            "search_max_rels": 6,
        },
        "profile_parameters": {
            "top5": {
                "search_max_rels": 8,
            }
        },
    }
    namespace = SimpleNamespace(
        profiles="top5",
        search_max_rels=12,
    )

    apply_action_config(namespace, config)

    assert namespace.search_max_rels == 8


def test_online_state_builder_uses_blob_identity_for_cache() -> None:
    builder = ExecutionStateBuilder("job", catalog=CatalogInfo({}))
    state = {
        "request_type": "dec",
        "sql": "SELECT * FROM title AS t",
        "relations": [
            {
                "alias": "t",
                "relname": "title",
                "estimated_rows": 100,
                "pages": 10,
            }
        ],
    }
    structured = builder.build("dec", "blob-1", state)
    assert structured.cache_key == ("online", "dec", "none", "blob-1")
    assert builder.build("dec", "blob-1", state) is structured


def test_semantic_action_ignores_policy_specific_aliases() -> None:
    fixed = {
        "ajoin_action": "aggressive",
        "filter_action": "selective",
    }
    learned = {
        **fixed,
        "adapt_action": "filter_selective+ajoin_aggressive",
        "action_index": 5,
        "policy_version": "iteration-7",
    }
    assert _semantic_action("adapt", fixed) == _semantic_action("adapt", learned)

    fixed_enum = {"enum_action": "top5", "enum_k": 5}
    learned_enum = {**fixed_enum, "enum_action": "top5"}
    assert _semantic_action("enum", fixed_enum) == _semantic_action(
        "enum", learned_enum
    )


def test_stable_state_normalizes_ephemeral_temp_relation_names() -> None:
    first = {
        "pid": 1,
        "run_id": 10,
        "cumulative_cost_ms": 12.0,
        "sql": "SELECT * FROM temp41 JOIN temp42 USING (id)",
        "relations": [
            {"relid": 100, "relname": "temp41"},
            {"relid": 101, "relname": "temp42"},
        ],
    }
    repeated = {
        "pid": 2,
        "run_id": 11,
        "cumulative_cost_ms": 18.0,
        "sql": "SELECT * FROM temp99 JOIN temp100 USING (id)",
        "relations": [
            {"relid": 200, "relname": "temp99"},
            {"relid": 201, "relname": "temp100"},
        ],
    }
    assert stable_state(first) == stable_state(repeated)


def test_stable_state_excludes_planner_measurement_latency() -> None:
    first = {"sql": "SELECT 1", "plan_state_ms": 1.5}
    repeated = {"sql": "SELECT 1", "plan_state_ms": 27.0}

    assert stable_state(first) == stable_state(repeated)


def test_policy_state_contract_accepts_hierarchical_states() -> None:
    events = [
        {
            "phase": "policy_decision",
            "state": {
                "request_type": "dec",
                "sql": "SELECT * FROM title AS t",
                "relations": [{"alias": "t"}],
                "round": 0,
            },
        },
        {
            "phase": "policy_decision",
            "state": {
                "request_type": "enum",
                "sql": "SELECT * FROM title AS t",
                "relations": [{"alias": "t"}],
                "round": 0,
            },
        },
        {
            "phase": "policy_decision",
            "state": {
                "request_type": "adapt",
                "plan_available": True,
                "plan_json": {"node": "SeqScan"},
                "enum_action": "default",
                "round": 0,
            },
        },
    ]

    validate_policy_state_contract(events)


@pytest.mark.parametrize(
    ("phase", "leaked_field"),
    [
        ("high", "plan_json"),
        ("search", "plan_total_cost"),
        ("low", "sql"),
        ("low", "relations"),
    ],
)
def test_policy_state_contract_rejects_cross_layer_leaks(
    phase: str, leaked_field: str
) -> None:
    state = {
        "request_type": phase,
        "plan_available": True,
        "plan_json": {"node": "HashJoin"},
    }
    if phase != "low":
        state.pop("plan_available")
        state.pop("plan_json")
    state[leaked_field] = {"node": "SeqScan"} if leaked_field == "plan_json" else []

    with pytest.raises(ValueError, match="hierarchical contract"):
        validate_policy_state_contract([{"phase": "policy_decision", "state": state}])


def test_trajectory_merge_keeps_phase_specific_state_and_action() -> None:
    states = {
        "dec": {
            "pid": 7,
            "run_id": 2,
            "round": 0,
            "request_type": "dec",
            "sql": "SELECT 1",
        },
        "sched": {
            "pid": 7,
            "run_id": 2,
            "round": 0,
            "request_type": "sched",
            "candidates": [{"candidate_id": 0}],
        },
        "enum": {
            "pid": 7,
            "run_id": 2,
            "round": 0,
            "request_type": "enum",
            "sql": "SELECT 1",
        },
        "adapt": {
            "pid": 7,
            "run_id": 2,
            "round": 0,
            "request_type": "adapt",
            "plan_json": {"node": "HashJoin"},
        },
    }
    db_event = {
        "round": 0,
        "phase": "split",
        "decision_states": states,
        "action": {
            "dec_action": "apply",
            "candidate_id": 0,
            "enum_action": "top5",
            "enum_k": 5,
            "ajoin_action": "off",
            "filter_action": "selective",
        },
        "timing_ms": {
            "search": 7.0,
            "planning": 20.0,
            "execution": 80.0,
            "total": 110.0,
        },
        "materialized": {"rows": 5, "bytes": 8192},
    }
    policy_events = []
    for phase, state in states.items():
        policy_events.append(
            {
                "phase": "policy_decision",
                "state": state,
                "action": {
                    "action": phase,
                    "action_index": 1,
                    "action_probability": 0.4,
                    "log_probability": -0.916,
                    "predicted_value": 0.5,
                    "policy_version": "v3",
                },
            }
        )

    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            counts = ingest_trajectory(
                episode_id="episode-1",
                environment_hash="environment",
                implementation_version="impl",
                db_events=[db_event],
                policy_events=policy_events,
                timeout_limit_ms=10_000,
                episode_status="ok",
            )

            assert counts["rounds"] == 1
            assert counts["decisions"] == 4
            decisions = counts["trajectory"]
            assert {row["phase"] for row in decisions} == {
                "dec",
                "sched",
                "enum",
                "adapt",
            }
            assert all(
                row["policy"]["action_probability"] == 0.4
                for row in decisions
            )
            adapt = next(row for row in decisions if row["phase"] == "adapt")
            assert adapt["runtime_ms"] == 110.0
            assert db_event["timing_ms"]["search"] == 7.0


def test_workload_speedup_uses_sums() -> None:
    baseline = {
        "a": {"median_charged_ms": 100.0},
        "b": {"median_charged_ms": 300.0},
    }
    profile = {
        "a": {"median_charged_ms": 50.0},
        "b": {"median_charged_ms": 150.0},
    }
    assert workload_speedup(profile, baseline) == 2.0


def test_workload_metrics_report_ws_gs_imp_and_failures() -> None:
    baseline = {
        "a": {"median_charged_ms": 100.0},
        "b": {"median_charged_ms": 300.0},
    }
    profile = {
        "a": {
            "median_charged_ms": 50.0,
            "timeouts": 0,
            "wrong_results": 0,
            "errors": 0,
            "search_applied": True,
            "lip_filters": 2,
            "aja_decided": 0,
            "action_applied": True,
        },
        "b": {
            "median_charged_ms": 600.0,
            "timeouts": 1,
            "wrong_results": 0,
            "errors": 0,
            "search_applied": False,
            "lip_filters": 0,
            "aja_decided": 1,
            "action_applied": True,
        },
    }

    metrics = workload_metrics(profile, baseline)

    assert metrics["workload_speedup"] == 400.0 / 650.0
    assert metrics["geometric_mean_speedup"] == 1.0
    assert metrics["improved_queries"] == 1
    assert metrics["improved_pct"] == 50.0
    assert metrics["regressed_queries"] == 1
    assert metrics["timeouts"] == 1
    assert metrics["search_application_pct"] == 50.0
    assert metrics["lip_application_pct"] == 50.0
    assert metrics["aja_application_pct"] == 50.0
    assert metrics["action_application_pct"] == 100.0


def test_workload_metrics_rejects_incomplete_expected_coverage() -> None:
    baseline = {"a": {"median_charged_ms": 100.0}}
    profile = {"a": {"median_charged_ms": 50.0}}

    with pytest.raises(ValueError, match="incomplete workload coverage"):
        workload_metrics(
            profile,
            baseline,
            expected_query_ids=("a", "b"),
        )
