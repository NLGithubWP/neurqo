import csv
import json

import pytest

from benchmarking.action_runner import (
    acquire_sql_execution_slot,
    release_sql_execution_slot,
)
from benchmarking.run_environment import validate_resume_manifest
from benchmarking.execution_cache import (
    cached_trajectory_state_key,
    replay_cached_trajectory,
    replay_cached_trajectories_statewise,
    select_cached_trajectory_by_state_actions,
    unique_cached_trajectory_states,
)
from benchmarking.trajectory import (
    EPISODE_FIELDS,
    EpisodeCsv,
    ingest_benchmark_trajectory,
)
from experience.store import (
    ExperienceStore,
    content_hash,
    semantic_trajectory_hash,
)
from optimization.actions import stable_state


def test_resume_manifest_accepts_exact_contract(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": "2026-07-30T00:00:00+00:00",
                "resume_contract": {"implementation_version": "abc"},
            }
        ),
        encoding="utf-8",
    )

    manifest = validate_resume_manifest(
        manifest_path,
        {"implementation_version": "abc"},
    )

    assert manifest["created_at"] == "2026-07-30T00:00:00+00:00"


def test_sql_execution_lock_pool_admits_two_distinct_slots(tmp_path):
    lock_path = tmp_path / "stack-sql.lock"
    first, first_index = acquire_sql_execution_slot(lock_path, 2)
    second, second_index = acquire_sql_execution_slot(lock_path, 2)
    try:
        assert {first_index, second_index} == {0, 1}
    finally:
        release_sql_execution_slot(second)
        release_sql_execution_slot(first)


def test_resume_manifest_compares_json_sequences_canonically(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"resume_contract": {"profiles": [{"sched_alpha_sequence": [0.5, 1.0]}]}}
        ),
        encoding="utf-8",
    )

    validate_resume_manifest(
        manifest_path,
        {"profiles": [{"sched_alpha_sequence": (0.5, 1.0)}]},
    )


def test_resume_manifest_rejects_changed_contract(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"resume_contract": {"implementation_version": "old"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="implementation_version"):
        validate_resume_manifest(
            manifest_path,
            {"implementation_version": "new"},
        )


def test_resume_manifest_is_required_for_existing_episodes(tmp_path):
    (tmp_path / "episodes.csv").write_text("profile,query_id\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without a manifest"):
        validate_resume_manifest(
            tmp_path / "manifest.json",
            {"implementation_version": "abc"},
        )


def test_episode_csv_restores_false_search_applied(tmp_path):
    path = tmp_path / "episodes.csv"
    row = {field: "" for field in EPISODE_FIELDS}
    row.update(
        {
            "result_key": "run:q:0",
            "profile": "query_split",
            "query_id": "q",
            "is_warmup": "False",
            "cache_hit": "False",
            "search_applied": "False",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    records = EpisodeCsv(path).profile_records("query_split")

    assert records[0]["search_applied"] is False


def test_empty_bypass_trajectory_matches_without_contacting_server(tmp_path):
    matched, events, offset = replay_cached_trajectory(
        container="unused",
        server_url="http://unused",
        policy_log=tmp_path / "missing.jsonl",
        policy_offset=17,
        trajectory=[],
    )

    assert matched is True
    assert events == []
    assert offset == 17


def _cached_candidate(cache_id, trajectory):
    return {
        "cache_id": cache_id,
        "trajectory": trajectory,
        "trajectory_hash": cache_id,
    }


def _cached_step(phase, state_hash, action):
    return {
        "phase": phase,
        "state_hash": state_hash,
        "state": {"request_type": phase, "semantic_id": state_hash},
        "action": action,
    }


def test_statewise_cache_match_evaluates_shared_graph_state_once():
    root_stop = _cached_step("dec", "query-graph", {"dec_action": "skip"})
    root_split = _cached_step("dec", "query-graph", {"dec_action": "apply"})
    select_half = _cached_step(
        "sched",
        "residual-graph",
        {"candidate_id": 2, "sched_alpha": 0.5},
    )
    low_none = _cached_step(
        "adapt",
        "plan-tree",
        {"ajoin_action": "off", "filter_action": "none"},
    )
    candidates = [
        _cached_candidate("new-stop", [root_stop]),
        _cached_candidate("older-split", [root_split, select_half, low_none]),
    ]

    states = unique_cached_trajectory_states(candidates)
    assert [cached_trajectory_state_key(item) for item in states] == [
        ("dec", "query-graph"),
        ("sched", "residual-graph"),
        ("adapt", "plan-tree"),
    ]
    selected = select_cached_trajectory_by_state_actions(
        candidates,
        {
            ("dec", "query-graph"): {"dec_action": "apply"},
            ("sched", "residual-graph"): {
                "candidate_id": 2,
                "sched_alpha": 0.5,
            },
            ("adapt", "plan-tree"): {
                "ajoin_action": "off",
                "filter_action": "none",
            },
        },
    )

    assert selected["cache_id"] == "older-split"


def test_statewise_cache_match_requires_one_complete_observed_trajectory():
    candidates = [
        _cached_candidate(
            "split-top5",
            [
                _cached_step("dec", "query", {"dec_action": "apply"}),
                _cached_step(
                    "enum",
                    "plan",
                    {"enum_action": "top5", "enum_k": 5},
                ),
            ],
        ),
        _cached_candidate(
            "stop-default",
            [
                _cached_step("dec", "query", {"dec_action": "skip"}),
                _cached_step(
                    "enum",
                    "plan",
                    {"enum_action": "native", "enum_k": 1},
                ),
            ],
        ),
    ]

    selected = select_cached_trajectory_by_state_actions(
        candidates,
        {
            ("dec", "query"): {"dec_action": "apply"},
            ("enum", "plan"): {"enum_action": "native", "enum_k": 1},
        },
    )

    assert selected is None


def test_statewise_cache_match_handles_deterministic_bypass():
    first = _cached_candidate("newest", [])
    second = _cached_candidate("older", [])

    assert (
        select_cached_trajectory_by_state_actions([first, second], {})["cache_id"]
        == "newest"
    )


def test_statewise_cache_match_rejects_mixed_bypass_and_policy_history():
    candidates = [
        _cached_candidate("bypass", []),
        _cached_candidate(
            "policy",
            [_cached_step("dec", "query", {"dec_action": "skip"})],
        ),
    ]

    with pytest.raises(RuntimeError, match="mixed bypass"):
        select_cached_trajectory_by_state_actions(candidates, {})


def test_statewise_replay_ignores_late_events_and_rebinds_episode_identity(
    tmp_path, monkeypatch
):
    state = {
        "request_type": "enum",
        "semantic_id": "plan",
        "pid": 17,
        "run_id": 3,
        "round": 0,
    }
    step = {
        "phase": "enum",
        "state_hash": content_hash(stable_state(state)),
        "state": state,
        "action": {"enum_action": "native", "enum_k": 1},
    }
    candidate = {
        "cache_id": "cached",
        "trajectory": [step],
        "trajectory_hash": semantic_trajectory_hash([step]),
    }
    policy_log = tmp_path / "policy.jsonl"
    policy_log.write_text("", encoding="utf-8")

    def fake_run(*_args, input, **_kwargs):
        replay_state = json.loads(input)[0]
        replay_pid = replay_state["pid"]
        assert replay_pid < 0
        events = [
            {
                "phase": "policy_decision",
                "state": {**replay_state, "pid": 999},
                "action": {
                    "enum_action": "top5",
                    "enum_k": 5,
                    "inference_mode": "stochastic",
                },
            },
            {
                "phase": "policy_decision",
                "state": replay_state,
                "action": {
                    "enum_action": "native",
                    "enum_k": 1,
                    "inference_mode": "stochastic",
                    "log_probability": -0.25,
                },
            },
        ]
        policy_log.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "benchmarking.execution_cache.subprocess.run",
        fake_run,
    )

    selected, events, _offset, state_count = replay_cached_trajectories_statewise(
        container="unused",
        server_url="http://unused",
        policy_log=policy_log,
        policy_offset=0,
        candidates=[candidate],
    )

    assert selected["cache_id"] == "cached"
    assert state_count == 1
    assert len(events) == 1
    assert events[0]["state"] == state
    assert events[0]["action"]["inference_mode"] == "stochastic"
    assert events[0]["action"]["log_probability"] == pytest.approx(-0.25)


def test_cache_hit_persists_sampled_policy_transition(tmp_path):
    state = {
        "request_type": "enum",
        "round": 0,
        "sql": "SELECT 1",
    }
    policy_events = [
        {
            "phase": "policy_decision",
            "state": state,
            "action": {
                "action": "enum",
                "enum_action": "top5",
                "enum_k": 5,
                "action_index": 1,
                "action_probability": 0.25,
                "log_probability": -1.386294,
                "predicted_value": -0.5,
                "action_mask": [True, True],
                "inference_mode": "stochastic",
                "policy_version": "policy-v1",
            },
        }
    ]
    db_events = [
        {
            "round": 0,
            "phase": "final",
            "decision_states": {"enum": state},
            "action": {"enum_action": "top5", "enum_k": 5},
            "timing_ms": {"total": 12.5},
        }
    ]
    with ExperienceStore(tmp_path / "experience.sqlite") as store:
        counts = ingest_benchmark_trajectory(
            episode_id="episode-1",
            environment_hash="environment",
            implementation_version="implementation",
            db_events=db_events,
            policy_events=policy_events,
            timeout_limit_ms=60_000,
            timeout_charged_ms=60_000.0,
            episode_status="ok",
            cache_hit=True,
        )

        assert counts["rounds"] == 1
        assert counts["decisions"] == 1
        decision = counts["trajectory"][0]
        assert decision["policy"]["policy_version"] == "policy-v1"
        assert decision["policy"]["action_probability"] == pytest.approx(0.25)
        assert decision["policy"]["log_probability"] == pytest.approx(-1.386294)
        assert decision["policy"]["predicted_value"] == pytest.approx(-0.5)
        assert decision["charged_runtime_ms"] == pytest.approx(12.5)
        assert decision["runtime_source"] == "cache"
