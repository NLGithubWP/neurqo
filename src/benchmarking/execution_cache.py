"""Semantic execution-cache lookup and policy replay."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from experience.store import (
    ExperienceStore,
    canonical_json,
    semantic_trajectory_hash,
)
from optimization.actions import model_input_hash, semantic_policy_trajectory
from optimization.action_vocabulary import (
    canonical_phase,
    normalize_policy_action,
    normalize_policy_state,
)


def read_jsonl_since(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], offset
    events = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for raw in handle:
            if raw.strip():
                event = json.loads(raw.decode("utf-8"))
                if isinstance(event, dict):
                    events.append(event)
        return events, handle.tell()


def replay_cached_trajectory(
    *,
    container: str,
    server_url: str,
    policy_log: Path,
    policy_offset: int,
    trajectory: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]], int]:
    """Replay cached states and require an exact semantic Action sequence."""
    # An empty trajectory means this SQL shape bypasses the NQO hook. The
    # bypass is deterministic for the original SQL, so there is no model
    # decision to replay before reusing its execution label.
    if not trajectory:
        return True, [], policy_offset

    replay_script = (
        "import json,sys,urllib.request;"
        "states=json.load(sys.stdin);"
        "url=sys.argv[1];"
        "[(lambda r:r.read())(urllib.request.urlopen("
        "urllib.request.Request(url,data=json.dumps(s).encode(),"
        "headers={'Content-Type':'application/json'},method='POST'),"
        "timeout=30.0)) for s in states]"
    )
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "python3",
            "-c",
            replay_script,
            server_url,
        ],
        input=canonical_json([item["state"] for item in trajectory]),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    replay_events, offset = read_jsonl_since(policy_log, policy_offset)
    decisions = semantic_policy_trajectory(replay_events)
    if len(decisions) != len(trajectory):
        return False, [], offset
    for expected, actual in zip(trajectory, decisions):
        expected_phase = canonical_phase(expected.get("phase"))
        expected_action = normalize_policy_action(
            expected.get("action") or {}, phase=expected_phase
        )
        if (
            actual["phase"] != expected_phase
            or actual["action"] != expected_action
        ):
            return False, [], offset
    return True, replay_events, offset


def cached_trajectory_state_key(item: dict[str, Any]) -> tuple[str, str]:
    """Identify one semantic policy state independently of its sampled Action."""
    return canonical_phase(item["phase"]), str(item["state_hash"])


def cached_trajectory_model_input_key(
    item: dict[str, Any],
) -> tuple[str, str]:
    """Identify the complete policy input, including dynamic model context."""
    return canonical_phase(item["phase"]), model_input_hash(
        normalize_policy_state(item["state"])
    )


def replay_cached_state_batch(
    *,
    container: str,
    server_url: str,
    policy_log: Path,
    policy_offset: int,
    states: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    int,
]:
    """Infer one Action for each distinct complete cached model input."""
    if not states:
        return {}, {}, policy_offset
    replay_script = (
        "import json,sys,urllib.request;"
        "states=json.load(sys.stdin);"
        "url=sys.argv[1];"
        "[(lambda r:r.read())(urllib.request.urlopen("
        "urllib.request.Request(url,data=json.dumps(s).encode(),"
        "headers={'Content-Type':'application/json'},method='POST'),"
        "timeout=30.0)) for s in states]"
    )
    replay_pid = -(
        os.getpid() * 1_000_000_000 + time.monotonic_ns() % 1_000_000_000
    )
    replay_states = []
    for item in states:
        replay_state = dict(item["state"])
        replay_state["pid"] = replay_pid
        replay_states.append(replay_state)
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "python3",
            "-c",
            replay_script,
            server_url,
        ],
        input=canonical_json(replay_states),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    replay_events, offset = read_jsonl_since(policy_log, policy_offset)
    decision_events = [
        event
        for event in replay_events
        if event.get("phase") == "policy_decision"
        and isinstance(event.get("state"), dict)
        and event["state"].get("pid") == replay_pid
    ]
    decisions = semantic_policy_trajectory(decision_events)
    if len(decisions) != len(states):
        raise RuntimeError(
            "prefix cache replay returned an unexpected decision count: "
            f"expected {len(states)}, got {len(decisions)}"
        )

    predicted_actions: dict[tuple[str, str], dict[str, Any]] = {}
    event_by_state: dict[tuple[str, str], dict[str, Any]] = {}
    for expected, actual, event in zip(states, decisions, decision_events):
        expected_key = cached_trajectory_model_input_key(expected)
        actual_key = cached_trajectory_model_input_key(actual)
        if actual_key != expected_key:
            raise RuntimeError(
                "prefix cache replay changed model-input identity: "
                f"expected {expected_key}, got {actual_key}"
            )
        predicted_actions[expected_key] = actual["action"]
        event_by_state[expected_key] = event
    return predicted_actions, event_by_state, offset


def replay_cached_trajectories_prefix(
    *,
    container: str,
    server_url: str,
    policy_log: Path,
    policy_offset: int,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int, int]:
    """Traverse immutable complete trajectories one matching prefix at a time.

    Each candidate remains one physical execution from root to terminal. At a
    decision depth, the checkpoint is evaluated only on states whose preceding
    Actions have already matched. No state, transition, or runtime is composed
    from a different trajectory. If several complete physical trajectories are
    compatible, the earliest stored observation is selected deterministically,
    so later cache growth cannot change an existing match.
    """
    if not candidates:
        return None, [], policy_offset, 0
    empty = [candidate for candidate in candidates if not candidate["trajectory"]]
    if empty:
        if len(empty) != len(candidates):
            raise RuntimeError(
                "mixed bypass and policy trajectories for one SQL cache key"
            )
        selected = min(
            empty,
            key=lambda item: (int(item["created_at_ms"]), str(item["cache_id"])),
        )
        return selected, [], policy_offset, 0

    active: list[tuple[dict[str, Any], int]] = [
        (candidate, 0) for candidate in candidates
    ]
    complete: list[dict[str, Any]] = []
    predicted_actions: dict[tuple[str, str], dict[str, Any]] = {}
    event_by_state: dict[tuple[str, str], dict[str, Any]] = {}

    while active:
        representatives: dict[tuple[str, str], dict[str, Any]] = {}
        for candidate, position in active:
            item = candidate["trajectory"][position]
            key = cached_trajectory_model_input_key(item)
            if key not in predicted_actions:
                representatives.setdefault(key, item)
        if representatives:
            new_actions, new_events, policy_offset = replay_cached_state_batch(
                container=container,
                server_url=server_url,
                policy_log=policy_log,
                policy_offset=policy_offset,
                states=list(representatives.values()),
            )
            predicted_actions.update(new_actions)
            event_by_state.update(new_events)

        next_active: list[tuple[dict[str, Any], int]] = []
        for candidate, position in active:
            item = candidate["trajectory"][position]
            key = cached_trajectory_model_input_key(item)
            if predicted_actions[key] != item["action"]:
                continue
            next_position = position + 1
            if next_position == len(candidate["trajectory"]):
                complete.append(candidate)
            else:
                next_active.append((candidate, next_position))
        active = next_active

    if not complete:
        return None, [], policy_offset, len(predicted_actions)
    selected = min(
        complete,
        key=lambda item: (int(item["created_at_ms"]), str(item["cache_id"])),
    )
    selected_events = []
    for item in selected["trajectory"]:
        event = dict(event_by_state[cached_trajectory_model_input_key(item)])
        event["state"] = item["state"]
        selected_events.append(event)
    selected_semantic = semantic_policy_trajectory(selected_events)
    if (
        semantic_trajectory_hash(selected_semantic)
        != selected.get(
            "canonical_trajectory_hash", selected["trajectory_hash"]
        )
    ):
        raise RuntimeError(
            "prefix cache matcher failed complete trajectory validation"
        )
    return selected, selected_events, policy_offset, len(predicted_actions)


def unique_cached_trajectory_states(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one representative for every graph/plan state in cache order."""
    representatives: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        for item in candidate["trajectory"]:
            key = cached_trajectory_state_key(item)
            representatives.setdefault(key, item)
    return list(representatives.values())


def select_cached_trajectory_by_state_actions(
    candidates: list[dict[str, Any]],
    predicted_actions: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the newest complete trajectory matching state-local Actions.

    Candidates are already ordered newest-first by ExperienceStore.  A result
    is reusable only when every graph/plan state in one *complete observed*
    trajectory maps to the Action sampled for that exact semantic state.  This
    deliberately does not compose transitions from different trajectories.
    """
    if not candidates:
        return None
    empty = [candidate for candidate in candidates if not candidate["trajectory"]]
    if empty:
        if len(empty) != len(candidates):
            raise RuntimeError(
                "mixed bypass and policy trajectories for one SQL cache key"
            )
        return empty[0]
    for candidate in candidates:
        if all(
            predicted_actions.get(cached_trajectory_state_key(item))
            == item["action"]
            for item in candidate["trajectory"]
        ):
            return candidate
    return None


def replay_cached_trajectories_statewise(
    *,
    container: str,
    server_url: str,
    policy_log: Path,
    policy_offset: int,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int, int]:
    """Match cache candidates with one policy sample per semantic state.

    The legacy matcher replayed every state of every complete candidate until
    one matched.  STACK candidates share most query-graph and plan-tree states,
    so that made cache lookup linear in both candidate count and trajectory
    length.  Here all unique state hashes are evaluated in one container call,
    then complete trajectories are filtered in memory by their state-local
    Actions.  The selected runtime still belongs to one exact physical episode.
    """
    if not candidates:
        return None, [], policy_offset, 0
    states = unique_cached_trajectory_states(candidates)
    if not states:
        selected = select_cached_trajectory_by_state_actions(candidates, {})
        return selected, [], policy_offset, 0

    replay_script = (
        "import json,sys,urllib.request;"
        "states=json.load(sys.stdin);"
        "url=sys.argv[1];"
        "[(lambda r:r.read())(urllib.request.urlopen("
        "urllib.request.Request(url,data=json.dumps(s).encode(),"
        "headers={'Content-Type':'application/json'},method='POST'),"
        "timeout=30.0)) for s in states]"
    )
    # A canceled physical STACK query can finish logging policy decisions
    # after its client has moved on to cache replay. Mark synthetic requests
    # through the ephemeral PID field so late decisions cannot contaminate
    # this batch. ``stable_state`` excludes PID, so cache identity is unchanged.
    replay_pid = -(
        os.getpid() * 1_000_000_000 + time.monotonic_ns() % 1_000_000_000
    )
    replay_states = []
    for item in states:
        replay_state = dict(item["state"])
        replay_state["pid"] = replay_pid
        replay_states.append(replay_state)
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "python3",
            "-c",
            replay_script,
            server_url,
        ],
        input=canonical_json(replay_states),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    replay_events, offset = read_jsonl_since(policy_log, policy_offset)
    decision_events = [
        event
        for event in replay_events
        if event.get("phase") == "policy_decision"
        and isinstance(event.get("state"), dict)
        and event["state"].get("pid") == replay_pid
    ]
    decisions = semantic_policy_trajectory(decision_events)
    if len(decisions) != len(states):
        raise RuntimeError(
            "statewise cache replay returned an unexpected decision count: "
            f"expected {len(states)}, got {len(decisions)}"
        )

    predicted_actions: dict[tuple[str, str], dict[str, Any]] = {}
    event_by_state: dict[tuple[str, str], dict[str, Any]] = {}
    for expected, actual, event in zip(states, decisions, decision_events):
        expected_key = cached_trajectory_state_key(expected)
        actual_key = cached_trajectory_state_key(actual)
        if actual_key != expected_key:
            raise RuntimeError(
                "statewise cache replay changed semantic state identity: "
                f"expected {expected_key}, got {actual_key}"
            )
        predicted_actions[expected_key] = actual["action"]
        event_by_state[expected_key] = event

    selected = select_cached_trajectory_by_state_actions(
        candidates,
        predicted_actions,
    )
    if selected is None:
        return None, [], offset, len(states)
    selected_events = []
    for item in selected["trajectory"]:
        # Preserve the new sample's probability/log-probability/value, but
        # restore the selected physical episode's PID/run/round identity.
        # Experience ingestion joins policy and DB events by that identity.
        event = dict(event_by_state[cached_trajectory_state_key(item)])
        event["state"] = item["state"]
        selected_events.append(event)
    selected_semantic = semantic_policy_trajectory(selected_events)
    if (
        semantic_trajectory_hash(selected_semantic)
        != selected.get(
            "canonical_trajectory_hash", selected["trajectory_hash"]
        )
    ):
        raise RuntimeError(
            "statewise cache matcher failed complete trajectory validation"
        )
    return selected, selected_events, offset, len(states)


def lookup_cached_execution(
    *,
    store: ExperienceStore,
    sql_hash: str,
    expected_result_hash: str | None,
    container: str,
    server_url: str,
    policy_log: Path,
    policy_offset: int,
    match_mode: str = "statewise",
    pinned_cache_id: str | None = None,
    pinned_query_wall_ms: float | None = None,
    pinned_status: str | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Find an exact complete trajectory cache hit for the current policy."""
    candidates = []
    for cached in store.trajectory_cache_candidates(
        sql_hash=sql_hash,
    ):
        if pinned_cache_id is not None and cached["cache_id"] != pinned_cache_id:
            continue
        if (
            pinned_cache_id is None
            and pinned_query_wall_ms is not None
            and abs(float(cached["query_wall_ms"]) - pinned_query_wall_ms) > 1e-6
        ):
            continue
        if pinned_status is not None and str(cached["status"]) != pinned_status:
            continue
        if (
            cached["status"] == "ok"
            and expected_result_hash is not None
            and cached["result_hash"] != expected_result_hash
        ):
            continue
        candidates.append(cached)

    selected = None
    replay_events: list[dict[str, Any]] = []
    state_count = 0
    if match_mode == "prefix":
        selected, replay_events, policy_offset, state_count = (
            replay_cached_trajectories_prefix(
                container=container,
                server_url=server_url,
                policy_log=policy_log,
                policy_offset=policy_offset,
                candidates=candidates,
            )
        )
    elif match_mode == "statewise":
        selected, replay_events, policy_offset, state_count = (
            replay_cached_trajectories_statewise(
                container=container,
                server_url=server_url,
                policy_log=policy_log,
                policy_offset=policy_offset,
                candidates=candidates,
            )
        )
    elif match_mode == "legacy":
        for cached in candidates:
            matched, replay_events, policy_offset = replay_cached_trajectory(
                container=container,
                server_url=server_url,
                policy_log=policy_log,
                policy_offset=policy_offset,
                trajectory=cached["trajectory"],
            )
            if matched:
                selected = cached
                break
    else:
        raise ValueError(f"unsupported cache match mode: {match_mode}")

    print(
        f"[cache-{match_mode}] candidates={len(candidates)} "
        f"states={state_count} matched={'yes' if selected else 'no'}",
        flush=True,
    )
    if selected is None:
        return None, policy_offset
    return (
        {
            "cache_source": str(selected["cache_id"]),
            "trajectory_hash": str(
                selected.get(
                    "canonical_trajectory_hash", selected["trajectory_hash"]
                )
            ),
            "cache_saved_wall_ms": float(selected["query_wall_ms"]),
            "policy_events": replay_events,
            "db_events": list(selected["db_events"]),
            "execution": {
                "status": selected["status"],
                "error": None,
                "client_wall_ms": float(selected["query_wall_ms"]),
                "charged_wall_ms": float(selected["charged_wall_ms"]),
                "result_hash": selected["result_hash"],
                "result_rows": selected["result_rows"],
            },
        },
        policy_offset,
    )
