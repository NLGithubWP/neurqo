"""Opt-in buffer replay against an already running release container, no SQL collection."""

import csv
import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = os.environ.get("NEURQO_REPLAY_TEST_CONTAINER")
pytestmark = pytest.mark.skipif(not CONTAINER, reason="requires a release container")


@pytest.mark.parametrize(
    "dataset,protocol,fold,query,method",
    [
        ("job", "random", "a", "10c", "neurqo"),
        ("job", "base_query", "c", "29a", "neurqo"),
        ("stack", "random", "a", "q11_0ea8bacd", "neurqo"),
        ("tpch", "random", "a", "3", "neurqo"),
        ("job", "", "", "29a", "query_split"),
    ],
)
def test_released_labels_are_reused(tmp_path, dataset, protocol, fold, query, method):
    buffer = ROOT / "results/buffers" / f"{dataset}_light.sql"
    before = hashlib.sha256(buffer.read_bytes()).hexdigest()
    output = tmp_path / "replayed.csv"
    command = [
        sys.executable,
        "scripts/reproduce/docker_replay.py",
        "--container",
        CONTAINER,
        "run",
        "--dataset",
        dataset,
        "--method",
        method,
        "--query-id",
        query,
        "--output",
        str(output),
    ]
    if protocol:
        command.extend(("--protocol", protocol, "--fold", fold))
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"executed": 0' in result.stdout
    assert hashlib.sha256(buffer.read_bytes()).hexdigest() == before
    with output.open(newline="") as handle:
        replayed = [
            row for row in csv.DictReader(handle) if row["method"] != "PostgreSQL"
        ]
    assert len(replayed) == 1
    row = replayed[0]
    labels = {"NeurQO", "NQO"} if method == "neurqo" else {"QuerySplit"}
    with (ROOT / "results/benchmark/nqo/nqo_runs.csv").open(newline="") as handle:
        original = next(
            item
            for item in csv.DictReader(handle)
            if item["dataset"].lower() == dataset
            and Path(item["sql_path"]).stem == query
            and item["protocol"] == protocol
            and item["fold"] == fold
            and item["method"] in labels
        )
    assert row["runtime_ms"] == original["runtime_ms"]
    assert row["cache_id"] == original["cache_id"]
    assert row["status"] == original["status"]


@pytest.mark.parametrize("experiment", ["ablation", "transfer"])
def test_other_producers_use_the_same_release_transport(tmp_path, experiment):
    env = dict(os.environ)
    env.update(
        NEURQO_REPLAY_CONTAINER=CONTAINER,
        NEURQO_REPLAY_ROOT=str(tmp_path),
        NEURQO_REPLAY_REMOTE_ROOT=f"/opt/neurqo/runtime/replay/test-{uuid.uuid4().hex}",
        PYTHONPATH=f"{ROOT}/src:{ROOT}",
    )
    code = """
import os, sys
from pathlib import Path
from scripts.reproduce.neurqo import run_abl_rl, run_transfer
common = dict(fold='a', device='cpu', port=19555, container=os.environ['NEURQO_REPLAY_CONTAINER'],
              runtime_root=Path(os.environ['NEURQO_REPLAY_ROOT']), cache_miss='error',
              sql_execution_lock=Path(os.environ['NEURQO_REPLAY_ROOT'])/'sql.lock',
              sql_execution_slots=1, host='127.0.0.1', pg_port=5432, user='neurdb',
              server_action_host='127.0.0.1', database_container=os.environ['NEURQO_REPLAY_CONTAINER'])
if sys.argv[1] == 'ablation':
    module = run_abl_rl
    task = module.Task(family='action', variant='no_enum', method='w/o Enum', workload='tpch', **common)
else:
    module = run_transfer
    task = module.Task(index=0, experiment='zero-shot', source='job', target='tpch', **common)
module.split_folds = lambda *a: {'random_a': {'test': ['3']}}
result = module.evaluate_task(task)
assert result['physical_executions'] == 0, result
assert not result['cache_misses'], result
assert result['query_count'] == 1, result
print('one checkpoint/query replay passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, experiment],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
