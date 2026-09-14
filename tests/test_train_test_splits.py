from __future__ import annotations

from pathlib import Path

from workloads.train_test import SPLITS


ROOT = Path(__file__).resolve().parents[1]
QUERY_DIRECTORIES = {
    "JOB": ROOT / "workloads" / "query_job",
    "STACK": ROOT / "workloads" / "query_stack",
    "TPCH": ROOT / "workloads" / "query_tpch",
}
REQUIRED_PROTOCOLS = {
    "JOB": ("base_query", "leave_one_out", "random"),
    "STACK": ("base_query", "leave_one_out", "random"),
    "TPCH": ("random",),
}


def _query_ids(workload: str) -> set[str]:
    query_directory = QUERY_DIRECTORIES[workload]
    return {path.stem for path in query_directory.glob("*.sql")}


def test_splits_match_repository_queries() -> None:
    """Every split must partition the SQL files shipped for its workload."""
    assert set(SPLITS) == set(QUERY_DIRECTORIES)

    for workload, query_directory in QUERY_DIRECTORIES.items():
        query_ids = _query_ids(workload)
        assert query_ids, f"no SQL files found in {query_directory}"

        for protocol in REQUIRED_PROTOCOLS[workload]:
            for fold in "abc":
                split_name = f"{protocol}_{fold}"
                assert split_name in SPLITS[workload]

        for split_name, split in SPLITS[workload].items():
            train = split["train"]
            test = split["test"]
            train_ids = set(train)
            test_ids = set(test)

            assert len(train) == len(train_ids), (
                f"{workload} {split_name} contains duplicate training queries"
            )
            assert len(test) == len(test_ids), (
                f"{workload} {split_name} contains duplicate test queries"
            )
            assert train_ids.isdisjoint(test_ids), (
                f"{workload} {split_name} has train/test overlap"
            )
            assert train_ids | test_ids == query_ids, (
                f"{workload} {split_name} does not cover exactly the released SQL"
            )


def test_random_test_folds_partition_each_workload() -> None:
    """The three random test folds must be disjoint and cover the workload."""
    for workload in QUERY_DIRECTORIES:
        query_ids = _query_ids(workload)
        test_folds = [
            set(SPLITS[workload][f"random_{fold}"]["test"])
            for fold in "abc"
        ]

        assert test_folds[0].isdisjoint(test_folds[1])
        assert test_folds[0].isdisjoint(test_folds[2])
        assert test_folds[1].isdisjoint(test_folds[2])
        assert set().union(*test_folds) == query_ids
