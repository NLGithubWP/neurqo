#!/usr/bin/env python3
"""Reproduce GenJoin on TPC-H with a documented single-run adaptation.

The released GenJoin repository supports only JOB and STACK and omits its
model-output decoder. This adapter preserves the released CVAE architecture,
query/plan encoding, random hint generation, and fold-isolated training while
adding:

* recursive TPC-H query-block parsing;
* a schema graph derived from the 22 TPC-H statements;
* one execution per candidate hint set;
* runtime-gap pairing instead of the original repeated-run t-test; and
* resumable collection, training, and evaluation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mo_sql_parsing import parse
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
REVISION_DIR = Path(__file__).resolve().parent
GENJOIN_ROOT = ROOT / "thrid_party" / "genjoin" / "GenJoin_Repository"
OUTPUT_DIR: Path
POSTGRES_CSV: Path

sys.path.insert(0, str(REVISION_DIR))
sys.path.insert(0, str(GENJOIN_ROOT))

from benchmarking.workloads import (  # noqa: E402
    DEFAULT_BASELINE_WORK_ROOT,
    CsvResultStore,
    PhaseTiming,
    connect,
    dynamic_timeout_s,
    execute_once,
    load_postgres_times,
    query_sql,
    reset_session,
    set_timeout,
    split_folds,
    utc_now,
)
from src.encoding.plan_decoder import model_output_to_hints  # noqa: E402
from src.models.cvae import CVAE_G  # noqa: E402


OUTPUT_DIR = DEFAULT_BASELINE_WORK_ROOT / "tpch" / "genjoin"
POSTGRES_CSV = DEFAULT_BASELINE_WORK_ROOT / "tpch" / "postgres.csv"


WORKLOAD = "TPCH"
PROTOCOL = "random"
JOIN_TYPES = ("HashJoin", "MergeJoin", "NestLoop")
DEFAULT_CANDIDATES = 200
DEFAULT_PAIR_GAP = 0.05
DEFAULT_MODELS = 1
DEFAULT_EPOCHS = 150
DEFAULT_SEED = 1234
NORM = 100

COMPARISON_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte"}
JOIN_CLAUSE_RE = re.compile(r"(?:^| )join$")
QUALIFIED_COLUMN_RE = re.compile(
    r'^"?([A-Za-z_][A-Za-z0-9_$]*)"?\."?([A-Za-z_][A-Za-z0-9_$]*)"?$'
)

COLLECTION_FIELDS = (
    "result_key",
    "query_id",
    "candidate_id",
    "seed",
    "valid_edge_count",
    "plan_encoding",
    "hints",
    "timeout_s",
    "wall_s",
    "execution_s",
    "planning_s",
    "charged_s",
    "error",
    "recorded_at_utc",
)

RAW_RESULT_FIELDS = (
    "result_key",
    "fold",
    "query_id",
    "run_id",
    "supported",
    "used_fallback",
    "hints",
    "encoding_s",
    "model_inference_s",
    "inference_s",
    "timeout_s",
    "db_runtime_s",
    "charged_db_s",
    "method_runtime_s",
    "error",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def clean_sql(query_id: str) -> str:
    sql = query_sql(WORKLOAD, query_id)
    sql = re.sub(r"^\s*--.*$", "", sql, flags=re.MULTILINE)
    return sql.strip().rstrip(";")


def sql_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def positional_encoding(position: np.ndarray | float | int, norm: int = NORM):
    const = 1e-4
    return np.arctan(norm / (position + const)) + 1 - np.arctan(1 / const)


def _is_query(value: object) -> bool:
    return isinstance(value, dict) and "select" in value and "from" in value


def _qualified_column(value: object) -> Optional[Tuple[str, str]]:
    if not isinstance(value, str):
        return None
    match = QUALIFIED_COLUMN_RE.fullmatch(value.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _iter_ctes(value: object) -> Iterator[Tuple[str, dict]]:
    items = value if isinstance(value, list) else [value]
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        query = item.get("value")
        if isinstance(name, str) and _is_query(query):
            yield name, query


def _nested_queries(value: object) -> Iterator[dict]:
    if _is_query(value):
        yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _nested_queries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_queries(child)


def _collect_local_joins(
    value: object,
    aliases: Mapping[str, Optional[str]],
) -> List[dict]:
    joins: List[dict] = []

    def visit(node: object) -> None:
        if _is_query(node):
            return
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return

        for operator, operands in node.items():
            if operator in COMPARISON_OPERATORS and isinstance(operands, list):
                if len(operands) >= 2:
                    left = _qualified_column(operands[0])
                    right = _qualified_column(operands[1])
                    if left and right and left[0] != right[0]:
                        if aliases.get(left[0]) and aliases.get(right[0]):
                            joins.append(
                                {
                                    "operator": operator,
                                    "left_alias": left[0],
                                    "left_column": left[1],
                                    "right_alias": right[0],
                                    "right_column": right[1],
                                }
                            )
            visit(operands)

    visit(value)
    return joins


def parse_query_blocks(sql: str) -> List[dict]:
    parsed = parse(sql)
    blocks: List[dict] = []
    visited: set[int] = set()

    def process_query(block: dict, inherited_ctes: set[str]) -> None:
        identity = id(block)
        if identity in visited:
            return
        visited.add(identity)

        block_id = len(blocks)
        spec = {
            "block_id": block_id,
            "aliases": {},
            "joins": [],
        }
        blocks.append(spec)

        local_ctes = set(inherited_ctes)
        for cte_name, cte_query in _iter_ctes(block.get("with", [])):
            local_ctes.add(cte_name)
            process_query(cte_query, local_ctes)

        aliases: Dict[str, Optional[str]] = {}
        on_expressions: List[object] = []

        def register_relation(relation: object) -> None:
            if isinstance(relation, str):
                aliases[relation] = None if relation in local_ctes else relation
                return
            if not isinstance(relation, dict):
                return

            join_keys = [
                key
                for key in relation
                if isinstance(key, str) and JOIN_CLAUSE_RE.search(key)
            ]
            if join_keys:
                for key in join_keys:
                    register_relation(relation[key])
                if "on" in relation:
                    on_expressions.append(relation["on"])
                return

            value = relation.get("value")
            alias = relation.get("name")
            if _is_query(value):
                process_query(value, local_ctes)
                if isinstance(alias, str):
                    aliases[alias] = None
            elif isinstance(value, str):
                resolved_alias = alias if isinstance(alias, str) else value
                aliases[resolved_alias] = None if value in local_ctes else value

        from_clause = block.get("from", [])
        for relation in from_clause if isinstance(from_clause, list) else [from_clause]:
            register_relation(relation)

        raw_joins: List[dict] = []
        if "where" in block:
            raw_joins.extend(_collect_local_joins(block["where"], aliases))
        for expression in on_expressions:
            raw_joins.extend(_collect_local_joins(expression, aliases))

        deduplicated: Dict[Tuple[str, str], dict] = {}
        for join in raw_joins:
            left_alias = join["left_alias"]
            right_alias = join["right_alias"]
            pair = tuple(sorted((left_alias, right_alias)))
            candidate = {
                **join,
                "left_table": aliases[left_alias],
                "right_table": aliases[right_alias],
            }
            previous = deduplicated.get(pair)
            candidate_key = (
                candidate["left_table"],
                candidate["left_column"],
                candidate["right_table"],
                candidate["right_column"],
            )
            if previous is None:
                deduplicated[pair] = candidate
            else:
                previous_key = (
                    previous["left_table"],
                    previous["left_column"],
                    previous["right_table"],
                    previous["right_column"],
                )
                if candidate_key < previous_key:
                    deduplicated[pair] = candidate

        spec["aliases"] = aliases
        spec["joins"] = sorted(
            deduplicated.values(),
            key=lambda item: (
                min(item["left_alias"], item["right_alias"]),
                max(item["left_alias"], item["right_alias"]),
            ),
        )

        for key, value in block.items():
            if key == "with":
                continue
            for nested in _nested_queries(value):
                if nested is not block:
                    process_query(nested, local_ctes)

    process_query(parsed, set())
    return blocks


def canonical_edge(join: Mapping[str, object]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    left = (str(join["left_table"]), str(join["left_column"]))
    right = (str(join["right_table"]), str(join["right_column"]))
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def explain_plan(cursor, sql: str) -> dict:
    cursor.execute("EXPLAIN (FORMAT JSON) " + sql)
    return cursor.fetchone()[0][0]["Plan"]


def scan_rows(plan: dict) -> Dict[str, Tuple[str, float]]:
    rows: Dict[str, Tuple[str, float]] = {}

    def visit(node: dict) -> None:
        relation = node.get("Relation Name")
        alias = node.get("Alias")
        if relation and alias:
            value = max(1.0, float(node.get("Plan Rows", 1.0)))
            previous = rows.get(alias)
            if previous is None or value > previous[1]:
                rows[alias] = (relation, value)
        for child in node.get("Plans", []):
            visit(child)

    visit(plan)
    return rows


def full_cardinalities(cursor) -> Dict[str, float]:
    cursor.execute(
        """
        SELECT c.relname, c.reltuples
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        """
    )
    return {name: max(1.0, float(rows)) for name, rows in cursor.fetchall()}


def alias_selectivities(
    spec: Mapping[str, object],
    scans: Mapping[str, Tuple[str, float]],
    cardinalities: Mapping[str, float],
) -> Dict[str, float]:
    aliases: Dict[str, str] = {}
    for block in spec["blocks"]:  # type: ignore[index]
        for alias, table in block["aliases"].items():
            if table:
                aliases[alias] = table

    result: Dict[str, float] = {}
    for alias, table in aliases.items():
        estimated_rows = scans.get(alias, (table, cardinalities.get(table, 1.0)))[1]
        full_rows = cardinalities.get(table, max(1.0, estimated_rows))
        estimated_rows = min(max(1.0, estimated_rows), max(1.0, full_rows))
        result[alias] = math.log(estimated_rows + 1.0) / math.log(full_rows + 1.0)
    return result


def query_encoding(
    spec: Mapping[str, object],
    num_edges: int,
    selectivities: Mapping[str, float],
) -> np.ndarray:
    encoding = np.zeros((num_edges, 3), dtype=np.float32)
    used: set[int] = set()
    for join in spec["joins"]:  # type: ignore[index]
        edge_idx = int(join["edge_idx"])
        if edge_idx in used:
            continue
        used.add(edge_idx)
        encoding[edge_idx, 0] = 1.0
        encoding[edge_idx, 1] = float(
            selectivities.get(str(join["encoding_left_alias"]), 1.0)
        )
        encoding[edge_idx, 2] = float(
            selectivities.get(str(join["encoding_right_alias"]), 1.0)
        )
    return encoding.reshape(-1)


def sample_plan_encoding(
    num_edges: int,
    valid_edges: Sequence[int],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((num_edges, len(JOIN_TYPES)), dtype=np.float32)
    valid_edges = sorted(set(int(edge) for edge in valid_edges))
    ranks = list(range(1, len(valid_edges) * len(JOIN_TYPES) + 1))
    remaining: List[Tuple[int, int]] = []

    for edge_idx in valid_edges:
        winner = int(rng.integers(0, len(JOIN_TYPES)))
        rank = ranks.pop(0)
        result[edge_idx, winner] = positional_encoding(rank)
        for join_type_idx in range(len(JOIN_TYPES)):
            if join_type_idx != winner:
                remaining.append((edge_idx, join_type_idx))

    rng.shuffle(ranks)
    for (edge_idx, join_type_idx), rank in zip(remaining, ranks):
        result[edge_idx, join_type_idx] = positional_encoding(rank)
    return result.reshape(-1)


class MetadataJoinGraph:
    def __init__(self, num_edges: int, valid_edges: Sequence[int]):
        self.num_undirected_edges = num_edges
        self.valid_edges = sorted(set(int(edge) for edge in valid_edges))

    def get_available_joins(self, remove_duplicates: bool = False) -> pd.DataFrame:
        del remove_duplicates
        return pd.DataFrame({"state_edge_idx": self.valid_edges})


def decoded_plan_encoding(
    logits: torch.Tensor,
    num_edges: int,
    valid_edges: Sequence[int],
) -> np.ndarray:
    graph = MetadataJoinGraph(num_edges, valid_edges)
    return model_output_to_hints(logits, graph, norm=NORM, method="argmax")


def hints_from_encoding(spec: Mapping[str, object], encoding: np.ndarray) -> str:
    hints: List[str] = []
    seen_pairs: set[Tuple[int, str, str]] = set()
    for join in spec["joins"]:  # type: ignore[index]
        block_id = int(join["block_id"])
        left_alias = str(join["left_alias"])
        right_alias = str(join["right_alias"])
        pair = (block_id, *sorted((left_alias, right_alias)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edge_idx = int(join["edge_idx"])
        values = encoding[
            len(JOIN_TYPES) * edge_idx : len(JOIN_TYPES) * (edge_idx + 1)
        ]
        join_type = JOIN_TYPES[int(np.argmax(values))]
        hints.append(f"{join_type}({left_alias} {right_alias})")
    return " ".join(hints)


def hinted_sql(sql: str, hints: str) -> str:
    return f"/*+ {hints} */ {sql}" if hints else sql


def load_metadata() -> dict:
    path = OUTPUT_DIR / "workload.json"
    if not path.is_file():
        raise RuntimeError("run the prepare stage first")
    return json.loads(path.read_text())


@contextmanager
def database_lock() -> Iterator[None]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / ".database.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another revision database experiment is running") from exc
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    query_specs: Dict[str, dict] = {}
    all_edges: set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()

    for query_id in map(str, range(1, 23)):
        sql = clean_sql(query_id)
        blocks = parse_query_blocks(sql)
        joins = []
        for block in blocks:
            for join in block["joins"]:
                edge = canonical_edge(join)
                all_edges.add(edge)
                left_endpoint = (join["left_table"], join["left_column"])
                if left_endpoint == edge[0]:
                    encoding_left_alias = join["left_alias"]
                    encoding_right_alias = join["right_alias"]
                else:
                    encoding_left_alias = join["right_alias"]
                    encoding_right_alias = join["left_alias"]
                joins.append(
                    {
                        **join,
                        "block_id": block["block_id"],
                        "edge": [list(edge[0]), list(edge[1])],
                        "encoding_left_alias": encoding_left_alias,
                        "encoding_right_alias": encoding_right_alias,
                    }
                )
        query_specs[query_id] = {
            "query_id": query_id,
            "sql_sha256": sql_hash(sql),
            "blocks": blocks,
            "joins": joins,
        }

    ordered_edges = sorted(all_edges)
    edge_to_idx = {edge: idx for idx, edge in enumerate(ordered_edges)}
    for spec in query_specs.values():
        for join in spec["joins"]:
            edge = (tuple(join["edge"][0]), tuple(join["edge"][1]))
            join["edge_idx"] = edge_to_idx[edge]
        spec["valid_edges"] = sorted({join["edge_idx"] for join in spec["joins"]})
        spec["supported"] = bool(spec["valid_edges"])

    with database_lock():
        connection = connect(WORKLOAD)
        cursor = connection.cursor()
        try:
            reset_session(cursor, load_pg_hint_plan=True)
            cardinalities = full_cardinalities(cursor)
            for query_id, spec in query_specs.items():
                plan = explain_plan(cursor, clean_sql(query_id))
                scans = scan_rows(plan)
                selectivities = alias_selectivities(spec, scans, cardinalities)
                spec["prepared_selectivities"] = selectivities
                spec["query_encoding"] = query_encoding(
                    spec, len(ordered_edges), selectivities
                ).tolist()
        finally:
            cursor.close()
            connection.close()

    metadata = {
        "method": "GenJoin",
        "workload": WORKLOAD,
        "protocol": PROTOCOL,
        "adapter": "single-run TPC-H reproduction",
        "created_at_utc": utc_now(),
        "source_commit": "7c6236f5eb163b0c2caf07d7c91e07cbcc87ff98",
        "candidate_runs_per_hint_set": 1,
        "num_edges": len(ordered_edges),
        "edges": [
            {
                "edge_idx": idx,
                "left_table": edge[0][0],
                "left_column": edge[0][1],
                "right_table": edge[1][0],
                "right_column": edge[1][1],
            }
            for idx, edge in enumerate(ordered_edges)
        ],
        "queries": query_specs,
        "supported_queries": [
            query_id for query_id, spec in query_specs.items() if spec["supported"]
        ],
        "fallback_queries": [
            query_id for query_id, spec in query_specs.items() if not spec["supported"]
        ],
    }
    atomic_json(OUTPUT_DIR / "workload.json", metadata)
    print(
        json.dumps(
            {
                "num_edges": metadata["num_edges"],
                "supported_queries": metadata["supported_queries"],
                "fallback_queries": metadata["fallback_queries"],
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def plan_join_nodes(plan: dict) -> List[dict]:
    result: List[dict] = []

    def aliases(node: dict) -> List[str]:
        values: set[str] = set()

        def visit(child: dict) -> None:
            if child.get("Alias"):
                values.add(str(child["Alias"]))
            for grandchild in child.get("Plans", []):
                visit(grandchild)

        visit(node)
        return sorted(values)

    def visit(node: dict) -> None:
        if node.get("Node Type") in {"Hash Join", "Merge Join", "Nested Loop"}:
            result.append({"node_type": node["Node Type"], "aliases": aliases(node)})
        for child in node.get("Plans", []):
            visit(child)

    visit(plan)
    return result


def validate() -> dict:
    metadata = load_metadata()
    num_edges = int(metadata["num_edges"])
    records = []

    # Decoder direction and masking sanity check.
    valid = [0] if num_edges else []
    scores = torch.zeros((1, num_edges * len(JOIN_TYPES)), dtype=torch.float32)
    if valid:
        scores[0, 0:3] = torch.tensor([1.0, 3.0, 2.0])
        decoded = decoded_plan_encoding(scores, num_edges, valid)
        if int(np.argmax(decoded[:3])) != 1:
            raise RuntimeError("decoder ranking sanity check failed")

    with database_lock():
        connection = connect(WORKLOAD)
        cursor = connection.cursor()
        try:
            reset_session(cursor, load_pg_hint_plan=True)
            set_timeout(cursor, 30.0)
            for query_id, spec in metadata["queries"].items():
                if not spec["supported"]:
                    records.append(
                        {
                            "query_id": query_id,
                            "supported": False,
                            "hints": "",
                            "status": "postgres_fallback_no_pairwise_join",
                        }
                    )
                    continue
                encoding = sample_plan_encoding(
                    num_edges,
                    spec["valid_edges"],
                    DEFAULT_SEED + int(query_id),
                )
                hints = hints_from_encoding(spec, encoding)
                try:
                    plan = explain_plan(
                        cursor, hinted_sql(clean_sql(query_id), hints)
                    )
                    records.append(
                        {
                            "query_id": query_id,
                            "supported": True,
                            "hints": hints,
                            "status": "ok",
                            "plan_join_nodes": plan_join_nodes(plan),
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "query_id": query_id,
                            "supported": True,
                            "hints": hints,
                            "status": "error",
                            "error": str(exc).replace("\n", " ")[:1000],
                        }
                    )
                    connection.rollback()
                    reset_session(cursor, load_pg_hint_plan=True)
        finally:
            cursor.close()
            connection.close()

    errors = [record for record in records if record["status"] == "error"]
    result = {
        "checked_at_utc": utc_now(),
        "query_count": len(records),
        "supported_count": sum(bool(record["supported"]) for record in records),
        "errors": len(errors),
        "records": records,
    }
    atomic_json(OUTPUT_DIR / "validation.json", result)
    print(
        json.dumps(
            {
                "query_count": result["query_count"],
                "supported_count": result["supported_count"],
                "errors": result["errors"],
            },
            indent=2,
        ),
        flush=True,
    )
    if errors:
        raise RuntimeError(f"{len(errors)} TPC-H hint validation checks failed")
    return result


def collection_rows(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect(candidates: int = DEFAULT_CANDIDATES) -> None:
    metadata = load_metadata()
    num_edges = int(metadata["num_edges"])
    pg_times = load_postgres_times(
        POSTGRES_CSV,
        time_column="run1_charged_s",
    )
    path = OUTPUT_DIR / "collection.csv"
    store = CsvResultStore(path, COLLECTION_FIELDS)
    supported = [
        query_id
        for query_id, spec in metadata["queries"].items()
        if spec["supported"]
    ]
    total = len(supported) * candidates
    timing = PhaseTiming(
        OUTPUT_DIR / "collection_timing.json",
        method="GenJoin",
        workload=WORKLOAD,
        phase="single_run_demonstration_collection",
        protocol=PROTOCOL,
        preexisting_records=len(collection_rows(path)),
        database_execution_s_baseline=0.0,
    )
    started = time.perf_counter()
    completed = 0

    with database_lock():
        connection = connect(WORKLOAD)
        cursor = connection.cursor()
        try:
            reset_session(cursor, load_pg_hint_plan=True)
            for query_id in supported:
                spec = metadata["queries"][query_id]
                timeout_s = dynamic_timeout_s(pg_times[query_id])
                sql = clean_sql(query_id)
                for candidate_id in range(candidates):
                    result_key = f"{query_id}:{candidate_id}"
                    previous = store.get(result_key)
                    if previous is not None:
                        completed += 1
                        timing.record(resumed=True)
                        continue

                    seed = DEFAULT_SEED + int(query_id) * 10000 + candidate_id
                    encoding = sample_plan_encoding(
                        num_edges, spec["valid_edges"], seed
                    )
                    hints = hints_from_encoding(spec, encoding)
                    explained = (
                        f"/*+ {hints} */ EXPLAIN "
                        f"(ANALYZE, VERBOSE, FORMAT JSON) {sql}"
                    )
                    set_timeout(cursor, timeout_s)
                    wall_started = time.perf_counter()
                    execution_s: Optional[float] = None
                    planning_s: Optional[float] = None
                    error = ""
                    try:
                        cursor.execute(explained)
                        payload = cursor.fetchone()[0][0]
                        wall_s = time.perf_counter() - wall_started
                        execution_s = float(payload["Execution Time"]) / 1000.0
                        planning_s = float(payload["Planning Time"]) / 1000.0
                        charged_s = execution_s
                    except Exception as exc:
                        wall_s = time.perf_counter() - wall_started
                        charged_s = timeout_s
                        error = str(exc).replace("\n", " ")[:1000]
                        connection.rollback()
                        reset_session(cursor, load_pg_hint_plan=True)

                    store.append(
                        {
                            "result_key": result_key,
                            "query_id": query_id,
                            "candidate_id": candidate_id,
                            "seed": seed,
                            "valid_edge_count": len(spec["valid_edges"]),
                            "plan_encoding": json.dumps(
                                encoding.tolist(), separators=(",", ":")
                            ),
                            "hints": hints,
                            "timeout_s": timeout_s,
                            "wall_s": wall_s,
                            "execution_s": "" if execution_s is None else execution_s,
                            "planning_s": "" if planning_s is None else planning_s,
                            "charged_s": charged_s,
                            "error": error,
                            "recorded_at_utc": utc_now(),
                        }
                    )
                    timing.record(resumed=False, database_execution_s=wall_s)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.perf_counter() - started
                        rate = completed / elapsed if elapsed else 0.0
                        eta = (total - completed) / rate if rate else 0.0
                        print(
                            f"collection {completed}/{total} "
                            f"({100.0 * completed / total:.1f}%), "
                            f"last={query_id}:{candidate_id} "
                            f"charged={charged_s:.3f}s eta={eta / 3600:.2f}h",
                            flush=True,
                        )
            timing.finish("completed")
        except BaseException:
            timing.finish("interrupted")
            raise
        finally:
            cursor.close()
            connection.close()


def load_complete_collection(metadata: Mapping[str, object], candidates: int) -> Dict[str, List[dict]]:
    rows = collection_rows(OUTPUT_DIR / "collection.csv")
    by_query: Dict[str, List[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    for query_id, spec in metadata["queries"].items():  # type: ignore[index]
        if not spec["supported"]:
            continue
        query_rows = by_query.get(query_id, [])
        ids = {int(row["candidate_id"]) for row in query_rows}
        expected = set(range(candidates))
        if ids != expected:
            raise RuntimeError(
                f"collection for query {query_id} is incomplete: "
                f"{len(ids)}/{candidates}"
            )
        query_rows.sort(key=lambda row: int(row["candidate_id"]))
    return by_query


@dataclass(frozen=True)
class Pair:
    query_id: str
    slower_id: int
    faster_id: int
    confidence: float


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Pair],
        encodings: Mapping[str, np.ndarray],
        query_encodings: Mapping[str, np.ndarray],
    ):
        self.pairs = list(pairs)
        self.encodings = encodings
        self.query_encodings = query_encodings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return (
            torch.from_numpy(self.encodings[pair.query_id][pair.slower_id]),
            torch.from_numpy(self.query_encodings[pair.query_id]),
            torch.from_numpy(self.encodings[pair.query_id][pair.faster_id]),
            torch.tensor([pair.confidence], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )


def build_pairs(
    metadata: Mapping[str, object],
    by_query: Mapping[str, List[dict]],
    train_queries: Sequence[str],
    pg_times: Mapping[str, float],
    pair_gap: float,
    candidates: int,
    seed: int,
) -> Tuple[List[Pair], Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    pairs: List[Pair] = []
    encodings: Dict[str, np.ndarray] = {}
    query_encodings: Dict[str, np.ndarray] = {}
    per_query: Dict[str, int] = {}

    for query_id in train_queries:
        spec = metadata["queries"][query_id]  # type: ignore[index]
        if not spec["supported"]:
            continue
        rows = by_query[query_id]
        matrix = np.asarray(
            [json.loads(row["plan_encoding"]) for row in rows], dtype=np.float32
        )
        runtimes = np.asarray([float(row["charged_s"]) for row in rows])
        encodings[query_id] = matrix
        query_encodings[query_id] = np.asarray(
            spec["query_encoding"], dtype=np.float32
        )
        count_before = len(pairs)
        for left in range(candidates):
            for right in range(left + 1, candidates):
                if runtimes[left] >= runtimes[right]:
                    slower, faster = left, right
                else:
                    slower, faster = right, left
                if runtimes[slower] < runtimes[faster] * (1.0 + pair_gap):
                    continue
                confidence = (
                    0.0
                    if runtimes[faster] <= pg_times[query_id] * (1.0 - pair_gap)
                    else 1.0
                )
                pairs.append(Pair(query_id, slower, faster, confidence))
        per_query[query_id] = len(pairs) - count_before

    rng = random.Random(seed)
    rng.shuffle(pairs)
    retained = max(1, int(0.30 * len(pairs))) if pairs else 0
    pairs = pairs[:retained]
    return (
        pairs,
        encodings,
        query_encodings,
        {
            "pair_gap": pair_gap,
            "candidate_pairs_before_subsample": sum(per_query.values()),
            "retained_pairs": len(pairs),
            "per_query_pairs": per_query,
            "subsample_ratio": 0.30,
        },
    )


def genjoin_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    mask = target > 0
    masked_target = target.masked_fill(~mask, float("-inf"))
    target_distribution = F.softmax(masked_target, dim=1)
    reconstruction = -(target_distribution * F.log_softmax(prediction, dim=1)).sum(
        dim=1
    ).mean()
    std = torch.exp(0.5 * logvar)
    latent = torch.mean(
        torch.log(1.0 / std)
        + (std.square() + mu.square()) / 2.0
        - 0.5
    )
    return reconstruction + latent


def model_config(num_edges: int) -> dict:
    plan_size = num_edges * len(JOIN_TYPES)
    return {
        "plan_size": plan_size,
        "query_size": num_edges * 3,
        "latent_size": 50,
        "layers_size": [max(75, plan_size * 2), max(60, plan_size + 15)],
        "batch_size": 128,
        "learning_rate": 1e-3,
        "step_size": 25,
    }


def train(
    *,
    candidates: int = DEFAULT_CANDIDATES,
    pair_gap: float = DEFAULT_PAIR_GAP,
    n_models: int = DEFAULT_MODELS,
    epochs: int = DEFAULT_EPOCHS,
) -> None:
    metadata = load_metadata()
    by_query = load_complete_collection(metadata, candidates)
    pg_times = load_postgres_times(
        POSTGRES_CSV,
        time_column="run1_charged_s",
    )
    config = model_config(int(metadata["num_edges"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timing_records = []

    for fold, split in split_folds(WORKLOAD, PROTOCOL).items():
        pairs, encodings, query_encodings, pair_summary = build_pairs(
            metadata,
            by_query,
            split["train"],
            pg_times,
            pair_gap,
            candidates,
            DEFAULT_SEED + sum(ord(char) for char in fold),
        )
        if not pairs:
            raise RuntimeError(f"no single-run training pairs for {fold}")
        atomic_json(OUTPUT_DIR / "models" / fold / "pairs.json", pair_summary)
        dataset = PairDataset(pairs, encodings, query_encodings)

        for run_id in range(n_models):
            model_path = OUTPUT_DIR / "models" / fold / f"run{run_id}.pt"
            if model_path.is_file():
                print(f"training skip existing {model_path}", flush=True)
                continue

            seed = DEFAULT_SEED + run_id + 100 * sum(ord(char) for char in fold)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator()
            generator.manual_seed(seed)
            loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=0,
                generator=generator,
            )
            model = CVAE_G(
                config["plan_size"],
                config["latent_size"],
                config["layers_size"],
                config["query_size"],
            ).to(device)
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=config["learning_rate"]
            )
            scheduler = StepLR(
                optimizer, step_size=config["step_size"], gamma=0.75
            )
            started = time.perf_counter()
            final_loss = float("nan")

            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0
                batches = 0
                for input_enc, query_enc, output_enc, confidence, output_distance in loader:
                    input_enc = input_enc.to(device)
                    query_enc = query_enc.to(device)
                    output_enc = output_enc.to(device)
                    confidence = confidence.to(device)
                    output_distance = output_distance.to(device)
                    optimizer.zero_grad()
                    prediction, mu, logvar, _ = model(
                        input_enc,
                        query_enc,
                        confidence,
                        output_distance,
                    )
                    loss = genjoin_loss(prediction, output_enc, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.detach().cpu())
                    batches += 1
                scheduler.step()
                final_loss = running_loss / max(1, batches)
                if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                    print(
                        f"training {fold} run={run_id} epoch={epoch}/{epochs} "
                        f"loss={final_loss:.6f}",
                        flush=True,
                    )

            elapsed = time.perf_counter() - started
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.to("cpu").state_dict(),
                    "config": config,
                    "fold": fold,
                    "run_id": run_id,
                    "seed": seed,
                    "epochs": epochs,
                    "pair_gap": pair_gap,
                    "final_loss": final_loss,
                },
                model_path,
            )
            timing_records.append(
                {
                    "fold": fold,
                    "run_id": run_id,
                    "device": str(device),
                    "training_s": elapsed,
                    "pairs": len(dataset),
                    "epochs": epochs,
                    "final_loss": final_loss,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    existing = []
    timing_path = OUTPUT_DIR / "training_timing.json"
    if timing_path.is_file():
        existing = json.loads(timing_path.read_text()).get("records", [])
    keyed = {
        (record["fold"], int(record["run_id"])): record
        for record in [*existing, *timing_records]
    }
    atomic_json(
        timing_path,
        {
            "method": "GenJoin",
            "workload": WORKLOAD,
            "records": list(keyed.values()),
        },
    )


def load_model(path: Path) -> CVAE_G:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = CVAE_G(
        config["plan_size"],
        config["latent_size"],
        config["layers_size"],
        config["query_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def live_query_encoding(
    cursor,
    query_id: str,
    spec: Mapping[str, object],
    num_edges: int,
    cardinalities: Mapping[str, float],
) -> np.ndarray:
    plan = explain_plan(cursor, clean_sql(query_id))
    selectivities = alias_selectivities(spec, scan_rows(plan), cardinalities)
    return query_encoding(spec, num_edges, selectivities)


def summarize_results(
    metadata: Mapping[str, object],
    pg_times: Mapping[str, float],
    n_models: int,
) -> dict:
    raw_path = OUTPUT_DIR / "random_model_results.csv"
    with raw_path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    raw_errors = sum(bool(row["error"]) for row in raw)
    by_key = {
        (row["fold"], row["query_id"], int(row["run_id"])): row for row in raw
    }
    expected = [
        (fold, query_id, run_id)
        for fold, split in split_folds(WORKLOAD, PROTOCOL).items()
        for query_id in split["test"]
        for run_id in range(n_models)
    ]
    if set(by_key) != set(expected):
        raise RuntimeError("GenJoin TPC-H result coverage is incomplete")

    aggregate_rows = []
    for fold, split in split_folds(WORKLOAD, PROTOCOL).items():
        for query_id in split["test"]:
            rows = [by_key[(fold, query_id, run_id)] for run_id in range(n_models)]
            aggregate_rows.append(
                {
                    "fold": fold,
                    "query_id": query_id,
                    "supported": rows[0]["supported"],
                    "used_fallback": rows[0]["used_fallback"],
                    "pg_runtime_s": pg_times[query_id],
                    "method_runtime_s": sum(
                        float(row["method_runtime_s"]) for row in rows
                    )
                    / n_models,
                    "mean_inference_s": sum(
                        float(row["inference_s"]) for row in rows
                    )
                    / n_models,
                    "timeouts_or_errors": sum(bool(row["error"]) for row in rows),
                }
            )

    aggregate_path = OUTPUT_DIR / "random_results.csv"
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_rows[0].keys())
        writer.writeheader()
        writer.writerows(aggregate_rows)

    seen: set[str] = set()
    pg_values = []
    method_values = []
    supported_values = []
    inference_values = []
    for row in aggregate_rows:
        query_id = str(row["query_id"])
        if query_id in seen:
            continue
        seen.add(query_id)
        pg_values.append(float(row["pg_runtime_s"]))
        method_values.append(float(row["method_runtime_s"]))
        supported_values.append(str(row["supported"]).lower() == "true")
        inference_values.append(float(row["mean_inference_s"]))

    speedups = [
        pg_time / method_time
        for pg_time, method_time in zip(pg_values, method_values)
    ]
    summary = {
        "method": "GenJoin (single-run TPC-H adapter)",
        "workload": WORKLOAD,
        "protocol": PROTOCOL,
        "test_records": len(aggregate_rows),
        "model_test_runs": len(raw),
        "unique_queries": len(seen),
        "supported_unique_queries": sum(supported_values),
        "effective_genjoin_coverage_pct": 100.0
        * sum(supported_values)
        / len(supported_values),
        "supported_workload_queries": sum(
            bool(spec["supported"])
            for spec in metadata["queries"].values()  # type: ignore[union-attr]
        ),
        "workload_queries": len(metadata["queries"]),  # type: ignore[arg-type]
        "pg_total_s": sum(pg_values),
        "method_total_s": sum(method_values),
        "mean_inference_s": sum(inference_values) / len(inference_values),
        "WS": sum(pg_values) / sum(method_values),
        "GS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "Imp_pct": 100.0
        * sum(
            method_time < pg_time
            for pg_time, method_time in zip(pg_values, method_values)
        )
        / len(method_values),
        "timeouts_or_errors_across_model_runs": raw_errors,
        "candidate_runs_per_hint_set": 1,
        "decoder_source": "paper semantics plus released ablation implementation",
    }
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    atomic_json(
        OUTPUT_DIR / "artifact_audit.json",
        {
            "source_repository": str(GENJOIN_ROOT),
            "source_commit": metadata["source_commit"],
            "adapter": str(Path(__file__).resolve()),
            "decoder": str(
                GENJOIN_ROOT / "src" / "encoding" / "plan_decoder.py"
            ),
            "candidate_runs_per_hint_set": 1,
            "model_file_count": len(
                list((OUTPUT_DIR / "models").glob("random_*/run*.pt"))
            ),
            "demonstration_records": len(
                collection_rows(OUTPUT_DIR / "collection.csv")
            ),
        },
    )
    return summary


def evaluate(n_models: int = DEFAULT_MODELS) -> dict:
    metadata = load_metadata()
    num_edges = int(metadata["num_edges"])
    pg_times = load_postgres_times(
        POSTGRES_CSV,
        time_column="run1_charged_s",
    )
    models = {
        (fold, run_id): load_model(
            OUTPUT_DIR / "models" / fold / f"run{run_id}.pt"
        )
        for fold in split_folds(WORKLOAD, PROTOCOL)
        for run_id in range(n_models)
    }
    store = CsvResultStore(
        OUTPUT_DIR / "random_model_results.csv", RAW_RESULT_FIELDS
    )

    with database_lock():
        connection = connect(WORKLOAD)
        cursor = connection.cursor()
        try:
            reset_session(cursor, load_pg_hint_plan=True)
            cardinalities = full_cardinalities(cursor)
            for fold, split in split_folds(WORKLOAD, PROTOCOL).items():
                for query_id in split["test"]:
                    keys = [
                        f"{fold}:{query_id}:{run_id}" for run_id in range(n_models)
                    ]
                    if all(store.get(key) is not None for key in keys):
                        continue
                    spec = metadata["queries"][query_id]
                    sql = clean_sql(query_id)

                    if not spec["supported"]:
                        for run_id, result_key in enumerate(keys):
                            if store.get(result_key) is not None:
                                continue
                            store.append(
                                {
                                    "result_key": result_key,
                                    "fold": fold,
                                    "query_id": query_id,
                                    "run_id": run_id,
                                    "supported": False,
                                    "used_fallback": True,
                                    "hints": "",
                                    "encoding_s": 0.0,
                                    "model_inference_s": 0.0,
                                    "inference_s": 0.0,
                                    "timeout_s": dynamic_timeout_s(
                                        pg_times[query_id]
                                    ),
                                    "db_runtime_s": pg_times[query_id],
                                    "charged_db_s": pg_times[query_id],
                                    "method_runtime_s": pg_times[query_id],
                                    "error": "",
                                }
                            )
                        continue

                    encoding_started = time.perf_counter()
                    try:
                        q_encoding = live_query_encoding(
                            cursor, query_id, spec, num_edges, cardinalities
                        )
                        encoding_s = time.perf_counter() - encoding_started
                    except Exception as exc:
                        encoding_s = time.perf_counter() - encoding_started
                        q_encoding = None
                        encoding_error = str(exc).replace("\n", " ")[:1000]
                        connection.rollback()
                        reset_session(cursor, load_pg_hint_plan=True)

                    for run_id, result_key in enumerate(keys):
                        if store.get(result_key) is not None:
                            continue
                        timeout_s = dynamic_timeout_s(pg_times[query_id])
                        if q_encoding is None:
                            store.append(
                                {
                                    "result_key": result_key,
                                    "fold": fold,
                                    "query_id": query_id,
                                    "run_id": run_id,
                                    "supported": True,
                                    "used_fallback": True,
                                    "hints": "",
                                    "encoding_s": encoding_s,
                                    "model_inference_s": 0.0,
                                    "inference_s": encoding_s,
                                    "timeout_s": timeout_s,
                                    "db_runtime_s": pg_times[query_id],
                                    "charged_db_s": pg_times[query_id],
                                    "method_runtime_s": encoding_s
                                    + pg_times[query_id],
                                    "error": encoding_error,
                                }
                            )
                            continue

                        inference_seed = (
                            DEFAULT_SEED
                            + 100000 * run_id
                            + 1000 * sum(ord(char) for char in fold)
                            + int(query_id)
                        )
                        input_encoding = sample_plan_encoding(
                            num_edges, spec["valid_edges"], inference_seed
                        )
                        torch.manual_seed(inference_seed)
                        inference_started = time.perf_counter()
                        with torch.no_grad():
                            output, _, _, _ = models[(fold, run_id)](
                                torch.from_numpy(input_encoding).view(1, -1),
                                torch.from_numpy(q_encoding).view(1, -1),
                                torch.zeros((1, 1), dtype=torch.float32),
                                torch.zeros((1, 1), dtype=torch.float32),
                            )
                        decoded = decoded_plan_encoding(
                            output, num_edges, spec["valid_edges"]
                        )
                        hints = hints_from_encoding(spec, decoded)
                        model_inference_s = time.perf_counter() - inference_started
                        inference_s = encoding_s + model_inference_s
                        result = execute_once(
                            cursor,
                            hinted_sql(sql, hints),
                            timeout_s=timeout_s,
                        )
                        if result.error:
                            connection.rollback()
                            reset_session(cursor, load_pg_hint_plan=True)
                        method_runtime_s = inference_s + result.charged_runtime_s
                        store.append(
                            {
                                "result_key": result_key,
                                "fold": fold,
                                "query_id": query_id,
                                "run_id": run_id,
                                "supported": True,
                                "used_fallback": False,
                                "hints": hints,
                                "encoding_s": encoding_s,
                                "model_inference_s": model_inference_s,
                                "inference_s": inference_s,
                                "timeout_s": timeout_s,
                                "db_runtime_s": ""
                                if result.runtime_s is None
                                else result.runtime_s,
                                "charged_db_s": result.charged_runtime_s,
                                "method_runtime_s": method_runtime_s,
                                "error": result.error or "",
                            }
                        )
                        print(
                            f"evaluate {fold} q={query_id} run={run_id} "
                            f"method={method_runtime_s:.3f}s "
                            f"pg={pg_times[query_id]:.3f}s",
                            flush=True,
                        )
        finally:
            cursor.close()
            connection.close()

    summary = summarize_results(metadata, pg_times, n_models)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "validate", "collect", "train", "evaluate", "all"),
    )
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--pair-gap", type=float, default=DEFAULT_PAIR_GAP)
    parser.add_argument("--models", type=int, default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR, POSTGRES_CSV
    args = parse_args()
    output_root = args.output_root.resolve()
    OUTPUT_DIR = output_root / "tpch" / "genjoin"
    POSTGRES_CSV = output_root / "tpch" / "postgres.csv"
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"validate", "all"}:
        validate()
    if args.stage in {"collect", "all"}:
        collect(args.candidates)
    if args.stage in {"train", "all"}:
        train(
            candidates=args.candidates,
            pair_gap=args.pair_gap,
            n_models=args.models,
            epochs=args.epochs,
        )
    if args.stage in {"evaluate", "all"}:
        evaluate(args.models)


if __name__ == "__main__":
    main()
