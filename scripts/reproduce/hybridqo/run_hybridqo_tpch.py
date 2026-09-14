#!/usr/bin/env python3
"""Adapt and evaluate the released HybridQO implementation on TPC-H.

The released implementation hard-codes JOB/STACK schema dimensions and only
parses a narrow SQL subset.  This runner keeps HybridQO's SPINN value model,
MCTS hint search, and online training procedure while providing:

* TPC-H database, alias, and column dimensions;
* a pglast-based encoder for the outer query block (or a derived-table block);
* fold-isolated training and inference-only test execution; and
* an explicit PostgreSQL fallback for unsupported or failed queries.

All PostgreSQL operations are issued sequentially.  Existing PostgreSQL run-1
measurements are reused as the default-plan feedback and fallback runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from pglast import ast, parse_sql
from pglast.stream import RawStream

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    dynamic_timeout_s,
    load_postgres_times,
    query_sql,
    split_folds,
)


ROOT = Path(__file__).resolve().parents[3]
HYBRID_ROOT = (
    ROOT
    / "thrid_party"
    / "genjoin"
    / "OtherMethods_Repository"
    / "hybrid_qo"
)
WORKLOAD = "TPCH"
ALIASES = (
    "start",
    "c",
    "c2",
    "l",
    "l1",
    "l2",
    "l3",
    "n",
    "n1",
    "n2",
    "o",
    "p",
    "ps",
    "ps2",
    "r",
    "r2",
    "s",
    "s2",
)
MAX_COLUMNS = 128
RESULT_FIELDS = (
    "fold",
    "query_id",
    "supported",
    "used_fallback",
    "fallback_reason",
    "chosen_plan",
    "hint",
    "pg_runtime_s",
    "method_runtime_s",
    "execution_runtime_s",
    "timed_out",
    "inference_s",
)


class UnsupportedQuery(RuntimeError):
    pass


def install_tpch_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Replace the baseline's hard-coded Config before importing its modules."""
    sys.path.insert(0, str(HYBRID_ROOT))
    import ImportantConfig

    aliases = dict(enumerate(ALIASES))

    class TPCHConfig:
        def __init__(self) -> None:
            self.schemaFile = "schema.sql"
            self.user = args.user
            self.password = ""
            self.userName = self.user
            self.dataset = WORKLOAD
            self.usegpu = torch.cuda.is_available()
            self.head_num = 10
            self.input_size = 9
            self.hidden_size = 64
            self.batch_size = 256
            self.ip = args.host
            self.port = args.port
            self.device = torch.device(
                args.device if torch.cuda.is_available() else "cpu"
            )
            self.cpudevice = self.device
            self.var_weight = 0.0
            self.cost_test_for_debug = False
            self.max_hint_num = 20
            self.max_time_out = 360 * 1000
            self.threshold = math.log(3) / math.log(self.max_time_out)
            self.leading_length = 2
            self.try_hint_num = 3
            self.mem_size = 2000
            self.mcts_v = 1.1
            self.searchFactor = 4
            self.U_factor = 0.0
            self.log_file = str(output_dir / "hybridqo_tpch.log")
            self.latency_file = str(output_dir / "selectivity_cache.jsonl")
            self.modelpath = str(output_dir / "models")
            self.offset = 20
            self.database = "tpch"
            self.max_alias_num = len(aliases)
            self.id2aliasname = aliases
            self.aliasname2id = {value: key for key, value in aliases.items()}
            self.max_column = MAX_COLUMNS
            self.n_epochs = args.epochs
            self.queries_file = ""
            self.mcts_input_size = (
                self.max_alias_num * self.max_alias_num + self.max_column
            )

    ImportantConfig.Config = TPCHConfig


def iter_nodes(value: object, *, descend_sublinks: bool = False) -> Iterator[ast.Node]:
    if isinstance(value, ast.SubLink) and not descend_sublinks:
        return
    if isinstance(value, ast.Node):
        yield value
        for field in value:
            child = getattr(value, field, None)
            yield from iter_nodes(child, descend_sublinks=descend_sublinks)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_nodes(child, descend_sublinks=descend_sublinks)


def column_refs(value: object) -> List[Tuple[str, str]]:
    refs = []
    for node in iter_nodes(value):
        if not isinstance(node, ast.ColumnRef) or len(node.fields) < 2:
            continue
        first, second = node.fields[0], node.fields[1]
        if isinstance(first, ast.String) and isinstance(second, ast.String):
            refs.append((first.sval, second.sval))
    return refs


def contains_sublink(value: object) -> bool:
    return any(
        isinstance(node, ast.SubLink)
        for node in iter_nodes(value, descend_sublinks=True)
    )


def query_scope(sql: str) -> ast.SelectStmt:
    statements = parse_sql(sql)
    if len(statements) != 1 or not isinstance(statements[0].stmt, ast.SelectStmt):
        raise UnsupportedQuery("not a single SELECT statement")
    scope = statements[0].stmt
    while (
        scope.fromClause
        and len(scope.fromClause) == 1
        and isinstance(scope.fromClause[0], ast.RangeSubselect)
        and isinstance(scope.fromClause[0].subquery, ast.SelectStmt)
    ):
        scope = scope.fromClause[0].subquery
    return scope


def from_tables(value: object) -> List[Tuple[str, str]]:
    tables: List[Tuple[str, str]] = []

    def visit(node: object) -> None:
        if isinstance(node, ast.RangeVar):
            alias = node.alias.aliasname if node.alias else node.relname
            tables.append((alias, node.relname))
        elif isinstance(node, ast.JoinExpr):
            visit(node.larg)
            visit(node.rarg)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
        elif node is not None:
            raise UnsupportedQuery(
                "unsupported FROM item {}".format(type(node).__name__)
            )

    visit(value)
    return tables


def join_quals(value: object) -> Iterator[object]:
    if isinstance(value, ast.JoinExpr):
        if value.quals is not None:
            yield value.quals
        yield from join_quals(value.larg)
        yield from join_quals(value.rarg)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from join_quals(child)


def predicate_leaves(value: object) -> Iterator[object]:
    if value is None or isinstance(value, ast.SubLink):
        return
    if isinstance(value, ast.BoolExpr):
        for child in value.args or ():
            yield from predicate_leaves(child)
        return
    if isinstance(value, (ast.A_Expr, ast.NullTest, ast.BooleanTest)):
        yield value
        return
    if isinstance(value, ast.Node):
        yield value


def connected(aliases: Set[str], joins: Set[Tuple[str, str]]) -> bool:
    if not aliases:
        return False
    graph = {alias: set() for alias in aliases}
    for left, right in joins:
        graph[left].add(right)
        graph[right].add(left)
    pending = [next(iter(aliases))]
    reached = set()
    while pending:
        alias = pending.pop()
        if alias in reached:
            continue
        reached.add(alias)
        pending.extend(graph[alias] - reached)
    return reached == aliases


def normalize_plan(plan: dict) -> dict:
    root = deepcopy(plan)
    if "Plan" in root:
        root["Plan"] = normalize_plan(root["Plan"])
        return root

    original_children = root.get("Plans", [])
    if (
        root.get("Node Type") == "Bitmap Heap Scan"
        and len(original_children) == 1
        and original_children[0].get("Node Type") == "Bitmap Index Scan"
    ):
        original_children[0]["Alias"] = root.get("Alias")
        original_children[0]["Relation Name"] = root.get("Relation Name")

    children = []
    for child in original_children:
        if child.get("Parent Relationship") in ("SubPlan", "InitPlan"):
            continue
        children.append(normalize_plan(child))
    if children:
        root["Plans"] = children
    else:
        root.pop("Plans", None)

    if root.get("Node Type") == "CTE Scan":
        root["Node Type"] = "Seq Scan"
        root["Relation Name"] = root.get("CTE Name", root.get("Alias", "cte"))
    return root


def execute_wall_ms(pgrunner, sql: str, timeout_ms: int) -> Tuple[float, bool, str]:
    try:
        if "Leading" in sql:
            pgrunner.cur.execute("SET geqo TO off")
        else:
            pgrunner.cur.execute("SET geqo TO on")
            pgrunner.cur.execute("SET geqo_threshold = 12")
        pgrunner.cur.execute("SET statement_timeout = %s", (timeout_ms,))
        started = time.perf_counter()
        pgrunner.cur.execute(sql)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        return runtime_ms, False, ""
    except Exception as exc:
        pgrunner.con.rollback()
        return float(timeout_ms), True, str(exc).replace("\n", " ")[:1000]


def initialize_weights(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if len(parameter.shape) == 2:
            torch.nn.init.xavier_normal_(parameter)
        else:
            torch.nn.init.uniform_(parameter)


def build_runtime(args: argparse.Namespace, output_dir: Path):
    install_tpch_config(args, output_dir)

    import mcts as mcts_module
    import sql2fea as sql2fea_module
    from Hinter import Hinter as BaseHinter
    from ImportantConfig import Config
    from NET import TreeNet
    from TreeLSTM import SPINN
    from mcts import MCTSHinterSearch

    config = Config()
    pgrunner = sql2fea_module.pgrunner

    pgrunner.cur.execute(
        """
        SELECT attribute.attname
        FROM pg_attribute AS attribute
        JOIN pg_class AS relation ON relation.oid = attribute.attrelid
        JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped
          AND relation.relkind IN ('r', 'p')
        ORDER BY relation.relname, attribute.attnum
        """
    )
    column_names = []
    for (column_name,) in pgrunner.cur.fetchall():
        if column_name not in column_names:
            column_names.append(column_name)
    if len(column_names) > config.max_column:
        raise RuntimeError(
            "TPC-H has {} columns, max_column={}".format(
                len(column_names), config.max_column
            )
        )
    column_ids = {name: index for index, name in enumerate(column_names)}

    class TPCHSql2Vec:
        def __init__(self) -> None:
            self.id2aliasname = config.id2aliasname
            self.aliasname2id = config.aliasname2id

        def to_vec(self, sql: str):
            scope = query_scope(sql)
            tables = from_tables(scope.fromClause or ())
            alias_to_table = dict(tables)
            aliases = set(alias_to_table)
            if len(aliases) < 2:
                raise UnsupportedQuery("query block has fewer than two relations")
            unknown = aliases - set(self.aliasname2id)
            if unknown:
                raise UnsupportedQuery(
                    "unknown aliases {}".format(",".join(sorted(unknown)))
                )

            joins: Set[Tuple[str, str]] = set()
            selections: List[Tuple[object, str, str]] = []
            has_predicate: Set[str] = set()
            roots: List[object] = [scope.whereClause]
            roots.extend(join_quals(scope.fromClause or ()))
            for root in roots:
                for predicate in predicate_leaves(root):
                    refs = [
                        ref for ref in column_refs(predicate) if ref[0] in aliases
                    ]
                    predicate_aliases = {alias for alias, _ in refs}
                    if (
                        isinstance(predicate, ast.A_Expr)
                        and len(predicate_aliases) == 2
                    ):
                        left_aliases = {
                            alias
                            for alias, _ in column_refs(predicate.lexpr)
                            if alias in aliases
                        }
                        right_aliases = {
                            alias
                            for alias, _ in column_refs(predicate.rexpr)
                            if alias in aliases
                        }
                        if len(left_aliases) == len(right_aliases) == 1:
                            left = next(iter(left_aliases))
                            right = next(iter(right_aliases))
                            if left != right:
                                joins.add(tuple(sorted((left, right))))
                                continue
                    if (
                        len(predicate_aliases) == 1
                        and refs
                        and not contains_sublink(predicate)
                    ):
                        alias = next(iter(predicate_aliases))
                        has_predicate.add(alias)
                        selections.append((predicate, alias, refs[0][1]))

            if not connected(aliases, joins):
                raise UnsupportedQuery(
                    "outer query-block join graph is not connected"
                )

            join_matrix = np.zeros(
                (config.max_alias_num, config.max_alias_num), dtype=float
            )
            for left, right in joins:
                left_id = self.aliasname2id[left]
                right_id = self.aliasname2id[right]
                join_matrix[left_id][right_id] = 1.0
                join_matrix[right_id][left_id] = 1.0

            selectivities = np.zeros(config.max_column, dtype=float)
            for predicate, alias, column in selections:
                if column not in column_ids:
                    continue
                predicate_sql = RawStream()(predicate)
                relation_sql = "{} AS {}".format(alias_to_table[alias], alias)
                try:
                    value = pgrunner.getSelectivity(relation_sql, predicate_sql)
                except Exception:
                    pgrunner.con.rollback()
                    continue
                if math.isfinite(value):
                    selectivities[column_ids[column]] += value

            self.aliasnames_root_set = aliases
            self.aliasname2fullname = alias_to_table
            self.aliasnames = aliases
            self.join_list = joins
            selected_joins = {
                pair
                for pair in joins
                if pair[0] in has_predicate or pair[1] in has_predicate
            }
            self.join_list_with_predicate = selected_joins or set(joins)
            return np.concatenate((join_matrix.flatten(), selectivities)), aliases

    class TPCHTreeBuilder(sql2fea_module.TreeBuilder):
        def plan_to_feature_tree(self, plan):
            return super().plan_to_feature_tree(normalize_plan(plan))

    class RevisionHinter(BaseHinter):
        def run(
            self,
            sql: str,
            *,
            pg_runtime_ms: float,
            timeout_ms: int,
            train: bool,
            execute: bool = True,
        ) -> dict:
            self.hinter_times += 1
            started = time.perf_counter()
            plan_json_pg = pgrunner.getCostPlanJson(sql, timeout=timeout_ms)
            sql_vec, aliases = self.sql2vec.to_vec(sql)
            mask = (
                torch.rand(1, config.head_num, device=config.device) < 0.9
            ).long()
            default_prediction = self.predictWithUncertaintyBatch(
                [plan_json_pg], sql_vec
            )[0]
            # The released helper accounts for outer EXPLAIN time through this
            # list, which its original hinterRun initializes before the call.
            self.explains_time_list.append(0.0)
            chosen = self.findBestHint(
                plan_json_PG=plan_json_pg,
                alias=aliases,
                sql_vec=sql_vec,
                sql=sql,
            )
            inference_s = time.perf_counter() - started

            knn_prediction = abs(self.knn.kNeightboursSample(default_prediction))
            use_hint = (
                chosen[0][0] < default_prediction[0]
                and knn_prediction < config.threshold
                and self.value_extractor.decode(default_prediction[0]) > 100
            )
            if not execute:
                replayed_pg_s = None if use_hint else pg_runtime_ms / 1000.0
                return {
                    "chosen_plan": "hint" if use_hint else "PG",
                    "hint": chosen[1] if use_hint else "",
                    "execution_runtime_s": replayed_pg_s,
                    "method_runtime_s": replayed_pg_s,
                    "timed_out": False,
                    "error": "",
                    "inference_s": inference_s,
                }

            samples = []
            hint = ""
            chosen_plan = "PG"
            execution_ms = pg_runtime_ms
            method_ms = pg_runtime_ms
            timed_out = False
            error = ""

            if use_hint:
                hint = chosen[1]
                chosen_plan = "hint"
                hinted_sql = hint + sql
                hinted_plan = pgrunner.getCostPlanJson(
                    hinted_sql, timeout=timeout_ms
                )
                execution_ms, timed_out, error = execute_wall_ms(
                    pgrunner, hinted_sql, timeout_ms
                )
                self.knn.insertAValue(
                    (
                        chosen[0],
                        self.value_extractor.encode(execution_ms) - chosen[0][0],
                    )
                )
                if timed_out:
                    method_ms = execution_ms + pg_runtime_ms
                    samples.append(
                        (
                            hinted_plan,
                            min(execution_ms, 1.8 * pg_runtime_ms),
                            mask,
                        )
                    )
                    samples.append((plan_json_pg, pg_runtime_ms, mask))
                    chosen_plan = "hint_timeout_then_pg"
                else:
                    method_ms = execution_ms
                    samples.append((hinted_plan, execution_ms, mask))
            else:
                self.knn.insertAValue(
                    (
                        default_prediction,
                        self.value_extractor.encode(pg_runtime_ms)
                        - default_prediction[0],
                    )
                )
                samples.append((plan_json_pg, pg_runtime_ms, mask))

            if train:
                for plan_json, runtime_ms, sample_mask in samples:
                    target = self.value_extractor.encode(runtime_ms)
                    self.model.train(
                        plan_json=plan_json,
                        sql_vec=sql_vec,
                        target_value=target,
                        mask=sample_mask,
                        is_train=True,
                    )
                    self.mcts_searcher.train(
                        tree_feature=self.model.tree_builder.plan_to_feature_tree(
                            plan_json
                        ),
                        sql_vec=sql_vec,
                        target_value=runtime_ms,
                        alias_set=aliases,
                    )
                self.model.optimize()
                self.mcts_searcher.optimize()
                self.model.optimize()
                self.mcts_searcher.optimize()

            return {
                "chosen_plan": chosen_plan,
                "hint": hint,
                "execution_runtime_s": execution_ms / 1000.0,
                "method_runtime_s": method_ms / 1000.0,
                "timed_out": timed_out,
                "error": error,
                "inference_s": inference_s,
            }

    tree_builder = TPCHTreeBuilder()
    sql2vec = TPCHSql2Vec()
    value_network = SPINN(
        head_num=config.head_num,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        table_num=config.max_alias_num,
        sql_size=config.mcts_input_size,
    ).to(config.device)
    initialize_weights(value_network)
    initialize_weights(mcts_module.predictionNet)
    model = TreeNet(tree_builder=tree_builder, value_network=value_network)
    searcher = MCTSHinterSearch()
    # The release periodically writes to a hard-coded ./model directory and
    # moves its network through CPU while doing so.  Checkpointing is not part
    # of inference and each revision fold runs to completion in one process.
    searcher.savemodel = lambda: None
    hinter = RevisionHinter(
        model=model,
        sql2vec=sql2vec,
        value_extractor=sql2fea_module.value_extractor,
        mcts_searcher=searcher,
    )
    hinter.revision_mcts_module = mcts_module
    return config, pgrunner, sql2vec, hinter


def compatibility(sql2vec, query_ids: Iterable[str]) -> Dict[str, str]:
    status = {}
    for query_id in query_ids:
        try:
            sql2vec.to_vec(query_sql(WORKLOAD, query_id))
            status[query_id] = ""
        except Exception as exc:
            status[query_id] = "{}: {}".format(
                type(exc).__name__, str(exc).replace("\n", " ")[:500]
            )
    return status


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def postgres_csv(args: argparse.Namespace) -> Path:
    if args.postgres_csv is not None:
        return args.postgres_csv.resolve()
    return args.output_root.resolve() / "tpch" / "postgres.csv"


def run_fold(args: argparse.Namespace, output_dir: Path) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config, pgrunner, sql2vec, hinter = build_runtime(args, output_dir)
    pg_times = load_postgres_times(
        postgres_csv(args),
        time_column="run1_charged_s",
    )
    spec = split_folds(WORKLOAD, "random")[args.fold]
    all_ids = list(dict.fromkeys(spec["train"] + spec["test"]))
    support = compatibility(sql2vec, all_ids)
    supported_train = [
        query_id for query_id in spec["train"] if not support[query_id]
    ]
    if args.max_train_queries:
        supported_train = supported_train[: args.max_train_queries]

    failures: List[dict] = []
    train_started = time.perf_counter()
    completed_steps = 0
    attempted_steps = 0
    total_steps = args.epochs * len(supported_train)
    for epoch in range(args.epochs):
        epoch_ids = list(supported_train)
        random.shuffle(epoch_ids)
        for query_id in epoch_ids:
            attempted_steps += 1
            sql = query_sql(WORKLOAD, query_id)
            timeout_ms = int(
                math.ceil(1000.0 * dynamic_timeout_s(pg_times[query_id]))
            )
            try:
                hinter.run(
                    sql,
                    pg_runtime_ms=pg_times[query_id] * 1000.0,
                    timeout_ms=timeout_ms,
                    train=True,
                )
                completed_steps += 1
            except Exception as exc:
                pgrunner.con.rollback()
                failures.append(
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "query_id": query_id,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            if attempted_steps % 25 == 0 or attempted_steps == total_steps:
                print(
                    "[HybridQO {}] epoch {}/{} ({}/{} attempted, {} passed)".format(
                        args.fold,
                        epoch + 1,
                        args.epochs,
                        attempted_steps,
                        total_steps,
                        completed_steps,
                    ),
                    flush=True,
                )
    training_s = time.perf_counter() - train_started

    checkpoint_dir = output_dir / "models" / args.fold
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": args.fold,
            "epochs": args.epochs,
            "seed": args.seed,
            "value_network": hinter.model.value_network.state_dict(),
            "mcts_prediction_network": (
                hinter.revision_mcts_module.predictionNet.state_dict()
            ),
            "knn_values": hinter.knn.kvs,
        },
        checkpoint_dir / "final.pt",
    )

    rows = []
    for query_id in spec["test"]:
        pg_runtime = pg_times[query_id]
        row = {
            "fold": args.fold,
            "query_id": query_id,
            "supported": not bool(support[query_id]),
            "used_fallback": True,
            "fallback_reason": support[query_id],
            "chosen_plan": "PG_fallback",
            "hint": "",
            "pg_runtime_s": pg_runtime,
            "method_runtime_s": pg_runtime,
            "execution_runtime_s": pg_runtime,
            "timed_out": False,
            "inference_s": 0.0,
        }
        if not support[query_id]:
            sql = query_sql(WORKLOAD, query_id)
            timeout_ms = int(
                math.ceil(1000.0 * dynamic_timeout_s(pg_runtime))
            )
            try:
                result = hinter.run(
                    sql,
                    pg_runtime_ms=pg_runtime * 1000.0,
                    timeout_ms=timeout_ms,
                    train=False,
                    execute=not args.prediction_only,
                )
                row.update(
                    {
                        "used_fallback": result["chosen_plan"]
                        == "hint_timeout_then_pg",
                        "fallback_reason": result["error"],
                        "chosen_plan": result["chosen_plan"],
                        "hint": result["hint"],
                        "method_runtime_s": result["method_runtime_s"],
                        "execution_runtime_s": result["execution_runtime_s"],
                        "timed_out": result["timed_out"],
                        "inference_s": result["inference_s"],
                    }
                )
            except Exception as exc:
                pgrunner.con.rollback()
                row["fallback_reason"] = "test error: {}: {}".format(
                    type(exc).__name__, str(exc).replace("\n", " ")[:500]
                )
                failures.append(
                    {
                        "phase": "test",
                        "query_id": query_id,
                        "error": row["fallback_reason"],
                        "traceback": traceback.format_exc(),
                    }
                )
        rows.append(row)
        runtime_text = (
            "not executed"
            if row["method_runtime_s"] is None
            else "{:.6f}s".format(float(row["method_runtime_s"]))
        )
        print(
            "[HybridQO {} test] Q{} {} {}".format(
                args.fold, query_id, row["chosen_plan"], runtime_text
            ),
            flush=True,
        )

    write_rows(output_dir / "{}_results.csv".format(args.fold), rows)
    audit = {
        "method": "HybridQO",
        "workload": WORKLOAD,
        "fold": args.fold,
        "epochs": args.epochs,
        "train_queries": len(spec["train"]),
        "supported_train_queries": supported_train,
        "completed_training_steps": completed_steps,
        "training_wall_s": training_s,
        "compatibility": support,
        "failures": failures,
        "fallback_policy": (
            "Unsupported/failed queries use the matching PostgreSQL run-1 time."
        ),
        "test_sql_executed": not args.prediction_only,
        "test_execution_policy": (
            "prediction only; selected test hints were not executed"
            if args.prediction_only
            else "selected test hints were executed once"
        ),
    }
    (output_dir / "{}_audit.json".format(args.fold)).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    pgrunner.cur.close()
    pgrunner.con.close()
    return audit


def summarize(output_dir: Path, postgres_path: Path) -> dict:
    pg_times = load_postgres_times(
        postgres_path,
        time_column="run1_charged_s",
    )
    rows_by_key = {}
    expected = []
    for fold, spec in split_folds(WORKLOAD, "random").items():
        path = output_dir / "{}_results.csv".format(fold)
        if not path.is_file():
            raise RuntimeError("missing {}".format(path))
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows_by_key[(fold, row["query_id"])] = row
        expected.extend((fold, query_id) for query_id in spec["test"])
    if set(rows_by_key) != set(expected):
        raise RuntimeError("HybridQO result coverage is incomplete")

    seen = set()
    pg_values = []
    method_values = []
    method_queries = []
    supported_queries = []
    for key in expected:
        query_id = key[1]
        if query_id in seen:
            continue
        seen.add(query_id)
        row = rows_by_key[key]
        pg_values.append(pg_times[query_id])
        method_values.append(float(row["method_runtime_s"]))
        if row["supported"].lower() == "true":
            supported_queries.append(query_id)
        if row["chosen_plan"] not in ("PG", "PG_fallback"):
            method_queries.append(query_id)
    speedups = [
        pg_time / method_time
        for pg_time, method_time in zip(pg_values, method_values)
    ]
    summary = {
        "method": "HybridQO with explicit PostgreSQL fallback",
        "workload": WORKLOAD,
        "protocol": "random",
        "test_records": len(expected),
        "unique_queries": len(seen),
        "encoder_supported_unique_queries": supported_queries,
        "encoder_coverage_pct": 100.0 * len(supported_queries) / len(seen),
        "hint_selected_unique_queries": method_queries,
        "hint_selection_pct": 100.0 * len(method_queries) / len(seen),
        "pg_total_s": sum(pg_values),
        "method_total_s": sum(method_values),
        "WS": sum(pg_values) / sum(method_values),
        "GS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "Imp_pct": 100.0
        * sum(
            method_time < pg_time
            for pg_time, method_time in zip(pg_values, method_values)
        )
        / len(method_values),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("scan", "run", "summarize"), default="run"
    )
    parser.add_argument(
        "--fold",
        choices=("random_a", "random_b", "random_c"),
        default="random_a",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="direct HybridQO output directory (overrides --output-root)",
    )
    parser.add_argument(
        "--postgres-csv",
        type=Path,
        help="PostgreSQL measurements to reuse for training feedback",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="record test plan/hint choices without executing test SQL",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.output_root.resolve() / "tpch" / "hybridqo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "summarize":
        print(json.dumps(summarize(output_dir, postgres_csv(args)), indent=2))
        return

    if args.phase == "scan":
        _, pgrunner, sql2vec, _ = build_runtime(args, output_dir)
        query_ids = [str(value) for value in range(1, 23)]
        print(json.dumps(compatibility(sql2vec, query_ids), indent=2))
        pgrunner.cur.close()
        pgrunner.con.close()
        return

    print(json.dumps(run_fold(args, output_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
