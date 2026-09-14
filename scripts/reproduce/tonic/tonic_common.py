#!/usr/bin/env python3
"""Shared TONIC plan-skeleton and QEP-S utilities."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarking.workloads import (
    ROOT,
    query_sql,
    reset_session,
    workload_query_ids,
)


TONIC_ROOT = ROOT / "thrid_party" / "TONIC"
JOB_FEEDBACK_DIR = TONIC_ROOT / "feedback" / "fullData"

JOIN_NODE_TYPES = {"Nested Loop", "Hash Join", "Merge Join"}
INDEPENDENT_PLAN_RELATIONSHIPS = {"SubPlan", "InitPlan", "CTE InitPlan"}
JOIN_HINT_RE = re.compile(r"(HashJoin|NestLoop)\(([^)]*)\)")
HINT_BLOCK_RE = re.compile(r"/\*\+(.*?)\*/\s+(?:time|best_time):", re.DOTALL)
FEEDBACK_RECORD_RE = re.compile(
    r"/\*\+(.*?)\*/\s+(?:time|best_time):\s+([^\s]+)",
    re.DOTALL,
)


@dataclass
class JoinStep:
    aliases: Tuple[str, ...]
    path_keys: Tuple[str, ...]
    controllable: bool = True

    def to_json(self):
        return {
            "aliases": list(self.aliases),
            "path_keys": list(self.path_keys),
            "controllable": self.controllable,
        }

    @staticmethod
    def from_json(value):
        return JoinStep(
            tuple(value["aliases"]),
            tuple(value["path_keys"]),
            bool(value.get("controllable", True)),
        )


@dataclass
class QuerySkeleton:
    query_id: str
    leading_tree: Optional[str]
    steps: List[JoinStep]
    default_assignment: Optional[str] = None

    @property
    def decision_count(self):
        return sum(step.controllable for step in self.steps)

    def to_json(self):
        return {
            "query_id": self.query_id,
            "leading_tree": self.leading_tree,
            "steps": [step.to_json() for step in self.steps],
            "default_assignment": self.default_assignment,
        }

    @staticmethod
    def from_json(value):
        return QuerySkeleton(
            value["query_id"],
            value.get("leading_tree"),
            [JoinStep.from_json(step) for step in value["steps"]],
            value.get("default_assignment"),
        )


def configure_tonic_session(cursor) -> None:
    reset_session(cursor, load_pg_hint_plan=True)
    cursor.execute("SET join_collapse_limit TO 1")
    cursor.execute("SET enable_nestloop TO false")


def explain_json(cursor, sql: str):
    cursor.execute("EXPLAIN (FORMAT JSON) " + sql)
    return cursor.fetchone()[0][0]["Plan"]


def _all_children(node):
    return node.get("Plans", [])


def _input_children(node):
    return [
        child
        for child in _all_children(node)
        if child.get("Parent Relationship") not in INDEPENDENT_PLAN_RELATIONSHIPS
    ]


def _scan_alias(node):
    if node.get("Relation Name"):
        return node.get("Alias") or node["Relation Name"]
    if node.get("Node Type") in {"CTE Scan", "WorkTable Scan", "Named Tuplestore Scan"}:
        return node.get("Alias") or node.get("CTE Name")
    return None


def _relation_aliases(node):
    alias = _scan_alias(node)
    if alias:
        return [alias]
    aliases = []
    for child in _input_children(node):
        aliases.extend(_relation_aliases(child))
    return aliases


def _alias_relation_map(node, result):
    alias = _scan_alias(node)
    if alias:
        relation = node.get("Relation Name") or node.get("CTE Name") or alias
        result[alias] = relation
    for child in _all_children(node):
        _alias_relation_map(child, result)


def _plan_join_steps(node, result):
    for child in _all_children(node):
        _plan_join_steps(child, result)
    if node.get("Node Type") not in JOIN_NODE_TYPES:
        return
    aliases = []
    for child in _input_children(node):
        aliases.extend(_relation_aliases(child))
    if len(aliases) >= 2:
        result.append(tuple(aliases))


def _plan_join_records(node, result):
    for child in _all_children(node):
        _plan_join_records(child, result)
    if node.get("Node Type") not in JOIN_NODE_TYPES:
        return
    aliases = []
    for child in _input_children(node):
        aliases.extend(_relation_aliases(child))
    if len(aliases) >= 2:
        result.append((tuple(aliases), node.get("Node Type")))


def _leading_tree(node):
    alias = _scan_alias(node)
    if alias:
        return alias

    children = _input_children(node)
    if node.get("Node Type") in JOIN_NODE_TYPES and len(children) >= 2:
        left = _leading_tree(children[0])
        right = _leading_tree(children[1])
        if left and right:
            return "({} {})".format(left, right)

    trees = [tree for tree in (_leading_tree(child) for child in children) if tree]
    if len(trees) == 1:
        return trees[0]
    if len(trees) >= 2:
        tree = trees[0]
        for child_tree in trees[1:]:
            tree = "({} {})".format(tree, child_tree)
        return tree
    return None


def _relation_id(alias, alias_relations):
    return alias_relations.get(alias, alias)


def _qeps_path_keys(join_aliases, alias_relations):
    """Translate post-order join groups into TONIC's plain QEP-S trie paths."""
    if not join_aliases:
        return []

    root_alias = join_aliases[-1][0]
    main_path = []
    previous_aliases = []
    previous_was_subquery = False
    result = []

    for step_idx, aliases in enumerate(join_aliases):
        is_subquery = aliases[0] != root_alias
        if is_subquery:
            previous_was_subquery = True
        elif previous_was_subquery:
            previous_was_subquery = False
            subquery_id = "#".join(
                _relation_id(alias, alias_relations)
                for alias in previous_aliases
            )
            if main_path:
                main_path.append(subquery_id)
            else:
                main_path = [
                    _relation_id(root_alias, alias_relations),
                    subquery_id,
                ]
        else:
            suffix = aliases[step_idx + 1 :]
            normalized_suffix = [
                _relation_id(alias, alias_relations) for alias in suffix
            ]
            if main_path:
                main_path.extend(normalized_suffix)
            else:
                main_path = [
                    _relation_id(root_alias, alias_relations),
                    *normalized_suffix,
                ]
        previous_aliases = aliases

        if is_subquery:
            normalized = [
                _relation_id(alias, alias_relations) for alias in aliases
            ]
            if len(normalized) == 2:
                path_keys = ("#" + normalized[-2], "#" + normalized[-1])
            else:
                path_keys = ("#" + normalized[-1],)
        elif len(aliases) == 2 and main_path and "#" not in main_path[-1]:
            path_keys = (main_path[-2], main_path[-1])
        else:
            path_keys = (main_path[-1],)

        result.append(path_keys)

    return result


def build_skeleton(cursor, workload: str, query_id: str) -> QuerySkeleton:
    workload = workload.upper()
    configure_tonic_session(cursor)
    plan = explain_json(cursor, query_sql(workload, query_id))
    aliases = []
    _plan_join_steps(plan, aliases)

    alias_relations = {}
    _alias_relation_map(plan, alias_relations)
    path_keys = _qeps_path_keys(aliases, alias_relations)
    steps = [
        JoinStep(tuple(step_aliases), tuple(step_path))
        for step_aliases, step_path in zip(aliases, path_keys)
    ]
    skeleton = QuerySkeleton(query_id, _leading_tree(plan), steps)

    default_records = []
    _plan_join_records(plan, default_records)
    default_by_aliases = {}
    duplicates = set()
    for record_aliases, node_type in default_records:
        key = tuple(sorted(record_aliases))
        if key in default_by_aliases:
            duplicates.add(key)
        default_by_aliases[key] = node_type
    for key in duplicates:
        default_by_aliases.pop(key, None)
    assignment = []
    for step in skeleton.steps:
        if not step.controllable:
            continue
        node_type = default_by_aliases.get(tuple(sorted(step.aliases)))
        assignment.append("N" if node_type == "Nested Loop" else "H")
    skeleton.default_assignment = "".join(assignment)
    return skeleton


def build_skeletons(cursor, workload: str):
    return {
        query_id: build_skeleton(cursor, workload, query_id)
        for query_id in workload_query_ids(workload)
    }


def save_skeletons(path: Path, skeletons: Dict[str, QuerySkeleton]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                query_id: skeleton.to_json()
                for query_id, skeleton in sorted(skeletons.items())
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(path)


def load_skeletons(path: Path):
    content = json.loads(path.read_text())
    return {
        query_id: QuerySkeleton.from_json(value)
        for query_id, value in content.items()
    }


def assignment_hint(skeleton: QuerySkeleton, assignment: str) -> str:
    if len(assignment) != skeleton.decision_count:
        raise ValueError(
            "{} assignment has {} operators for {} joins".format(
                skeleton.query_id,
                len(assignment),
                skeleton.decision_count,
            )
        )
    hints = []
    assignment_idx = 0
    for step in skeleton.steps:
        if not step.controllable:
            continue
        operator = assignment[assignment_idx]
        assignment_idx += 1
        name = "HashJoin" if operator == "H" else "NestLoop"
        hints.append("{}({})".format(name, " ".join(step.aliases)))
    return " ".join(hints)


def hinted_sql(workload: str, skeleton: QuerySkeleton, assignment: str) -> str:
    hint = assignment_hint(skeleton, assignment)
    if not hint:
        return query_sql(workload, skeleton.query_id)
    return "/*+ {} */\n{}".format(hint, query_sql(workload, skeleton.query_id))


def _assignment_from_hint(skeleton: QuerySkeleton, hint: str) -> str:
    parsed = [
        ("H" if operator == "HashJoin" else "N", tuple(relations.split()))
        for operator, relations in JOIN_HINT_RE.findall(hint)
    ]
    if len(parsed) != len(skeleton.steps):
        raise ValueError(
            "{} source hint has {} joins, expected {}".format(
                skeleton.query_id, len(parsed), len(skeleton.steps)
            )
        )
    return "".join(
        operator
        for (operator, _), step in zip(parsed, skeleton.steps)
        if step.controllable
    )


def job_source_assignments(skeleton: QuerySkeleton):
    return list(job_source_feedback(skeleton))


def job_source_feedback(skeleton: QuerySkeleton):
    content = (JOB_FEEDBACK_DIR / "{}.sql".format(skeleton.query_id)).read_text()
    feedback = {}
    for hint, runtime in FEEDBACK_RECORD_RE.findall(content):
        assignment = _assignment_from_hint(skeleton, hint)
        value = float(runtime)
        feedback[assignment] = min(value, feedback.get(assignment, value))
    if not feedback:
        raise RuntimeError("no source TONIC candidates for {}".format(skeleton.query_id))
    return feedback


def initial_assignments(workload: str, skeleton: QuerySkeleton):
    if workload.upper() == "JOB":
        return job_source_assignments(skeleton)
    return [
        "".join(bits)
        for bits in itertools.product(
            ("H", "N"),
            repeat=skeleton.decision_count,
        )
    ]


class QepsNode:
    def __init__(self):
        self.children = {}
        self.hash_cost = 0.0
        self.nest_cost = 0.0

    def child(self, key, create):
        if key in self.children:
            return self.children[key]
        if not create:
            return None
        child = QepsNode()
        self.children[key] = child
        return child

    def update(self, hash_cost, nest_cost):
        self.hash_cost += hash_cost
        self.nest_cost += nest_cost

    def recommendation(self):
        return "H" if self.hash_cost <= self.nest_cost else "N"


def integrate_feedback(
    root: QepsNode,
    skeleton: QuerySkeleton,
    runtimes: Dict[str, float],
    *,
    active_join_count: Optional[int] = None,
) -> None:
    if not runtimes:
        raise ValueError("missing feedback for {}".format(skeleton.query_id))
    best = min(runtimes, key=lambda assignment: (runtimes[assignment], assignment))
    if active_join_count is None:
        active_join_count = len(skeleton.steps)
    node = root
    assignment_idx = 0
    for idx, step in enumerate(skeleton.steps):
        for key in step.path_keys:
            node = node.child(key, create=True)
        if not step.controllable:
            continue
        hash_assignment = (
            best[:assignment_idx] + "H" + best[assignment_idx + 1 :]
        )
        nest_assignment = (
            best[:assignment_idx] + "N" + best[assignment_idx + 1 :]
        )
        if hash_assignment not in runtimes or nest_assignment not in runtimes:
            raise RuntimeError(
                "{} feedback is incomplete at join {}".format(skeleton.query_id, idx)
            )
        if idx < active_join_count:
            node.update(runtimes[hash_assignment], runtimes[nest_assignment])
        assignment_idx += 1


def predict_assignment(root: QepsNode, skeleton: QuerySkeleton) -> str:
    assignment = []
    node = root
    for step in skeleton.steps:
        for key in step.path_keys:
            node = node.child(key, create=False) if node else None
        if step.controllable:
            assignment.append(node.recommendation() if node else "H")
    return "".join(assignment)
