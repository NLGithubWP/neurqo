#!/usr/bin/env python3
"""Shared utilities for baseline reproduction measurements."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import psycopg2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_WORK_ROOT = (
    ROOT / "results" / "benchmark" / "temp_baseline_reproduction"
)

WORKLOAD_DATABASES = {
    "JOB": "imdb_ori",
    "STACK": "so",
    "TPCH": "tpch",
}

MAX_QUERY_TIMEOUT_S = 360.0

SPLIT_PROTOCOLS = {
    "JOB": ("base_query", "leave_one_out", "random"),
    "STACK": ("base_query", "leave_one_out", "random"),
    "TPCH": ("random",),
}


def dynamic_timeout_s(pg_time_s: float, factor: float = 5.0) -> float:
    return min(MAX_QUERY_TIMEOUT_S, factor * pg_time_s)


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def load_splits() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Read the literal split definitions without executing train_test.py."""
    source = (ROOT / "workloads" / "train_test.py").read_text()
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "SPLITS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("SPLITS was not found in workloads/train_test.py")


def split_folds(workload: str, protocol: str) -> Dict[str, Dict[str, List[str]]]:
    workload = workload.upper()
    splits = load_splits()[workload]
    folds = {}
    for suffix in ("a", "b", "c"):
        key = "{}_{}".format(protocol, suffix)
        if key not in splits:
            raise KeyError("{} does not define {}".format(workload, key))
        folds[key] = splits[key]
    return folds


def workload_query_ids(workload: str) -> List[str]:
    workload = workload.upper()
    query_ids = set()
    for protocol in SPLIT_PROTOCOLS[workload]:
        for split in split_folds(workload, protocol).values():
            query_ids.update(split["train"])
            query_ids.update(split["test"])
    return sorted(query_ids, key=natural_key)


def query_path(workload: str, query_id: str) -> Path:
    workload = workload.upper()
    query_directories = {
        "JOB": "query_job",
        "STACK": "query_stack",
        "TPCH": "query_tpch",
    }
    if workload not in query_directories:
        raise KeyError("unknown workload {}".format(workload))
    path = (
        ROOT
        / "workloads"
        / query_directories[workload]
        / "{}.sql".format(query_id)
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def query_sql(workload: str, query_id: str) -> str:
    sql = query_path(workload, query_id).read_text()
    # Some canonical workload files also serve as saved hinted inputs for NQO.
    # Baseline systems must always start from the unhinted statement.
    return re.sub(r"/\*\+.*?\*/", "", sql, flags=re.DOTALL)


def sql_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def connect(
    workload: str,
    *,
    host: str = "localhost",
    port: int = 15432,
    user: str = "pgdb",
):
    connection = psycopg2.connect(
        host=host,
        port=port,
        dbname=WORKLOAD_DATABASES[workload.upper()],
        user=user,
    )
    connection.autocommit = True
    return connection


def reset_session(cursor, *, load_pg_hint_plan: bool = False) -> None:
    # A statement_timeout cancellation can arrive just after a completed query
    # and cancel the following RESET. The cancellation is consumed once, so
    # retry the session cleanup before running the next measured candidate.
    for attempt in range(2):
        try:
            cursor.execute("RESET ALL")
            break
        except psycopg2.errors.QueryCanceled:
            if attempt:
                raise
    cursor.execute("SET search_path TO public")
    if load_pg_hint_plan:
        cursor.execute("LOAD 'pg_hint_plan'")
        cursor.execute("SET pg_hint_plan.enable_hint TO on")
        cursor.execute("SET client_min_messages TO warning")


def set_timeout(cursor, timeout_s: Optional[float]) -> int:
    timeout_ms = 0 if timeout_s is None else max(1, int(math.ceil(1000.0 * timeout_s)))
    cursor.execute("SET statement_timeout = {}".format(timeout_ms))
    return timeout_ms


@dataclass
class ThreeRunResult:
    runtimes_s: List[Optional[float]]
    charged_runtimes_s: List[Optional[float]]
    errors: List[Optional[str]]
    timeout_s: Optional[float]

    @property
    def measured_s(self) -> Optional[float]:
        return self.charged_runtimes_s[2]

    @property
    def measured_error(self) -> Optional[str]:
        return self.errors[2]


@dataclass
class OneRunResult:
    runtime_s: Optional[float]
    charged_runtime_s: float
    error: Optional[str]
    timeout_s: Optional[float]


def execute_once(
    cursor,
    sql: str,
    *,
    timeout_s: Optional[float],
) -> OneRunResult:
    """Execute one SQL statement once, charging failures at the timeout."""
    set_timeout(cursor, timeout_s)
    started = time.perf_counter()
    try:
        cursor.execute(sql)
        runtime_s = time.perf_counter() - started
        return OneRunResult(runtime_s, runtime_s, None, timeout_s)
    except Exception as exc:
        if timeout_s is None:
            raise
        return OneRunResult(
            None,
            timeout_s,
            str(exc).replace("\n", " ")[:1000],
            timeout_s,
        )


def execute_three(
    cursor,
    sql: str,
    *,
    timeout_s: Optional[float],
    stop_after_first_timeout: bool = False,
) -> ThreeRunResult:
    """Execute one SQL statement three times and use the third result."""
    set_timeout(cursor, timeout_s)
    runtimes = []
    charged = []
    errors = []
    for run_index in range(3):
        started = time.perf_counter()
        try:
            cursor.execute(sql)
            runtime_s = time.perf_counter() - started
            runtimes.append(runtime_s)
            charged.append(runtime_s)
            errors.append(None)
        except Exception as exc:
            runtime_s = time.perf_counter() - started
            error = str(exc).replace("\n", " ")[:1000]
            runtimes.append(None)
            charged.append(timeout_s)
            errors.append(error)
            if (
                stop_after_first_timeout
                and run_index == 0
                and isinstance(exc, psycopg2.errors.QueryCanceled)
            ):
                runtimes.extend([None, None])
                charged.extend([timeout_s, timeout_s])
                errors.extend([error, error])
                break
    return ThreeRunResult(runtimes, charged, errors, timeout_s)


def result_to_row(result: ThreeRunResult) -> Dict[str, object]:
    row = {
        "timeout_s": "" if result.timeout_s is None else result.timeout_s,
        "measured_s": "" if result.measured_s is None else result.measured_s,
        "measured_error": result.measured_error or "",
    }
    for idx in range(3):
        row["run{}_s".format(idx + 1)] = (
            "" if result.runtimes_s[idx] is None else result.runtimes_s[idx]
        )
        row["run{}_charged_s".format(idx + 1)] = (
            ""
            if result.charged_runtimes_s[idx] is None
            else result.charged_runtimes_s[idx]
        )
        row["run{}_error".format(idx + 1)] = result.errors[idx] or ""
    return row


def one_result_to_row(result: OneRunResult) -> Dict[str, object]:
    """Store a one-run label measurement in the shared measurement schema."""
    row = {
        "timeout_s": "" if result.timeout_s is None else result.timeout_s,
        "run1_s": "" if result.runtime_s is None else result.runtime_s,
        "run1_charged_s": result.charged_runtime_s,
        "run1_error": result.error or "",
        "measured_s": result.charged_runtime_s,
        "measured_error": result.error or "",
    }
    for idx in (2, 3):
        row["run{}_s".format(idx)] = ""
        row["run{}_charged_s".format(idx)] = ""
        row["run{}_error".format(idx)] = ""
    return row


class CsvResultStore:
    """Append-only CSV store with query-level resume support."""

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed = self._load_completed()

    def _load_completed(self) -> Dict[str, Dict[str, str]]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return {}
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        completed = {}
        for row in rows:
            key = row.get("result_key", "")
            if key:
                completed[key] = row
        return completed

    def get(self, result_key: str) -> Optional[Dict[str, str]]:
        return self._completed.get(result_key)

    def append(self, row: Dict[str, object]) -> None:
        unexpected = set(row) - set(self.fieldnames)
        if unexpected:
            raise ValueError("unexpected CSV columns: {}".format(sorted(unexpected)))
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        self._completed[str(row["result_key"])] = {
            key: str(value) for key, value in row.items()
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PhaseTiming:
    """Checkpoint wall-clock timing for a resumable collection phase."""

    def __init__(
        self,
        path: Path,
        *,
        method: str,
        workload: str,
        phase: str,
        protocol: str,
        preexisting_records: int,
        database_execution_s_baseline: float,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity = {
            "method": method,
            "workload": workload,
            "phase": phase,
            "protocol": protocol,
        }
        if self.path.is_file():
            self.data = json.loads(self.path.read_text())
            for key, expected in identity.items():
                if self.data.get(key) != expected:
                    raise RuntimeError(
                        "{} timing identity mismatch for {}".format(key, self.path)
                    )
            for session in self.data.get("sessions", []):
                if session.get("status") == "running":
                    session["status"] = "interrupted"
            if "database_execution_s_baseline" not in self.data:
                self.data["database_execution_s_baseline"] = self.data.pop(
                    "preexisting_database_execution_s",
                    database_execution_s_baseline,
                )
        else:
            self.data = {
                **identity,
                "preexisting_records_before_timing": preexisting_records,
                "database_execution_s_baseline": database_execution_s_baseline,
                "sessions": [],
            }

        self.started = time.perf_counter()
        self.session = {
            "session_id": uuid.uuid4().hex,
            "started_at_utc": utc_now(),
            "last_checkpoint_at_utc": utc_now(),
            "wall_s": 0.0,
            "database_execution_s": 0.0,
            "new_records": 0,
            "resumed_records": 0,
            "status": "running",
        }
        self.data["sessions"].append(self.session)
        self._events_since_write = 0
        self._write()

    def _write(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)
        self._events_since_write = 0

    def record(self, *, resumed: bool, database_execution_s: float = 0.0) -> None:
        key = "resumed_records" if resumed else "new_records"
        self.session[key] += 1
        self.session["database_execution_s"] += database_execution_s
        self.session["wall_s"] = time.perf_counter() - self.started
        self.session["last_checkpoint_at_utc"] = utc_now()
        self._events_since_write += 1
        if not resumed or self._events_since_write >= 100:
            self._write()

    def finish(self, status: str) -> None:
        self.session["wall_s"] = time.perf_counter() - self.started
        self.session["last_checkpoint_at_utc"] = utc_now()
        self.session["finished_at_utc"] = utc_now()
        self.session["status"] = status
        self._write()


def measurement_database_execution_s(row: Dict[str, object]) -> float:
    total = 0.0
    for idx in range(1, 4):
        value = row.get("run{}_charged_s".format(idx))
        if value not in (None, ""):
            total += float(value)
    return total


def load_postgres_times(
    path: Path,
    time_column: str = "measured_s",
) -> Dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    run_prefix = time_column.split("_", 1)[0]
    error_column = (
        "{}_error".format(run_prefix)
        if re.fullmatch(r"run\d+", run_prefix)
        else "measured_error"
    )
    values = {}
    for row in rows:
        if row[error_column]:
            raise RuntimeError(
                "PostgreSQL baseline failed for {}: {}".format(
                    row["query_id"], row[error_column]
                )
            )
        values[row["query_id"]] = float(row[time_column])
    return values


MEASUREMENT_COLUMNS = [
    "timeout_s",
    "run1_s",
    "run1_charged_s",
    "run1_error",
    "run2_s",
    "run2_charged_s",
    "run2_error",
    "run3_s",
    "run3_charged_s",
    "run3_error",
    "measured_s",
    "measured_error",
]
