#!/usr/bin/env python3
"""Probe the PostgreSQL FK-Center implementation for complete workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psycopg2


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "workload_fk_center_analysis.json"
sys.path.insert(0, str(REPO_ROOT / "src"))

from optimization.query_compatibility import (  # noqa: E402
    classify_spj_compatibility,
)

WORKLOADS = {
    "JOB": (REPO_ROOT / "workloads" / "query_job", "imdb_ori"),
    "STACK": (REPO_ROOT / "workloads" / "query_stack", "so"),
    "TPCH": (REPO_ROOT / "workloads" / "query_tpch", "tpch"),
}
DEC_PHASES = {"dec", "high"}
SCHED_PHASES = {"sched", "select"}

def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value)
    ]


def load_sql(path: Path) -> str:
    sql = path.read_text()
    return re.sub(r"/\*\+.*?\*/", "", sql, flags=re.DOTALL)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def response_for_phase(phase: str, *, enumerate_centers: bool = False) -> bytes:
    if phase == "high":
        action = "split" if enumerate_centers else "stop"
        return (
            f"action={action}\n"
            f"stop={0 if enumerate_centers else 1}\n"
            f"high_action={action}\n"
            "order_decision=only_cost\n"
            "note=fk-center-probe\n"
        ).encode()
    if phase == "dec":
        action = "apply" if enumerate_centers else "skip"
        return (
            f"action={action}\n"
            f"stop={0 if enumerate_centers else 1}\n"
            f"dec_action={action}\n"
            "order_decision=only_cost\n"
            "note=fk-center-probe\n"
        ).encode()
    if phase in SCHED_PHASES:
        return (
            f"action={phase}\n"
            "stop=0\n"
            "candidate_id=0\n"
            "note=fk-center-probe\n"
        ).encode()
    return b"action=none\nstop=1\nnote=fk-center-probe\n"


@dataclass
class ProbeRecord:
    query_id: str
    sql_file: Path
    sql_sha256: str
    pid: int
    dec_state: dict[str, Any] | None = None
    sched_state: dict[str, Any] | None = None
    unexpected_states: list[dict[str, Any]] = field(default_factory=list)
    terminal_request: threading.Event = field(default_factory=threading.Event)
    release_response: threading.Event = field(default_factory=threading.Event)


class ProbeCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[int, ProbeRecord] = {}

    def register(self, record: ProbeRecord) -> None:
        with self._lock:
            self._records[record.pid] = record

    def unregister(self, pid: int) -> None:
        with self._lock:
            self._records.pop(pid, None)

    def record_state(self, state: dict[str, Any]) -> tuple[ProbeRecord | None, str]:
        pid = int(state.get("pid") or 0)
        phase = str(state.get("request_type") or "").lower()
        with self._lock:
            record = self._records.get(pid)
            if record is None:
                return None, phase
            if phase in DEC_PHASES:
                record.dec_state = state
                if int(state.get("remaining_splits") or 0) <= 1:
                    record.terminal_request.set()
            elif phase in SCHED_PHASES:
                record.sched_state = state
                record.terminal_request.set()
            else:
                record.unexpected_states.append(state)
                record.terminal_request.set()
            return record, phase


class ProbeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], coordinator: ProbeCoordinator):
        super().__init__(address, ProbeHandler)
        self.coordinator = coordinator


class ProbeHandler(BaseHTTPRequestHandler):
    server: ProbeHTTPServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            state = json.loads(self.rfile.read(length))
            record, phase = self.server.coordinator.record_state(state)
            enumerate_centers = (
                phase in DEC_PHASES
                and int(state.get("remaining_splits") or 0) > 1
            )
            if record is not None and not enumerate_centers:
                record.release_response.wait(timeout=30.0)
            body = response_for_phase(
                phase,
                enumerate_centers=enumerate_centers,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            body = f"action=none\nstop=1\nnote={type(exc).__name__}\n".encode()
            try:
                self.send_response(500)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass


def connect(args: argparse.Namespace, database: str):
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=database,
    )


def set_config(cursor: Any, name: str, value: object) -> None:
    cursor.execute("SELECT set_config(%s, %s, false)", (name, str(value)))


def configure_probe_session(
    cursor: Any,
    *,
    server_url: str,
    statement_timeout_ms: int,
) -> None:
    set_config(cursor, "search_path", "public")
    set_config(cursor, "client_min_messages", "warning")
    set_config(cursor, "statement_timeout", statement_timeout_ms)
    set_config(cursor, "nqo.server_timeout_ms", statement_timeout_ms)
    set_config(cursor, "nqo.server_url", server_url)
    set_config(cursor, "nqo.trajectory_log", "")
    set_config(cursor, "nqo.max_rounds", 1)
    set_config(cursor, "nqo", "on")


def cancel_backend(control_cursor: Any, pid: int) -> bool:
    control_cursor.execute("SELECT pg_cancel_backend(%s)", (pid,))
    return bool(control_cursor.fetchone()[0])


def compact_relations(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not state:
        return []
    return [
        {
            key: relation.get(key)
            for key in ("alias", "relname", "relid")
            if relation.get(key) is not None
        }
        for relation in state.get("relations", [])
    ]


def center_aliases(record: ProbeRecord) -> list[str] | None:
    if record.dec_state is None:
        return None
    if int(record.dec_state.get("remaining_splits") or 0) <= 1:
        return []
    if record.sched_state is None:
        return None
    relations = compact_relations(record.dec_state)
    centers = []
    for candidate in record.sched_state.get("candidates", []):
        center_index = int(candidate.get("center_index", -1))
        if 0 <= center_index < len(relations):
            center = relations[center_index].get("alias")
        else:
            center = f"rte_{center_index + 1}"
        if center and center not in centers:
            centers.append(str(center))
    return centers


def run_query_probe(
    args: argparse.Namespace,
    *,
    database: str,
    query_id: str,
    sql_file: Path,
    sql: str,
    spj: dict[str, Any],
    server_url: str,
    coordinator: ProbeCoordinator,
    control_cursor: Any,
) -> dict[str, Any]:
    connection = connect(args, database)
    connection.autocommit = True
    cursor = connection.cursor()
    configure_probe_session(
        cursor,
        server_url=server_url,
        statement_timeout_ms=args.statement_timeout_ms,
    )
    pid = connection.get_backend_pid()
    record = ProbeRecord(
        query_id=query_id,
        sql_file=sql_file,
        sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
        pid=pid,
    )
    coordinator.register(record)
    outcome: dict[str, str | None] = {"status": None, "error": None}
    finished = threading.Event()

    def execute() -> None:
        try:
            cursor.execute(sql)
            outcome["status"] = "completed"
        except psycopg2.errors.QueryCanceled:
            outcome["status"] = "cancelled"
        except Exception as exc:  # noqa: BLE001
            outcome["status"] = "error"
            outcome["error"] = str(exc).replace("\n", " ")[:500]
        finally:
            finished.set()

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    deadline = time.monotonic() + args.callback_timeout_ms / 1000.0
    while not record.terminal_request.is_set() and not finished.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        record.terminal_request.wait(min(0.05, remaining))

    cancellation_requested = False
    if not finished.is_set():
        cancellation_requested = cancel_backend(control_cursor, pid)
    record.release_response.set()
    worker.join(timeout=max(5.0, args.statement_timeout_ms / 1000.0 + 1.0))

    centers = center_aliases(record)
    if record.dec_state is None:
        probe_status = "no_dec_request"
    elif int(record.dec_state.get("remaining_splits") or 0) > 1 and centers is None:
        probe_status = "no_center_list"
    else:
        probe_status = "ok"
    result = {
        "query_id": query_id,
        "sql_file": relative_path(sql_file),
        "sql_sha256": record.sql_sha256,
        "probe_status": probe_status,
        "query_outcome": outcome["status"],
        "error": outcome["error"],
        "base_relation_count": (
            None
            if record.dec_state is None
            else int(record.dec_state.get("base_rels") or 0)
        ),
        "centers": centers,
        **spj,
        "relations": compact_relations(record.dec_state),
        "cancellation_requested": cancellation_requested,
    }
    coordinator.unregister(pid)
    cursor.close()
    connection.close()
    return result


def skipped_non_spj_result(
    *,
    query_id: str,
    sql_file: Path,
    sql: str,
    spj: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "sql_file": relative_path(sql_file),
        "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
        "probe_status": "skipped_non_spj",
        "query_outcome": "not_executed",
        "error": None,
        "base_relation_count": None,
        "centers": None,
        **spj,
        "relations": [],
        "cancellation_requested": False,
    }


def analyze_workload(
    args: argparse.Namespace,
    *,
    workload: str,
    sql_dir: Path,
    database: str,
    server_url: str,
    coordinator: ProbeCoordinator,
) -> dict[str, Any]:
    sql_files = sorted(sql_dir.glob("*.sql"), key=lambda path: natural_key(path.stem))
    if args.limit is not None:
        sql_files = sql_files[: args.limit]
    if not sql_files:
        raise FileNotFoundError(f"no SQL files found in {sql_dir}")
    control = None
    control_cursor = None
    queries = []
    try:
        for index, sql_file in enumerate(sql_files, start=1):
            sql = load_sql(sql_file)
            spj = classify_spj_compatibility(sql)
            if not spj["is_spj_compatible"]:
                result = skipped_non_spj_result(
                    query_id=sql_file.stem,
                    sql_file=sql_file,
                    sql=sql,
                    spj=spj,
                )
            else:
                if control is None:
                    control = connect(args, database)
                    control.autocommit = True
                    control_cursor = control.cursor()
                result = run_query_probe(
                    args,
                    database=database,
                    query_id=sql_file.stem,
                    sql_file=sql_file,
                    sql=sql,
                    spj=spj,
                    server_url=server_url,
                    coordinator=coordinator,
                    control_cursor=control_cursor,
                )
            queries.append(result)
            centers = result["centers"]
            label = "n/a" if centers is None else str(len(centers))
            print(
                f"[{workload} {index}/{len(sql_files)}] {sql_file.stem}: "
                f"centers={label} status={result['probe_status']}",
                flush=True,
            )
    finally:
        if control_cursor is not None:
            control_cursor.close()
        if control is not None:
            control.close()

    compatible_count = sum(query["is_spj_compatible"] for query in queries)
    compatible_ratio = compatible_count / len(queries)
    center_histogram = Counter(
        "not_applicable" if query["centers"] is None else len(query["centers"])
        for query in queries
    )
    return {
        "database": database,
        "sql_directory": relative_path(sql_dir),
        "query_count": len(queries),
        "policy_request_count": sum(
            query["centers"] is not None for query in queries
        ),
        "no_policy_request_count": sum(
            query["centers"] is None for query in queries
        ),
        "skipped_non_spj_query_count": sum(
            query["probe_status"] == "skipped_non_spj" for query in queries
        ),
        "spj_compatible_query_count": compatible_count,
        "spj_compatible_query_ratio": round(compatible_ratio, 6),
        "non_spj_compatible_query_count": len(queries) - compatible_count,
        "recommend_decomposition_enabled": compatible_count > 0,
        "center_count_histogram": {
            str(key): center_histogram[key]
            for key in sorted(center_histogram, key=lambda value: str(value))
        },
        "queries": {
            query["query_id"]: {
                "centers": query["centers"],
                "is_spj_compatible": query["is_spj_compatible"],
                "non_spj_features": query["non_spj_features"],
                "center_probe_status": query["probe_status"],
            }
            for query in queries
        },
        "probe_issues": {
            query["query_id"]: query["probe_status"]
            for query in queries
            if query["probe_status"] not in {"ok", "skipped_non_spj"}
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe NQO's PostgreSQL FK-Center implementation over workloads."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--password", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument(
        "--callback-host",
        default="172.17.0.1",
        help="probe host reachable from the PostgreSQL process",
    )
    parser.add_argument("--callback-timeout-ms", type=int, default=5000)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--workload",
        action="append",
        choices=tuple(WORKLOADS),
        help="workload to probe; repeat as needed (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--job-dir", type=Path, default=WORKLOADS["JOB"][0])
    parser.add_argument("--stack-dir", type=Path, default=WORKLOADS["STACK"][0])
    parser.add_argument("--tpch-dir", type=Path, default=WORKLOADS["TPCH"][0])
    parser.add_argument("--job-database", default=WORKLOADS["JOB"][1])
    parser.add_argument("--stack-database", default=WORKLOADS["STACK"][1])
    parser.add_argument("--tpch-database", default=WORKLOADS["TPCH"][1])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coordinator = ProbeCoordinator()
    server = ProbeHTTPServer((args.listen_host, args.listen_port), coordinator)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    server_port = int(server.server_address[1])
    server_url = f"http://{args.callback_host}:{server_port}/action"
    workload_inputs = {
        "JOB": (args.job_dir.resolve(), args.job_database),
        "STACK": (args.stack_dir.resolve(), args.stack_database),
        "TPCH": (args.tpch_dir.resolve(), args.tpch_database),
    }
    selected_workloads = set(args.workload or workload_inputs)
    try:
        workloads = {
            workload: analyze_workload(
                args,
                workload=workload,
                sql_dir=sql_dir,
                database=database,
                server_url=server_url,
                coordinator=coordinator,
            )
            for workload, (sql_dir, database) in workload_inputs.items()
            if workload in selected_workloads
        }
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)

    output = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "static QuerySplit compatibility filtering followed by direct "
            "PostgreSQL FK-Center enumeration"
        ),
        "postgresql": {
            "host": args.host,
            "port": args.port,
            "user": "pgdb",
        },
        "decision_rule": {
            "description": (
                "Expose query decomposition only to queries whose join core is "
                "classified as QuerySplit SPJ-compatible. FK-Center counts are "
                "reported for analysis but do not determine eligibility."
            ),
            "eligibility_field": "is_spj_compatible",
        },
        "field_semantics": {
            "queries": (
                "Each query records its PostgreSQL-generated FK-Center aliases "
                "when applicable and a static QuerySplit SPJ-compatibility "
                "classification. Incompatible SQL is not sent to PostgreSQL."
            ),
            "is_spj_compatible": (
                "True when the join core avoids structures unsafe for the "
                "current QuerySplit rewrite. Simple MIN(column), COUNT(*), and "
                "COUNT(DISTINCT column) result reducers are allowed."
            ),
            "non_spj_features": (
                "Syntactic reasons for classifying the SQL as incompatible; "
                "this classification does not require PostgreSQL."
            ),
            "empty_array": (
                "PostgreSQL found at most one raw center, so no intermediate "
                "decomposition center is exposed."
            ),
            "null": (
                "The query was excluded by the compatibility check or bypassed "
                "the NQO decomposition hook, so FK-Centers were not enumerated."
            ),
        },
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
