"""PostgreSQL execution primitives shared by benchmark workflows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import psycopg2

from benchmarking.workloads import WORKLOAD_DATABASES
from database.catalog import read_postgres_catalog, write_catalog_snapshot
from optimization.actions import ActionProfile, hash_result_rows


def connect(args: argparse.Namespace):
    connection = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        dbname=WORKLOAD_DATABASES[args.workload],
    )
    connection.autocommit = True
    return connection


def create_catalog_snapshot(args: argparse.Namespace, output: Path) -> str:
    """Capture the current database catalog once for model-state encoding."""
    connection = connect(args)
    try:
        snapshot = read_postgres_catalog(connection, schema="public")
    finally:
        connection.close()
    return write_catalog_snapshot(output, snapshot)


def prewarm_database(args: argparse.Namespace) -> dict[str, Any]:
    connection = connect(args)
    cursor = connection.cursor()
    started = time.perf_counter()
    try:
        cursor.execute(
            """
            SELECT c.oid::regclass::text
            FROM pg_class AS c
            JOIN pg_namespace AS n ON n.oid=c.relnamespace
            WHERE n.nspname='public'
              AND c.relkind IN ('r', 'i', 'm')
              AND c.relpersistence <> 't'
            ORDER BY c.relpages DESC, c.oid
            """
        )
        relations = [row[0] for row in cursor.fetchall()]
        blocks = 0
        for relation in relations:
            cursor.execute(
                "SELECT pg_prewarm(%s::regclass, 'buffer')",
                (relation,),
            )
            blocks += int(cursor.fetchone()[0] or 0)
    finally:
        cursor.close()
        connection.close()
    return {
        "enabled": True,
        "mode": "buffer",
        "relations": len(relations),
        "blocks": blocks,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }

def set_config(cursor, name: str, value: Any) -> None:
    cursor.execute("SELECT set_config(%s, %s, false)", (name, str(value)))


def run_query_once(
    args: argparse.Namespace,
    *,
    sql: str,
    profile: ActionProfile,
    timeout_ms: int,
    db_trace_container: str,
    server_url: Optional[str],
) -> dict[str, Any]:
    connection = connect(args)
    cursor = connection.cursor()
    started = None
    status = "ok"
    error = None
    result_hash = None
    result_rows = None
    try:
        set_config(cursor, "statement_timeout", timeout_ms)
        set_config(cursor, "client_min_messages", "warning")
        set_config(cursor, "search_path", "public")
        if profile.nqo_enabled:
            if server_url is None:
                raise RuntimeError("NQO profile requires a policy server")
            for name, value in profile.guc_settings().items():
                set_config(cursor, name, value)
            set_config(cursor, "nqo.server_url", server_url)
            set_config(cursor, "nqo.trajectory_log", db_trace_container)
            set_config(cursor, "nqo", "on")
        else:
            set_config(cursor, "nqo", "off")
        started = time.perf_counter()
        cursor.execute(sql)
        if cursor.description is None:
            result_hash, result_rows = hash_result_rows([])
        else:
            result_hash, result_rows = hash_result_rows(cursor)
    except psycopg2.errors.QueryCanceled as exc:
        status = "timeout"
        error = str(exc).replace("\n", " ")[:1000]
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = repr(exc)[:1000]
    finally:
        wall_ms = (
            (time.perf_counter() - started) * 1000.0 if started is not None else 0.0
        )
        cursor.close()
        connection.close()
    return {
        "status": status,
        "error": error,
        "client_wall_ms": wall_ms,
        "charged_wall_ms": wall_ms,
        "result_hash": result_hash,
        "result_rows": result_rows,
    }
