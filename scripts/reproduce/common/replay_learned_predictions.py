#!/usr/bin/env python3
"""Replay saved learned-baseline decisions with a shared three-run protocol."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from benchmarking.workloads import (
    MEASUREMENT_COLUMNS,
    CsvResultStore,
    connect,
    dynamic_timeout_s,
    execute_three,
    query_path,
    query_sql,
    reset_session,
    result_to_row,
    sql_sha256,
)
from scripts.reproduce.fastgres.measure_fastgres_labels import set_fastgres_hint
from scripts.reproduce.tonic.tonic_common import configure_tonic_session


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_RESULTS = ROOT / "results" / "benchmark"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "revision" / "unified_3run"
OTHER_BASELINES = (
    BENCHMARK_RESULTS
    / "genjoin_baselines"
    / "genjoin_hybridqo_autosteer_test_executions.csv"
)
HYBRIDQO_TPCH = (
    ROOT
    / "results"
    / "revision"
    / "tpch"
    / "corrected_splits"
    / "hybridqo"
    / "resolved_predictions.csv"
)
AUTOSTEER_TPCH = (
    ROOT
    / "results"
    / "revision"
    / "tpch"
    / "corrected_splits"
    / "autosteer"
    / "selected_experience.csv"
)
AUTOSTEER_KNOBS = (
    ROOT
    / "thrid_party"
    / "genjoin"
    / "OtherMethods_Repository"
    / "autosteer"
    / "knobs"
    / "postgres.txt"
)


RESULT_FIELDS = [
    "result_key",
    "method",
    "workload",
    "protocol",
    "fold",
    "query_id",
    "model_run_id",
    "decision",
    "sql_path",
    "sql_sha256",
    "inference_s",
    "pg_reference_s",
] + MEASUREMENT_COLUMNS


@dataclass(frozen=True)
class ReplayCase:
    method: str
    workload: str
    protocol: str
    fold: str
    query_id: str
    model_run_id: str
    decision: str
    inference_s: float
    fallback: bool = False

    @property
    def key(self) -> str:
        return ":".join(
            (
                self.method,
                self.workload,
                self.protocol,
                self.fold,
                self.query_id,
                self.model_run_id or "0",
            )
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def pg_times(path: Path, workload: str, run: int) -> dict[str, float]:
    rows = [row for row in read_csv(path) if row["workload"].upper() == workload]
    if not rows:
        raise RuntimeError(f"{path} has no {workload} PostgreSQL rows")
    second_columns = (f"run{run}_charged_s", f"run{run}_s")
    millisecond_columns = (f"run{run}_charged_ms", f"run{run}_ms")
    result = {}
    for row in rows:
        value = next((row.get(column) for column in second_columns if row.get(column)), None)
        scale = 1.0
        if value is None:
            value = next(
                (row.get(column) for column in millisecond_columns if row.get(column)),
                None,
            )
            scale = 0.001
        if value is None:
            raise RuntimeError(f"{path} has no run-{run} value for {row['query_id']}")
        result[row["query_id"]] = float(value) * scale
    return result


def fastgres_cases(workload: str) -> list[ReplayCase]:
    path = BENCHMARK_RESULTS / "fastgres" / workload.lower() / "predicted_hint_executions.csv"
    return [
        ReplayCase(
            method="fastgres",
            workload=workload,
            protocol=row["protocol"],
            fold=row["fold"],
            query_id=row["query_id"],
            model_run_id="",
            decision=row["predicted_hint"],
            inference_s=float(row["inference_s"]),
        )
        for row in read_csv(path)
    ]


def tonic_cases(workload: str) -> list[ReplayCase]:
    path = (
        BENCHMARK_RESULTS
        / "tonic"
        / workload.lower()
        / "predicted_join_assignment_executions.csv"
    )
    return [
        ReplayCase(
            method="tonic",
            workload=workload,
            protocol=row["protocol"],
            fold=row["fold"],
            query_id=row["query_id"],
            model_run_id="",
            decision=row["hint"],
            inference_s=float(row["inference_s"]),
        )
        for row in read_csv(path)
    ]


def genjoin_tpch_cases() -> list[ReplayCase]:
    return [
        ReplayCase(
            method="genjoin",
            workload="TPCH",
            protocol=row["protocol"],
            fold=row["fold"],
            query_id=row["query_id"],
            model_run_id=row["run_id"],
            decision=row["decision"],
            inference_s=float(row["inference_s"]),
            fallback=parse_bool(row["used_postgres_fallback"]),
        )
        for row in read_csv(OTHER_BASELINES)
        if row["method"] == "GenJoin" and row["workload"] == "TPCH"
    ]


def hybridqo_tpch_cases() -> list[ReplayCase]:
    return [
        ReplayCase(
            method="hybridqo",
            workload="TPCH",
            protocol="random",
            fold=row["fold"],
            query_id=row["query_id"],
            model_run_id="",
            decision=row["hint"],
            inference_s=float(row["inference_s"]),
            fallback=parse_bool(row["used_fallback"])
            or row["chosen_plan"].startswith("PG"),
        )
        for row in read_csv(HYBRIDQO_TPCH)
    ]


def autosteer_tpch_cases() -> list[ReplayCase]:
    return [
        ReplayCase(
            method="autosteer",
            workload="TPCH",
            protocol="random",
            fold=row["fold"],
            query_id=row["query_id"],
            model_run_id="",
            decision=row["predicted_config"],
            inference_s=float(row["inference_s"]),
        )
        for row in read_csv(AUTOSTEER_TPCH)
    ]


def load_cases(method: str, workload: str) -> list[ReplayCase]:
    if method == "fastgres":
        return fastgres_cases(workload)
    if method == "tonic":
        return tonic_cases(workload)
    if workload != "TPCH":
        raise ValueError(f"saved replay decisions for {method} are available only on TPCH")
    if method == "genjoin":
        return genjoin_tpch_cases()
    if method == "hybridqo":
        return hybridqo_tpch_cases()
    if method == "autosteer":
        return autosteer_tpch_cases()
    raise KeyError(method)


def hinted_sql(sql: str, hint: str) -> str:
    hint = hint.strip()
    if not hint:
        return sql
    if hint.startswith("/*+"):
        return f"{hint}\n{sql}"
    return f"/*+ {hint} */\n{sql}"


def autosteer_knobs() -> list[str]:
    return [
        line.strip()
        for line in AUTOSTEER_KNOBS.read_text().splitlines()
        if line.strip()
    ]


def set_autosteer_config(cursor, knobs: list[str], decision: str) -> None:
    disabled = set()
    if decision and decision != "None":
        disabled = {item.strip() for item in decision.split(",") if item.strip()}
    cursor.execute("RESET ALL")
    cursor.execute("SET search_path TO public")
    for knob in knobs:
        cursor.execute(f"SET {knob} TO {'off' if knob in disabled else 'on'}")


def execute_case(cursor, case: ReplayCase, timeout_s: float, knobs: list[str]):
    sql = query_sql(case.workload, case.query_id)
    if case.method == "fastgres":
        reset_session(cursor)
        set_fastgres_hint(cursor, int(case.decision))
    elif case.method == "tonic":
        configure_tonic_session(cursor)
        sql = hinted_sql(sql, case.decision)
    elif case.method in {"genjoin", "hybridqo"}:
        reset_session(cursor, load_pg_hint_plan=not case.fallback)
        if not case.fallback:
            sql = hinted_sql(sql, case.decision)
    elif case.method == "autosteer":
        set_autosteer_config(cursor, knobs, case.decision)
    else:
        raise KeyError(case.method)
    return execute_three(
        cursor,
        sql,
        timeout_s=timeout_s,
        stop_after_first_timeout=True,
    )


def validate_cases(cases: list[ReplayCase], baselines: dict[str, float]) -> None:
    if not cases:
        raise RuntimeError("no replay cases were loaded")
    keys = [case.key for case in cases]
    if len(keys) != len(set(keys)):
        raise RuntimeError("replay cases contain duplicate keys")
    missing = sorted({case.query_id for case in cases} - set(baselines))
    if missing:
        raise RuntimeError(f"PostgreSQL baseline is missing queries: {missing}")
    for case in cases:
        query_path(case.workload, case.query_id)


def run(args: argparse.Namespace) -> None:
    workload = args.workload.upper()
    cases = load_cases(args.method, workload)
    if args.protocol != "all":
        cases = [case for case in cases if case.protocol == args.protocol]
    baselines = pg_times(args.pg_csv, workload, args.pg_run)
    validate_cases(cases, baselines)
    counts = {}
    for case in cases:
        key = (case.protocol, case.fold)
        counts[key] = counts.get(key, 0) + 1
    print(
        f"validated method={args.method} workload={workload} "
        f"cases={len(cases)} pg_run={args.pg_run}"
    )
    for key, count in sorted(counts.items()):
        print(f"  protocol={key[0]} fold={key[1]} cases={count}")
    if args.dry_run:
        return

    output = args.output_root / args.method / f"{workload.lower()}_results.csv"
    store = CsvResultStore(output, RESULT_FIELDS)
    connection = connect(
        workload,
        host=args.host,
        port=args.port,
        user=args.user,
    )
    cursor = connection.cursor()
    knobs = autosteer_knobs() if args.method == "autosteer" else []
    try:
        for index, case in enumerate(cases, start=1):
            sql = query_sql(workload, case.query_id)
            digest = sql_sha256(sql)
            baseline = baselines[case.query_id]
            timeout_s = dynamic_timeout_s(baseline, args.timeout_factor)
            previous = store.get(case.key)
            if (
                previous
                and previous.get("sql_sha256") == digest
                and previous.get("decision", "") == case.decision
                and abs(float(previous["timeout_s"]) - timeout_s) < 1e-9
            ):
                print(
                    f"[{index}/{len(cases)}] {case.key}: resume "
                    f"{previous['measured_s']}s",
                    flush=True,
                )
                continue
            result = execute_case(cursor, case, timeout_s, knobs)
            row = {
                "result_key": case.key,
                "method": case.method,
                "workload": workload,
                "protocol": case.protocol,
                "fold": case.fold,
                "query_id": case.query_id,
                "model_run_id": case.model_run_id,
                "decision": case.decision,
                "sql_path": str(query_path(workload, case.query_id).relative_to(ROOT)),
                "sql_sha256": digest,
                "inference_s": case.inference_s,
                "pg_reference_s": baseline,
            }
            row.update(result_to_row(result))
            store.append(row)
            print(
                f"[{index}/{len(cases)}] {case.key}: "
                f"runs={','.join(str(value) for value in result.charged_runtimes_s)}",
                flush=True,
            )
    finally:
        cursor.close()
        connection.close()
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=("fastgres", "tonic", "genjoin", "hybridqo", "autosteer"),
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=("JOB", "STACK", "TPCH"),
    )
    parser.add_argument(
        "--protocol",
        choices=("all", "base_query", "leave_one_out", "random"),
        default="all",
    )
    parser.add_argument("--pg-csv", type=Path, required=True)
    parser.add_argument("--pg-run", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--timeout-factor", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.pg_csv = args.pg_csv.resolve()
    args.output_root = args.output_root.resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
