#!/usr/bin/env python3
"""Build the unified first-run Overall performance comparison table."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BENCHMARK_RESULTS = REPO / "results" / "benchmark"
NQO_RUNS = BENCHMARK_RESULTS / "nqo" / "nqo_runs.csv"
DEFAULT_OUTPUT = BENCHMARK_RESULTS / "overall_performance_comparison.csv"

COLUMNS = (
    ("JOB Base-query", "JOB", "base_query"),
    ("JOB Leave-one-out", "JOB", "leave_one_out"),
    ("JOB Random", "JOB", "random"),
    ("STACK Base-query", "STACK", "base_query"),
    ("STACK Leave-one-out", "STACK", "leave_one_out"),
    ("STACK Random", "STACK", "random"),
    ("TPC-H Random", "TPCH", "random"),
)
WORKLOADS = ("JOB", "STACK", "TPCH")
PROTOCOLS = ("base_query", "leave_one_out", "random")
LEARNED = ("FASTgres", "TONIC", "GenJoin", "HybridQO", "AutoSteer")
NON_LEARNED = (
    "QuerySplit",
    "LIP (Sel.)",
    "LIP (Full)",
    "AJA (Cons.)",
    "AJA (Aggr.)",
    "TOP-5 (DP)",
    "TOP-10 (DP)",
)


@dataclass(frozen=True)
class Summary:
    ws: float
    gs: float
    improved: int
    query_count: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def summarize(records: list[tuple[float, float]]) -> Summary:
    if not records:
        raise ValueError("cannot summarize an empty record set")
    ws = sum(pg for pg, _ in records) / sum(runtime for _, runtime in records)
    gs = math.exp(
        sum(math.log(pg / runtime) for pg, runtime in records) / len(records)
    )
    improved = sum(runtime < pg for pg, runtime in records)
    return Summary(ws, gs, improved, len(records))


def format_summary(summary: Summary | None) -> str:
    if summary is None:
        return "—"
    pct = 100.0 * summary.improved / summary.query_count
    return (
        f"{summary.ws:.6f} / {summary.gs:.6f} / "
        f"{summary.improved}/{summary.query_count} ({pct:.2f}%)"
    )


def normalized_learned_record(
    method: str,
    workload: str,
    row: dict[str, str],
) -> tuple[float, float, float]:
    pg = float(row["postgres_first_s"])
    execution = float(row["execution_first_s"])

    if method == "FASTgres" and workload == "TPCH" and row.get("predicted_hint") == "63":
        execution = pg
    if as_bool(row.get("used_postgres_fallback", "False")):
        execution = pg
    if method == "AutoSteer" and as_bool(
        row.get("selected_postgres_default", "False")
    ):
        execution = pg

    execution = min(execution, 5.0 * pg, 360.0)
    return pg, execution, float(row["inference_s"])


def learned_summaries() -> dict[tuple[str, str, str, bool], Summary]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    specs = (
        ("FASTgres", "fastgres", "predicted_hint_executions.csv"),
        ("TONIC", "tonic", "predicted_join_assignment_executions.csv"),
    )
    for method, directory, filename in specs:
        for workload in WORKLOADS:
            for row in read_csv(BENCHMARK_RESULTS / directory / workload.lower() / filename):
                protocol = row["protocol"]
                grouped.setdefault((method, workload, protocol), []).append(
                    normalized_learned_record(method, workload, row)
                )

    for row in read_csv(
        BENCHMARK_RESULTS
        / "genjoin_baselines"
        / "genjoin_hybridqo_autosteer_test_executions.csv"
    ):
        method = row["method"]
        workload = row["workload"]
        protocol = row["protocol"]
        grouped.setdefault((method, workload, protocol), []).append(
            normalized_learned_record(method, workload, row)
        )

    result = {}
    for (method, workload, protocol), records in grouped.items():
        for without_inference in (False, True):
            pairs = [
                (
                    pg,
                    max(0.000001, execution + (0.0 if without_inference else inference)),
                )
                for pg, execution, inference in records
            ]
            result[(method, workload, protocol, without_inference)] = summarize(pairs)
    return result


def nqo_summaries() -> dict[tuple[str, str, str, bool], Summary]:
    rows = read_csv(NQO_RUNS)
    required = {
        "dataset",
        "protocol",
        "fold",
        "sql_path",
        "method",
        "runtime_ms",
        "inference_ms",
        "status",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise RuntimeError(f"missing NQO CSV columns: {sorted(missing)}")

    result_keys = [
        (
            row["dataset"],
            row["protocol"],
            row["fold"],
            row["sql_path"],
            row["method"],
        )
        for row in rows
    ]
    if len(result_keys) != len(set(result_keys)):
        raise RuntimeError("nqo_runs.csv contains duplicate result keys")

    pg = {}
    for row in rows:
        if row["method"] != "PostgreSQL":
            continue
        key = (row["dataset"], row["sql_path"])
        if key in pg:
            raise RuntimeError(f"duplicate PostgreSQL baseline: {key}")
        pg[key] = float(row["runtime_ms"])

    result = {}
    for method in (*NON_LEARNED, "NQO"):
        for _, workload, protocol in COLUMNS:
            if method in NON_LEARNED:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == workload
                    and row["method"] == method
                    and row["protocol"] == protocol
                ]
                if not selected:
                    selected = [
                        row
                        for row in rows
                        if row["dataset"] == workload
                        and row["method"] == method
                        and not row["protocol"]
                    ]
            else:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == workload
                    and row["protocol"] == protocol
                    and row["method"] == method
                ]
            if not selected:
                continue
            variants = (False, True) if method == "NQO" else (False,)
            for without_inference in variants:
                pairs = []
                for row in selected:
                    key = (workload, row["sql_path"])
                    if key not in pg:
                        raise RuntimeError(f"missing PostgreSQL baseline: {key}")
                    baseline = pg[key]
                    runtime = float(row["runtime_ms"])
                    if without_inference and row["status"] == "ok":
                        runtime = max(0.001, runtime - float(row["inference_ms"]))
                    pairs.append((baseline, runtime))
                result[(method, workload, protocol, without_inference)] = summarize(
                    pairs
                )
    return result


def comparison_rows() -> list[dict[str, str]]:
    learned = learned_summaries()
    nqo = nqo_summaries()
    output: list[dict[str, str]] = []

    def append_row(
        category: str,
        label: str,
        method: str,
        without_inference: bool,
        source: dict[tuple[str, str, str, bool], Summary],
    ) -> None:
        row = {"Category": category, "Method": label}
        for column, workload, protocol in COLUMNS:
            row[column] = format_summary(
                source.get((method, workload, protocol, without_inference))
            )
        output.append(row)

    for method in LEARNED:
        append_row("Learned", method, method, False, learned)
        append_row(
            "Learned",
            f"{method} w/o inference time",
            method,
            True,
            learned,
        )
    for method in NON_LEARNED:
        append_row("Non-learned", method, method, False, nqo)
    append_row("NQO", "NQO", "NQO", False, nqo)
    append_row("NQO", "NQO w/o inference time", "NQO", True, nqo)
    return output


def write_comparison(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Category", "Method", *(column for column, _, _ in COLUMNS)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, str]]) -> str:
    fields = ["Category", "Method", *(column for column, _, _ in COLUMNS)]
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |")
    return "\n".join(lines) + "\n"


def update_record(path: Path, table: str) -> None:
    text = path.read_text()
    start_marker = "# Overall performance"
    end_marker = "# 1. Learning efficiency"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    intro = (
        "# Overall performance\n\n"
        "This table is generated from "
        "`results/benchmark/overall_performance_comparison.csv`. "
        "Each cell reports `WS / GS / Imp` under the `first-vs-first` protocol. "
        "Non-learned methods are independent of the train-test split, so the "
        "same result is repeated for all three JOB/STACK protocols; QuerySplit "
        "does not apply to TPC-H.\n\n"
    )
    path.write_text(text[:start] + intro + table + "\n" + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument(
        "--update-record",
        type=Path,
        help="replace the Overall performance section in this Markdown file",
    )
    args = parser.parse_args()

    rows = comparison_rows()
    write_comparison(args.output, rows)
    table = render_markdown(rows)
    if args.update_record:
        update_record(args.update_record, table)
        print(f"updated {args.update_record}")
    if args.print_markdown:
        print(table, end="")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
