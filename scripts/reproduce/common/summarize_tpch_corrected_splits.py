#!/usr/bin/env python3
"""Summarize corrected-fold TPC-H learned baselines under two query sets."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TPCH_DIR = ROOT / "results" / "revision" / "tpch"
CORRECTED_DIR = TPCH_DIR / "corrected_splits"
EXCLUDED_QUERIES = {"7", "8", "9", "13", "15", "22"}
METHOD_ORDER = ("FASTgres", "TONIC", "GenJoin", "HybridQO", "AutoSteer")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def postgres_times() -> dict[str, float]:
    return {
        row["query_id"]: float(row["run1_charged_s"])
        for row in read_csv(TPCH_DIR / "postgres.csv")
    }


def single_record_method(
    method: str,
    path: Path,
    runtime_column: str,
    source: str,
) -> list[dict]:
    return [
        {
            "method": method,
            "query_id": row["query_id"],
            "method_runtime_s": float(row[runtime_column]),
            "runtime_source": row.get("runtime_source") or source,
            "model_runs": 1,
            "fallback": False,
        }
        for row in read_csv(path)
    ]


def genjoin_records() -> list[dict]:
    rows = read_csv(CORRECTED_DIR / "genjoin" / "resolved_predictions.csv")
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    result = []
    for query_id, query_rows in by_query.items():
        if len(query_rows) != 3:
            raise RuntimeError(f"GenJoin Q{query_id} has {len(query_rows)} model runs")
        runtimes = [float(row["resolved_runtime_s"]) for row in query_rows]
        sources = sorted({row["runtime_source"] for row in query_rows})
        result.append(
            {
                "method": "GenJoin",
                "query_id": query_id,
                "method_runtime_s": sum(runtimes) / len(runtimes),
                "runtime_source": ";".join(sources),
                "model_runs": len(runtimes),
                "fallback": all(
                    row["used_postgres_fallback"] == "True" for row in query_rows
                ),
            }
        )
    return result


def hybridqo_records() -> list[dict]:
    return [
        {
            "method": "HybridQO",
            "query_id": row["query_id"],
            "method_runtime_s": float(row["resolved_runtime_s"]),
            "runtime_source": row["runtime_source"],
            "model_runs": 1,
            "fallback": row["chosen_plan"] == "PG_fallback",
        }
        for row in read_csv(CORRECTED_DIR / "hybridqo" / "resolved_predictions.csv")
    ]


def per_query_records() -> list[dict]:
    pg = postgres_times()
    rows = []
    rows.extend(
        single_record_method(
            "FASTgres",
            CORRECTED_DIR / "fastgres" / "resolved_predictions.csv",
            "resolved_runtime_s",
            "resolved_label_or_standard_retest",
        )
    )
    rows.extend(
        single_record_method(
            "TONIC",
            CORRECTED_DIR / "tonic" / "resolved_predictions.csv",
            "resolved_runtime_s",
            "resolved_feedback_or_standard_retest",
        )
    )
    rows.extend(genjoin_records())
    rows.extend(hybridqo_records())
    rows.extend(
        single_record_method(
            "AutoSteer",
            CORRECTED_DIR / "autosteer" / "selected_experience.csv",
            "charged_runtime_s",
            "exact_candidate_experience",
        )
    )

    expected = {str(value) for value in range(1, 23)}
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        ids = [row["query_id"] for row in method_rows]
        if len(ids) != 22 or set(ids) != expected or len(ids) != len(set(ids)):
            raise RuntimeError(f"{method} does not have exactly one row per query")
    for row in rows:
        row["pg_runtime_s"] = pg[row["query_id"]]
        row["speedup"] = row["pg_runtime_s"] / row["method_runtime_s"]
        row["improved"] = row["method_runtime_s"] < row["pg_runtime_s"]
    return rows


def summarize(rows: list[dict], case: str, excluded: set[str]) -> list[dict]:
    summaries = []
    for method in METHOD_ORDER:
        selected = [
            row
            for row in rows
            if row["method"] == method and row["query_id"] not in excluded
        ]
        pg_total = sum(float(row["pg_runtime_s"]) for row in selected)
        method_total = sum(float(row["method_runtime_s"]) for row in selected)
        speedups = [float(row["speedup"]) for row in selected]
        summaries.append(
            {
                "case": case,
                "excluded_queries": ",".join(sorted(excluded, key=int)),
                "method": method,
                "query_count": len(selected),
                "fallback_queries": sum(bool(row["fallback"]) for row in selected),
                "pg_total_s": pg_total,
                "method_total_s": method_total,
                "WS": pg_total / method_total,
                "GS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
                "Imp_pct": 100.0
                * sum(bool(row["improved"]) for row in selected)
                / len(selected),
            }
        )
    return summaries


def main() -> None:
    rows = per_query_records()
    write_csv(CORRECTED_DIR / "resolved_per_query.csv", rows)
    summaries = [
        *summarize(rows, "all_22_with_pg_fallback", set()),
        *summarize(
            rows,
            "exclude_q7_q8_q9_q13_q15_q22",
            EXCLUDED_QUERIES,
        ),
    ]
    write_csv(CORRECTED_DIR / "summary_two_cases.csv", summaries)
    for row in summaries:
        print(
            "{case} {method}: n={query_count} WS={WS:.6f} GS={GS:.6f} "
            "Imp={Imp_pct:.3f}%".format(**row)
        )


if __name__ == "__main__":
    main()
