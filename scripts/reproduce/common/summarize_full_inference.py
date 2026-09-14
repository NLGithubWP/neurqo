#!/usr/bin/env python3
"""Validate full-inference measurements and recompute affected metrics."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
REVISION = ROOT / "results" / "revision"
INFERENCE = REVISION / "inference_full"
TPCH_NORMALIZED = (
    REVISION
    / "tpch"
    / "corrected_splits"
    / "resolved_per_query_pg_20260726_collection_default_normalized.csv"
)
BENCHMARK_BASELINES = (
    ROOT
    / "results"
    / "benchmark"
    / "genjoin_baselines"
    / "genjoin_hybridqo_autosteer_test_executions.csv"
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def inference_summary() -> list[dict]:
    rows = []
    for path in sorted(INFERENCE.glob("*/*.csv")):
        records = read_csv(path)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            grouped[record["protocol"]].append(record)
        for protocol, selected in sorted(grouped.items()):
            values = [float(record["inference_s"]) for record in selected]
            rows.append(
                {
                    "method": selected[0]["method"],
                    "workload": selected[0]["workload"],
                    "protocol": protocol,
                    "records": len(selected),
                    "unique_queries": len({record["query_id"] for record in selected}),
                    "model_runs": len(
                        {
                            record.get("run_id", "")
                            for record in selected
                            if record.get("run_id", "") != ""
                        }
                    )
                    or 1,
                    "total_inference_s": sum(values),
                    "mean_inference_s": statistics.mean(values),
                    "median_inference_s": statistics.median(values),
                    "p95_inference_s": percentile(values, 0.95),
                    "max_inference_s": max(values),
                    "prediction_mismatches": sum(
                        record.get("prediction_match", "").lower() != "true"
                        for record in selected
                    ),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return rows


def metric_row(
    method: str,
    workload: str,
    protocol: str,
    timeout_policy: str,
    records: Iterable[Mapping[str, float | str]],
    source: str,
    include_inference: bool = True,
) -> dict:
    selected = list(records)
    pg = [float(record["pg_runtime_s"]) for record in selected]
    execution = [float(record["execution_runtime_s"]) for record in selected]
    measured_inference = [float(record["inference_s"]) for record in selected]
    inference = measured_inference if include_inference else [0.0] * len(selected)
    method_times = [value + overhead for value, overhead in zip(execution, inference)]
    speedups = [base / method_time for base, method_time in zip(pg, method_times)]
    return {
        "method": method,
        "workload": workload,
        "protocol": protocol,
        "timeout_policy": timeout_policy,
        "query_count": len(selected),
        "inference_accounting": "included" if include_inference else "excluded",
        "pg_total_s": sum(pg),
        "execution_total_s": sum(execution),
        "measured_inference_total_s": sum(measured_inference),
        "inference_total_s": sum(inference),
        "end_to_end_total_s": sum(method_times),
        "WS": sum(pg) / sum(method_times),
        "GS": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "Imp_pct": 100.0
        * sum(method_time < base for base, method_time in zip(pg, method_times))
        / len(selected),
        "source": source,
    }


def fastgres_tonic_metrics(include_inference: bool = True) -> list[dict]:
    summaries = []
    for method in ("fastgres", "tonic"):
        display = "FASTgres" if method == "fastgres" else "TONIC"
        for workload in ("job", "stack"):
            pg = {
                row["query_id"]: float(row["run1_charged_s"])
                for row in read_csv(REVISION / workload / "postgres.csv")
            }
            inference = {
                (row["protocol"], row["fold"], row["query_id"]): float(
                    row["inference_s"]
                )
                for row in read_csv(INFERENCE / method / f"{workload}.csv")
            }
            method_dir = (
                "tonic_original"
                if method == "tonic" and workload == "job"
                else method
            )
            for protocol in ("base_query", "leave_one_out", "random"):
                source_path = REVISION / workload / method_dir / f"{protocol}_results.csv"
                base_records = []
                for row in read_csv(source_path):
                    query_id = row["query_id"]
                    base_records.append(
                        {
                            "pg_runtime_s": pg[query_id],
                            "execution_runtime_s": float(row["run1_charged_s"]),
                            "inference_s": inference[
                                (protocol, row["fold"], query_id)
                            ],
                        }
                    )
                source = str(source_path.relative_to(ROOT))
                summaries.append(
                    metric_row(
                        display,
                        workload.upper(),
                        protocol,
                        "5xPG_recorded",
                        base_records,
                        source,
                        include_inference,
                    )
                )
                if method == "tonic":
                    censored = [
                        {
                            **record,
                            "execution_runtime_s": min(
                                float(record["execution_runtime_s"]),
                                2.0 * float(record["pg_runtime_s"]),
                            ),
                        }
                        for record in base_records
                    ]
                    summaries.append(
                        metric_row(
                            display,
                            workload.upper(),
                            protocol,
                            "2xPG_posthoc_censored",
                            censored,
                            source,
                            include_inference,
                        )
                    )
    return summaries


def benchmark_job_stack_metrics(include_inference: bool = True) -> list[dict]:
    rows = read_csv(BENCHMARK_BASELINES)
    summaries = []
    for workload in ("JOB", "STACK"):
        for protocol in ("base_query", "leave_one_out", "random"):
            for method in ("GenJoin", "HybridQO", "AutoSteer"):
                selected = [
                    row
                    for row in rows
                    if row["workload"] == workload
                    and row["protocol"] == protocol
                    and row["method"] == method
                ]
                records = [
                    {
                        "pg_runtime_s": float(row["postgres_first_s"]),
                        "execution_runtime_s": float(row["execution_first_s"]),
                        "inference_s": float(row["inference_s"]),
                    }
                    for row in selected
                ]
                summaries.append(
                    metric_row(
                        method,
                        workload,
                        protocol,
                        "5xPG_recorded_first_run",
                        records,
                        str(BENCHMARK_BASELINES.relative_to(ROOT)),
                        include_inference,
                    )
                )
    return summaries


def tpch_inference_maps() -> dict[str, dict[str, float]]:
    result = {}
    for directory, method in (
        ("fastgres", "FASTgres"),
        ("tonic", "TONIC"),
        ("autosteer", "AutoSteer"),
    ):
        result[method] = {
            row["query_id"]: float(row["inference_s"])
            for row in read_csv(INFERENCE / directory / "tpch.csv")
        }

    genjoin: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(INFERENCE / "genjoin" / "tpch.csv"):
        genjoin[row["query_id"]].append(float(row["inference_s"]))
    result["GenJoin"] = {
        query_id: statistics.mean(values) for query_id, values in genjoin.items()
    }
    result["HybridQO"] = {
        row["query_id"]: float(row["inference_s"])
        for row in read_csv(
            REVISION
            / "tpch"
            / "corrected_splits"
            / "hybridqo"
            / "resolved_predictions.csv"
        )
    }
    return result


def tpch_metrics(include_inference: bool = True) -> list[dict]:
    source_rows = read_csv(TPCH_NORMALIZED)
    by_method: dict[str, list[dict]] = defaultdict(list)
    inference = tpch_inference_maps()
    for row in source_rows:
        method = row["method"]
        by_method[method].append(
            {
                "pg_runtime_s": float(row["pg_runtime_s"]),
                "execution_runtime_s": float(row["method_runtime_s"]),
                "inference_s": inference[method][row["query_id"]],
            }
        )
    return [
        metric_row(
            method,
            "TPCH",
            "random",
            "5xPG_recorded_with_PG_fallback",
            records,
            str(TPCH_NORMALIZED.relative_to(ROOT)),
            include_inference,
        )
        for method, records in sorted(by_method.items())
    ]


def main() -> None:
    INFERENCE.mkdir(parents=True, exist_ok=True)
    measurements = inference_summary()
    all_metrics = [
        *fastgres_tonic_metrics(),
        *benchmark_job_stack_metrics(),
        *tpch_metrics(),
    ]
    all_metrics_without_inference = [
        *fastgres_tonic_metrics(include_inference=False),
        *benchmark_job_stack_metrics(include_inference=False),
        *tpch_metrics(include_inference=False),
    ]
    metrics_official = [
        row
        for row in all_metrics
        if not (
            row["method"] == "TONIC"
            and row["workload"] in {"JOB", "STACK"}
            and row["timeout_policy"] == "2xPG_posthoc_censored"
        )
    ]
    metrics_5x = [
        row
        for row in metrics_official
        if str(row["timeout_policy"]).startswith("5xPG")
    ]
    tonic_2x = [
        row
        for row in all_metrics
        if row["method"] == "TONIC"
        and row["workload"] in {"JOB", "STACK"}
        and row["timeout_policy"] == "2xPG_posthoc_censored"
    ]
    metrics_without_inference = [
        row
        for row in all_metrics_without_inference
        if not (
            row["method"] == "TONIC"
            and row["workload"] in {"JOB", "STACK"}
            and row["timeout_policy"] == "2xPG_posthoc_censored"
        )
    ]
    write_csv(INFERENCE / "inference_summary.csv", measurements)
    write_csv(INFERENCE / "metrics_with_inference.csv", metrics_official)
    write_csv(INFERENCE / "metrics_with_inference_official.csv", metrics_official)
    write_csv(INFERENCE / "metrics_with_inference_5x.csv", metrics_5x)
    write_csv(INFERENCE / "tonic_2x_with_inference.csv", tonic_2x)
    write_csv(
        INFERENCE / "metrics_without_inference_official.csv",
        metrics_without_inference,
    )
    for row in metrics_official:
        print(
            "{method} {workload} {protocol} {timeout_policy}: "
            "WS={WS:.6f} GS={GS:.6f} Imp={Imp_pct:.3f}% "
            "inference={inference_total_s:.3f}s".format(**row)
        )


if __name__ == "__main__":
    main()
