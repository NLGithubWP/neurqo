#!/usr/bin/env python3
"""Measure full test-time inference without executing the selected SQL.

The timed region starts from an unseen test query and includes query-specific
parsing/encoding, auxiliary EXPLAIN calls, model prediction, and hint creation.
Fold training, model loading, and fold-independent schema metadata are excluded.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "revision"
DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "inference_full"

sys.path.insert(0, str(SCRIPT_DIR))

from benchmarking.workloads import (  # noqa: E402
    SPLIT_PROTOCOLS,
    connect,
    natural_key,
    query_sql,
    split_folds,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty result file {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_expected_predictions(method: str, workload: str, protocol: str) -> dict:
    method = method.lower()
    workload = workload.upper()
    if workload == "TPCH":
        path = (
            RESULT_ROOT
            / "tpch"
            / "corrected_splits"
            / method
            / f"{protocol}_predictions.json"
        )
    else:
        method_dir = "tonic_original" if method == "tonic" and workload == "JOB" else method
        path = RESULT_ROOT / workload.lower() / method_dir / f"{protocol}_predictions.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("predictions", {})


def selected_protocols(workload: str) -> Sequence[str]:
    return SPLIT_PROTOCOLS[workload.upper()]


def limit_test_ids(test_ids: Iterable[str], limit: int | None) -> list[str]:
    values = list(test_ids)
    return values if limit is None else values[:limit]


def train_fastgres_models(query_objects, features, archive, fold_spec):
    from sklearn.ensemble import GradientBoostingClassifier

    training_names = [f"{query_id}.sql" for query_id in fold_spec["train"]]
    by_context: Dict[object, list[str]] = defaultdict(list)
    for name in training_names:
        by_context[query_objects[name].context].append(name)

    models = {}
    for context, names in by_context.items():
        labels = [int(archive[name]["opt"]) for name in names]
        unique = np.unique(labels)
        if len(unique) == 1:
            models[context] = int(unique[0])
            continue
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=1000,
            random_state=29,
        )
        model.fit([features[name] for name in names], labels)
        models[context] = model
    return models


def new_fastgres_query(fastgres, workload, query_id, query_dir, type_dict, native_query):
    name = f"{query_id}.sql"
    if workload == "TPCH":
        from mo_sql_parsing import parse

        return fastgres.GenericQuery(
            name,
            parse((query_dir / name).read_text()),
            type_dict,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        return native_query(name, str(query_dir) + "/")


def measure_fastgres(args: argparse.Namespace) -> list[dict]:
    from scripts.reproduce.fastgres import run_fastgres_full as fastgres

    workload = args.workload
    utility, featurize, native_query = fastgres.import_fastgres()
    query_dir = RESULT_ROOT / workload.lower() / "fastgres" / "queries"
    connection = connect(workload, host=args.host, port=args.port, user=args.user)
    try:
        type_dict = fastgres.catalog_types(connection)
        query_objects = fastgres.build_query_objects(
            workload, query_dir, type_dict, native_query
        )
        if workload == "JOB":
            job_assets = (
                ROOT
                / "artifacts"
                / "baseline_caches"
                / "rebuttal"
                / "r1_o2_baselines"
                / "fastgres_job"
            )
            minmax = utility.load_pickle(str(job_assets / "mm_dict.pkl"))
            encoders = utility.load_pickle(str(job_assets / "label_encoders.pkl"))
            wildcards = utility.load_json(
                str(fastgres.FASTGRES_ROOT / "db_info" / "imdb" / "wildcard_dict.json")
            )
            skipped = {}
        else:
            minmax, encoders, wildcards, skipped = fastgres.load_or_build_metadata(
                workload,
                connection,
                query_objects,
                type_dict,
                utility,
                RESULT_ROOT,
            )
    finally:
        connection.close()

    dbname = {"JOB": "imdb_ori", "STACK": "so", "TPCH": "tpch"}[workload]
    connection_string = (
        f"dbname={dbname} user={args.user} host={args.host} port={args.port}"
    )
    features = fastgres.build_features(
        query_objects,
        connection_string,
        type_dict,
        minmax,
        encoders,
        wildcards,
        skipped,
        utility,
        featurize,
    )
    archive = json.loads(
        (RESULT_ROOT / workload.lower() / "fastgres" / "archive.json").read_text()
    )

    rows: list[dict] = []
    for protocol in selected_protocols(workload):
        expected = load_expected_predictions("fastgres", workload, protocol)
        for fold, spec in split_folds(workload, protocol).items():
            models = train_fastgres_models(query_objects, features, archive, spec)
            for query_id in limit_test_ids(spec["test"], args.limit):
                gc.collect()
                total_started = time.perf_counter()
                parsing_started = time.perf_counter()
                query = new_fastgres_query(
                    fastgres,
                    workload,
                    query_id,
                    query_dir,
                    type_dict,
                    native_query,
                )
                parsing_s = time.perf_counter() - parsing_started

                feature_started = time.perf_counter()
                name = f"{query_id}.sql"
                feature = fastgres.build_features(
                    {name: query},
                    connection_string,
                    type_dict,
                    minmax,
                    encoders,
                    wildcards,
                    skipped,
                    utility,
                    featurize,
                )[name]
                encoding_s = time.perf_counter() - feature_started

                prediction_started = time.perf_counter()
                model = models.get(query.context)
                if model is None:
                    prediction = 63
                elif isinstance(model, int):
                    prediction = model
                else:
                    prediction = int(model.predict([feature])[0])
                model_s = time.perf_counter() - prediction_started

                post_started = time.perf_counter()
                hint_bits = format(prediction, "06b")
                postprocessing_s = time.perf_counter() - post_started
                inference_s = time.perf_counter() - total_started
                expected_prediction = expected.get(fold, {}).get(query_id)
                rows.append(
                    {
                        "method": "FASTgres",
                        "workload": workload,
                        "protocol": protocol,
                        "fold": fold,
                        "query_id": query_id,
                        "parsing_s": parsing_s,
                        "encoding_or_explain_s": encoding_s,
                        "model_s": model_s,
                        "postprocessing_s": postprocessing_s,
                        "inference_s": inference_s,
                        "prediction": prediction,
                        "hint": hint_bits,
                        "expected_prediction": "" if expected_prediction is None else expected_prediction,
                        "prediction_match": expected_prediction is None
                        or int(expected_prediction) == prediction,
                    }
                )
                print(
                    f"FASTgres {workload} {fold} {query_id}: "
                    f"{inference_s:.6f}s match={rows[-1]['prediction_match']}",
                    flush=True,
                )
    return rows


def train_tonic_root(tonic, runner, workload, spec, skeletons, feedback_path, source):
    if source == "original_job":
        feedback = runner.load_original_job_training_feedback(spec["train"], skeletons)
    else:
        feedback = runner.load_training_feedback(feedback_path, spec["train"])
    root = tonic.QepsNode()
    for query_id in sorted(spec["train"], key=natural_key):
        skeleton = skeletons[query_id]
        tonic.integrate_feedback(
            root,
            skeleton,
            feedback[query_id],
            active_join_count=len(skeleton.steps),
        )
    return root


def measure_tonic(args: argparse.Namespace) -> list[dict]:
    from scripts.reproduce.tonic import run_tonic_full as tonic_runner
    from scripts.reproduce.tonic import tonic_common as tonic

    workload = args.workload
    source_dir = RESULT_ROOT / workload.lower() / "tonic"
    skeletons = tonic.load_skeletons(source_dir / "skeletons.json")
    feedback_path = source_dir / "feedback.csv"
    feedback_source = "original_job" if workload == "JOB" else "measured"
    connection = connect(workload, host=args.host, port=args.port, user=args.user)
    cursor = connection.cursor()
    rows: list[dict] = []
    try:
        for protocol in selected_protocols(workload):
            expected = load_expected_predictions("tonic", workload, protocol)
            for fold, spec in split_folds(workload, protocol).items():
                root = train_tonic_root(
                    tonic,
                    tonic_runner,
                    workload,
                    spec,
                    skeletons,
                    feedback_path,
                    feedback_source,
                )
                for query_id in limit_test_ids(spec["test"], args.limit):
                    gc.collect()
                    total_started = time.perf_counter()
                    skeleton_started = time.perf_counter()
                    live_skeleton = tonic.build_skeleton(cursor, workload, query_id)
                    skeleton_s = time.perf_counter() - skeleton_started
                    model_started = time.perf_counter()
                    prediction = tonic.predict_assignment(root, live_skeleton)
                    model_s = time.perf_counter() - model_started
                    post_started = time.perf_counter()
                    hint = tonic.assignment_hint(live_skeleton, prediction)
                    tonic.hinted_sql(workload, live_skeleton, prediction)
                    postprocessing_s = time.perf_counter() - post_started
                    inference_s = time.perf_counter() - total_started
                    expected_prediction = expected.get(fold, {}).get(query_id)
                    stored_skeleton = skeletons[query_id]
                    skeleton_match = (
                        live_skeleton.leading_tree == stored_skeleton.leading_tree
                        and [step.to_json() for step in live_skeleton.steps]
                        == [step.to_json() for step in stored_skeleton.steps]
                    )
                    rows.append(
                        {
                            "method": "TONIC",
                            "workload": workload,
                            "protocol": protocol,
                            "fold": fold,
                            "query_id": query_id,
                            "parsing_s": 0.0,
                            "encoding_or_explain_s": skeleton_s,
                            "model_s": model_s,
                            "postprocessing_s": postprocessing_s,
                            "inference_s": inference_s,
                            "prediction": prediction,
                            "hint": hint,
                            "expected_prediction": ""
                            if expected_prediction is None
                            else expected_prediction,
                            "prediction_match": expected_prediction is None
                            or expected_prediction == prediction,
                            "skeleton_match": skeleton_match,
                        }
                    )
                    print(
                        f"TONIC {workload} {fold} {query_id}: "
                        f"{inference_s:.6f}s match={rows[-1]['prediction_match']} "
                        f"skeleton={skeleton_match}",
                        flush=True,
                    )
    finally:
        cursor.close()
        connection.close()
    return rows


def canonical_hint(value: str) -> str:
    return " ".join(value.split())


def measure_genjoin_tpch(args: argparse.Namespace) -> list[dict]:
    import torch
    from mo_sql_parsing import parse
    from scripts.reproduce.genjoin import run_genjoin_tpch as genjoin

    workload = "TPCH"
    method_dir = RESULT_ROOT / "tpch" / "corrected_splits" / "genjoin"
    metadata = json.loads((method_dir / "workload.json").read_text())
    num_edges = int(metadata["num_edges"])
    expected_rows = read_csv(method_dir / "predictions_with_experience.csv")
    expected = {
        (row["fold"], row["query_id"], int(row["run_id"])): row
        for row in expected_rows
    }
    parse("SELECT 1")
    connection = connect(workload, host=args.host, port=args.port, user=args.user)
    cursor = connection.cursor()
    genjoin.reset_session(cursor, load_pg_hint_plan=True)
    cardinalities = genjoin.full_cardinalities(cursor)
    rows: list[dict] = []
    try:
        for fold, spec in split_folds(workload, "random").items():
            models = [
                genjoin.load_model(method_dir / "models" / fold / f"run{run_id}.pt")
                for run_id in range(3)
            ]
            for query_id in limit_test_ids(spec["test"], args.limit):
                gc.collect()
                fixed_started = time.perf_counter()
                sql = genjoin.clean_sql(query_id)
                genjoin.parse_query_blocks(sql)
                parsing_s = time.perf_counter() - fixed_started
                query_spec = metadata["queries"][query_id]
                if query_spec["supported"]:
                    encoding_started = time.perf_counter()
                    query_encoding = genjoin.live_query_encoding(
                        cursor,
                        query_id,
                        query_spec,
                        num_edges,
                        cardinalities,
                    )
                    encoding_s = time.perf_counter() - encoding_started
                else:
                    query_encoding = None
                    encoding_s = 0.0
                fixed_s = parsing_s + encoding_s

                for run_id, model in enumerate(models):
                    if not query_spec["supported"]:
                        row_expected = expected[(fold, query_id, run_id)]
                        rows.append(
                            {
                                "method": "GenJoin",
                                "workload": workload,
                                "protocol": "random",
                                "fold": fold,
                                "query_id": query_id,
                                "run_id": run_id,
                                "supported": False,
                                "parsing_s": parsing_s,
                                "encoding_or_explain_s": 0.0,
                                "model_s": 0.0,
                                "postprocessing_s": 0.0,
                                "inference_s": fixed_s,
                                "prediction": "",
                                "expected_prediction": row_expected["predicted_hints"],
                                "prediction_match": not row_expected["predicted_hints"],
                            }
                        )
                        continue

                    seed = (
                        genjoin.DEFAULT_SEED
                        + 100000 * run_id
                        + 1000 * sum(ord(char) for char in fold)
                        + int(query_id)
                    )
                    model_started = time.perf_counter()
                    input_encoding = genjoin.sample_plan_encoding(
                        num_edges, query_spec["valid_edges"], seed
                    )
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        output, _, _, _ = model(
                            torch.from_numpy(input_encoding).view(1, -1),
                            torch.from_numpy(query_encoding).view(1, -1),
                            torch.zeros((1, 1), dtype=torch.float32),
                            torch.zeros((1, 1), dtype=torch.float32),
                        )
                    model_s = time.perf_counter() - model_started
                    post_started = time.perf_counter()
                    decoded = genjoin.decoded_plan_encoding(
                        output, num_edges, query_spec["valid_edges"]
                    )
                    hints = genjoin.hints_from_encoding(query_spec, decoded)
                    postprocessing_s = time.perf_counter() - post_started
                    inference_s = fixed_s + model_s + postprocessing_s
                    row_expected = expected[(fold, query_id, run_id)]
                    rows.append(
                        {
                            "method": "GenJoin",
                            "workload": workload,
                            "protocol": "random",
                            "fold": fold,
                            "query_id": query_id,
                            "run_id": run_id,
                            "supported": True,
                            "parsing_s": parsing_s,
                            "encoding_or_explain_s": encoding_s,
                            "model_s": model_s,
                            "postprocessing_s": postprocessing_s,
                            "inference_s": inference_s,
                            "prediction": hints,
                            "expected_prediction": row_expected["predicted_hints"],
                            "prediction_match": canonical_hint(hints)
                            == canonical_hint(row_expected["predicted_hints"]),
                        }
                    )
                query_rows = rows[-3:]
                print(
                    f"GenJoin TPCH {fold} {query_id}: "
                    f"mean={np.mean([float(row['inference_s']) for row in query_rows]):.6f}s "
                    f"matches={sum(bool(row['prediction_match']) for row in query_rows)}/3",
                    flush=True,
                )
    finally:
        cursor.close()
        connection.close()
    return rows


def load_autosteer_model(autosteer, model_path: Path):
    model_type, preprocessor_type = autosteer.import_autosteer_model()
    model = model_type(preprocessor_type())
    model.load(str(model_path))
    return model


def measure_autosteer_tpch(args: argparse.Namespace) -> list[dict]:
    from scripts.reproduce.autosteer import run_autosteer_tpch as autosteer

    workload = "TPCH"
    source_dir = RESULT_ROOT / "tpch" / "autosteer"
    method_dir = RESULT_ROOT / "tpch" / "corrected_splits" / "autosteer"
    records = autosteer.load_collection(source_dir / "collection.csv")
    expected = load_expected_predictions("autosteer", workload, "random")
    knobs = autosteer.load_knobs()
    connection = connect(workload, host=args.host, port=args.port, user=args.user)
    cursor = connection.cursor()
    rows: list[dict] = []
    try:
        for fold, spec in split_folds(workload, "random").items():
            model = load_autosteer_model(
                autosteer, method_dir / "models" / fold
            )
            for query_id in limit_test_ids(spec["test"], args.limit):
                gc.collect()
                total_started = time.perf_counter()
                sql = query_sql(workload, query_id)
                span_started = time.perf_counter()
                effective, dependencies, _ = autosteer.find_query_span(
                    cursor, sql, knobs
                )
                span_s = time.perf_counter() - span_started

                plans_started = time.perf_counter()
                configs = sorted(records[query_id])
                plans = [
                    autosteer.normalize_plan(
                        autosteer.explain_plan(cursor, sql, knobs, config)
                    )
                    for config in configs
                ]
                plan_generation_s = time.perf_counter() - plans_started

                model_started = time.perf_counter()
                estimates = model.predict(plans).reshape(-1)
                model_s = time.perf_counter() - model_started
                post_started = time.perf_counter()
                prediction = configs[int(np.argmin(estimates))]
                prediction_text = autosteer.config_text(prediction)
                postprocessing_s = time.perf_counter() - post_started
                inference_s = time.perf_counter() - total_started
                expected_prediction = expected.get(fold, {}).get(query_id)
                rows.append(
                    {
                        "method": "AutoSteer",
                        "workload": workload,
                        "protocol": "random",
                        "fold": fold,
                        "query_id": query_id,
                        "candidate_plans": len(configs),
                        "effective_knobs": len(effective),
                        "dependency_knobs": len(dependencies),
                        "parsing_s": 0.0,
                        "query_span_s": span_s,
                        "encoding_or_explain_s": plan_generation_s,
                        "model_s": model_s,
                        "postprocessing_s": postprocessing_s,
                        "inference_s": inference_s,
                        "prediction": prediction_text,
                        "expected_prediction": ""
                        if expected_prediction is None
                        else expected_prediction,
                        "prediction_match": expected_prediction is None
                        or expected_prediction == prediction_text,
                    }
                )
                print(
                    f"AutoSteer TPCH {fold} {query_id}: {inference_s:.6f}s "
                    f"plans={len(configs)} match={rows[-1]['prediction_match']}",
                    flush=True,
                )
    finally:
        cursor.close()
        connection.close()
    return rows


def output_path(args: argparse.Namespace) -> Path:
    return args.output_root / args.method / f"{args.workload.lower()}.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("fastgres", "tonic", "genjoin", "autosteer"), required=True
    )
    parser.add_argument("--workload", choices=("JOB", "STACK", "TPCH"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    if args.method in {"genjoin", "autosteer"} and args.workload != "TPCH":
        parser.error(f"{args.method} is measured here only for TPC-H")

    started = time.perf_counter()
    if args.method == "fastgres":
        rows = measure_fastgres(args)
    elif args.method == "tonic":
        rows = measure_tonic(args)
    elif args.method == "genjoin":
        rows = measure_genjoin_tpch(args)
    else:
        rows = measure_autosteer_tpch(args)
    elapsed_s = time.perf_counter() - started

    path = output_path(args)
    write_csv(path, rows)
    atomic_json(
        path.with_suffix(".json"),
        {
            "method": args.method,
            "workload": args.workload,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_s": elapsed_s,
            "records": len(rows),
            "prediction_mismatches": sum(
                not bool(row.get("prediction_match", True)) for row in rows
            ),
            "definition": (
                "Unseen-query parsing/encoding, auxiliary EXPLAIN calls, model "
                "prediction, and hint generation; excludes fold training, model "
                "loading, static schema metadata, and selected SQL execution."
            ),
            "csv": str(path),
        },
    )
    print(
        f"wrote {path} records={len(rows)} wall={elapsed_s:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
