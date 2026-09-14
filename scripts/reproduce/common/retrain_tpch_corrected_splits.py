#!/usr/bin/env python3
"""Retrain TPC-H baselines on the corrected folds without test execution.

The expensive, fold-independent candidate measurements are reused. Selected
test actions are joined back to those measurements only on an exact match; no
aggregate metrics are computed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
REVISION_SCRIPTS = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "results" / "revision" / "tpch"
DEFAULT_OUTPUT_DIR = SOURCE_DIR / "corrected_splits"

sys.path.insert(0, str(REVISION_SCRIPTS))

from benchmarking.workloads import (  # noqa: E402
    connect,
    load_postgres_times,
    split_folds,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
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


def split_snapshot() -> dict:
    folds = split_folds("TPCH", "random")
    expected = {str(value) for value in range(1, 23)}
    test_ids = [query_id for spec in folds.values() for query_id in spec["test"]]
    if len(test_ids) != 22 or set(test_ids) != expected:
        raise RuntimeError("corrected TPC-H test folds must partition all 22 queries")
    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("corrected TPC-H test folds contain duplicate queries")
    for fold, spec in folds.items():
        if set(spec["train"]) & set(spec["test"]):
            raise RuntimeError(f"train/test overlap in {fold}")
        if set(spec["train"]) | set(spec["test"]) != expected:
            raise RuntimeError(f"{fold} does not cover all TPC-H queries")
    split_path = ROOT / "workloads" / "train_test.py"
    return {
        "split_file": str(split_path.relative_to(ROOT)),
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "folds": folds,
        "test_partition": test_ids,
    }


def run_fastgres(output_root: Path) -> None:
    from scripts.reproduce.fastgres import run_fastgres_full as fastgres

    method_dir = output_root / "fastgres"
    method_dir.mkdir(parents=True, exist_ok=True)
    utility, featurize, native_query = fastgres.import_fastgres()
    query_dir = SOURCE_DIR / "fastgres" / "queries"

    connection = connect("TPCH")
    try:
        type_dict = fastgres.catalog_types(connection)
        query_objects = fastgres.build_query_objects(
            "TPCH", query_dir, type_dict, native_query
        )
        minmax, encoders, wildcards, skipped = fastgres.load_or_build_metadata(
            "TPCH",
            connection,
            query_objects,
            type_dict,
            utility,
            ROOT / "results" / "revision",
        )
    finally:
        connection.close()

    connection_string = "dbname=tpch user=pgdb host=localhost port=15432"
    feature_started = time.perf_counter()
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
    feature_s = time.perf_counter() - feature_started
    archive = json.loads((SOURCE_DIR / "fastgres" / "archive.json").read_text())

    labels = {
        (row["query_id"], int(row["hint"])): row
        for row in read_csv(SOURCE_DIR / "fastgres" / "labels.csv")
    }
    predictions_by_fold = {}
    audits = {}
    selected = []
    for fold, spec in split_folds("TPCH", "random").items():
        predictions, audit = fastgres.predict_fold(
            spec, query_objects, features, archive
        )
        predictions_by_fold[fold] = predictions
        audits[fold] = audit
        for query_id, hint in predictions.items():
            source = labels.get((query_id, hint))
            if source is None:
                raise RuntimeError(f"FASTgres label missing for Q{query_id} hint {hint}")
            selected.append(
                {
                    "fold": fold,
                    "query_id": query_id,
                    "predicted_hint": hint,
                    "experience_match": True,
                    "experience_source": "fastgres/labels.csv",
                    **source,
                }
            )
        print(
            f"FASTgres {fold}: trained={len(spec['train'])} "
            f"predicted={len(predictions)} training_s={audit['training_s']:.3f}",
            flush=True,
        )

    atomic_json(
        method_dir / "random_predictions.json",
        {
            "method": "FASTgres",
            "execution_policy": "exact replay from collected labels; no test SQL executed",
            "feature_preparation_s": feature_s,
            "predictions": predictions_by_fold,
            "fold_audit": audits,
            "split": split_snapshot(),
        },
    )
    write_csv(method_dir / "selected_experience.csv", selected)


def run_tonic(output_root: Path) -> None:
    from scripts.reproduce.tonic import run_tonic_full as tonic
    from scripts.reproduce.tonic.tonic_common import load_skeletons

    method_dir = output_root / "tonic"
    method_dir.mkdir(parents=True, exist_ok=True)
    source_dir = SOURCE_DIR / "tonic"
    skeletons = load_skeletons(source_dir / "skeletons.json")
    feedback_path = source_dir / "feedback.csv"
    feedback_rows = read_csv(feedback_path)
    feedback = {(row["query_id"], row["assignment"]): row for row in feedback_rows}

    predictions_by_fold = {}
    audits = {}
    selected = []
    for fold, spec in split_folds("TPCH", "random").items():
        predictions, audit = tonic.train_and_predict_fold(
            "TPCH", spec, skeletons, feedback_path, "measured"
        )
        predictions_by_fold[fold] = predictions
        audits[fold] = audit
        for query_id, assignment in predictions.items():
            source = feedback.get((query_id, assignment))
            if source is None:
                raise RuntimeError(
                    f"TONIC feedback missing for Q{query_id} assignment {assignment!r}"
                )
            selected.append(
                {
                    "fold": fold,
                    "query_id": query_id,
                    "predicted_assignment": assignment,
                    "experience_match": True,
                    "experience_source": "tonic/feedback.csv",
                    **source,
                }
            )
        print(
            f"TONIC {fold}: trained={len(spec['train'])} "
            f"predicted={len(predictions)} training_s={audit['training_s']:.6f}",
            flush=True,
        )

    atomic_json(
        method_dir / "random_predictions.json",
        {
            "method": "TONIC",
            "execution_policy": "exact replay from collected feedback; no test SQL executed",
            "predictions": predictions_by_fold,
            "fold_audit": audits,
            "split": split_snapshot(),
        },
    )
    write_csv(method_dir / "selected_experience.csv", selected)


def canonical_hint(value: str) -> str:
    return " ".join(value.split())


def run_genjoin(output_root: Path, *, models: int, epochs: int) -> None:
    from scripts.reproduce.genjoin import run_genjoin_tpch as genjoin

    method_dir = output_root / "genjoin"
    method_dir.mkdir(parents=True, exist_ok=True)
    for name in ("workload.json", "collection.csv"):
        source = SOURCE_DIR / "genjoin" / name
        destination = method_dir / name
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            shutil.copy2(source, destination)

    genjoin.OUTPUT_DIR = method_dir
    genjoin.train(
        candidates=genjoin.DEFAULT_CANDIDATES,
        pair_gap=genjoin.DEFAULT_PAIR_GAP,
        n_models=models,
        epochs=epochs,
    )

    metadata = genjoin.load_metadata()
    num_edges = int(metadata["num_edges"])
    pg_times = load_postgres_times(
        SOURCE_DIR / "postgres.csv", time_column="run1_charged_s"
    )
    source_rows = read_csv(SOURCE_DIR / "genjoin" / "collection.csv")
    experience: dict[tuple[str, str], list[dict]] = {}
    for row in source_rows:
        experience.setdefault(
            (row["query_id"], canonical_hint(row["hints"])), []
        ).append(row)

    rows = []
    for fold, spec in split_folds("TPCH", "random").items():
        fold_models = [
            genjoin.load_model(method_dir / "models" / fold / f"run{run_id}.pt")
            for run_id in range(models)
        ]
        for query_id in spec["test"]:
            query_spec = metadata["queries"][query_id]
            if not query_spec["supported"]:
                for run_id in range(models):
                    rows.append(
                        {
                            "fold": fold,
                            "query_id": query_id,
                            "run_id": run_id,
                            "supported": False,
                            "used_postgres_fallback": True,
                            "pg_runtime_s": pg_times[query_id],
                            "predicted_hints": "",
                            "predicted_plan_encoding": "",
                            "model_inference_s": 0.0,
                            "experience_match": True,
                            "experience_source": "postgres.csv",
                            "matched_candidate_ids": "",
                            "matched_charged_s": pg_times[query_id],
                        }
                    )
                continue

            query_encoding = np.asarray(
                query_spec["query_encoding"], dtype=np.float32
            )
            for run_id, model in enumerate(fold_models):
                seed = (
                    genjoin.DEFAULT_SEED
                    + 100000 * run_id
                    + 1000 * sum(ord(char) for char in fold)
                    + int(query_id)
                )
                input_encoding = genjoin.sample_plan_encoding(
                    num_edges, query_spec["valid_edges"], seed
                )
                torch.manual_seed(seed)
                started = time.perf_counter()
                with torch.no_grad():
                    output, _, _, _ = model(
                        torch.from_numpy(input_encoding).view(1, -1),
                        torch.from_numpy(query_encoding).view(1, -1),
                        torch.zeros((1, 1), dtype=torch.float32),
                        torch.zeros((1, 1), dtype=torch.float32),
                    )
                decoded = genjoin.decoded_plan_encoding(
                    output, num_edges, query_spec["valid_edges"]
                )
                hints = genjoin.hints_from_encoding(query_spec, decoded)
                inference_s = time.perf_counter() - started
                matches = experience.get((query_id, canonical_hint(hints)), [])
                rows.append(
                    {
                        "fold": fold,
                        "query_id": query_id,
                        "run_id": run_id,
                        "supported": True,
                        "used_postgres_fallback": False,
                        "pg_runtime_s": pg_times[query_id],
                        "predicted_hints": hints,
                        "predicted_plan_encoding": json.dumps(decoded.tolist()),
                        "model_inference_s": inference_s,
                        "experience_match": bool(matches),
                        "experience_source": "genjoin/collection.csv" if matches else "",
                        "matched_candidate_ids": ";".join(
                            row["candidate_id"] for row in matches
                        ),
                        "matched_charged_s": ";".join(
                            row["charged_s"] for row in matches
                        ),
                    }
                )
        print(
            f"GenJoin {fold}: trained={len(spec['train'])} "
            f"predicted={len(spec['test']) * models}",
            flush=True,
        )

    write_csv(method_dir / "predictions_with_experience.csv", rows)
    atomic_json(
        method_dir / "prediction_audit.json",
        {
            "method": "GenJoin",
            "models_per_fold": models,
            "epochs": epochs,
            "execution_policy": (
                "No test SQL executed; exact collected hint matches are attached, "
                "and unmatched generated hints retain no runtime."
            ),
            "query_encoding_source": "reused workload.json",
            "records": len(rows),
            "exact_experience_matches": sum(
                str(row["experience_match"]).lower() == "true" for row in rows
            ),
            "split": split_snapshot(),
        },
    )


def run_autosteer(output_root: Path, *, seed: int) -> None:
    from scripts.reproduce.autosteer import run_autosteer_tpch as autosteer

    method_dir = output_root / "autosteer"
    method_dir.mkdir(parents=True, exist_ok=True)
    records = autosteer.load_collection(SOURCE_DIR / "autosteer" / "collection.csv")
    missing = [str(value) for value in range(1, 23) if not records.get(str(value))]
    if missing:
        raise RuntimeError(f"AutoSteer collection missing queries {missing}")

    predictions_by_fold = {}
    inference_by_fold = {}
    audits = {}
    selected = []
    for index, (fold, spec) in enumerate(split_folds("TPCH", "random").items()):
        model, audit = autosteer.train_fold(
            fold,
            spec,
            records,
            method_dir / "models",
            seed + index,
        )
        predictions, inference = autosteer.choose_test_configs(model, spec, records)
        predictions_by_fold[fold] = {
            query_id: autosteer.config_text(config)
            for query_id, config in predictions.items()
        }
        inference_by_fold[fold] = inference
        audits[fold] = audit
        for query_id, config in predictions.items():
            source = records[query_id][config]
            selected.append(
                {
                    "fold": fold,
                    "query_id": query_id,
                    "predicted_config": autosteer.config_text(config),
                    "inference_s": inference[query_id],
                    "experience_match": True,
                    "experience_source": "autosteer/collection.csv",
                    **autosteer.candidate_to_row(query_id, source),
                }
            )
        print(
            f"AutoSteer {fold}: candidates={audit['training_candidates']} "
            f"predicted={len(predictions)} training_s={audit['training_s']:.3f}",
            flush=True,
        )

    atomic_json(
        method_dir / "random_predictions.json",
        {
            "method": "AutoSteer",
            "execution_policy": "exact replay from collected candidates; no test SQL executed",
            "predictions": predictions_by_fold,
            "inference_s": inference_by_fold,
            "fold_audit": audits,
            "split": split_snapshot(),
        },
    )
    write_csv(method_dir / "selected_experience.csv", selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", choices=("fastgres", "tonic", "genjoin", "autosteer")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "split_audit.json", split_snapshot())
    if args.method == "fastgres":
        run_fastgres(output_root)
    elif args.method == "tonic":
        run_tonic(output_root)
    elif args.method == "genjoin":
        run_genjoin(output_root, models=args.models, epochs=args.epochs)
    elif args.method == "autosteer":
        run_autosteer(output_root, seed=args.seed)


if __name__ == "__main__":
    main()
