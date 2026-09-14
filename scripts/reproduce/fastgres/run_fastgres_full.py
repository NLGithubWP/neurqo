#!/usr/bin/env python3
"""Train and evaluate full-space FASTgres with the shared timing policy."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import psycopg2
from psycopg2 import sql as pg_sql

from benchmarking.workloads import (
    DEFAULT_BASELINE_WORK_ROOT,
    MEASUREMENT_COLUMNS,
    SPLIT_PROTOCOLS,
    CsvResultStore,
    connect,
    dynamic_timeout_s,
    execute_once,
    load_postgres_times,
    query_path,
    query_sql,
    reset_session,
    one_result_to_row,
    split_folds,
    sql_sha256,
    workload_query_ids,
)
from scripts.reproduce.fastgres.measure_fastgres_labels import set_fastgres_hint


ROOT = Path(__file__).resolve().parents[3]
FASTGRES_ROOT = ROOT / "thrid_party" / "FASTgres-PVLDBv16"
ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "baseline_caches"
    / "revision"
    / "fastgres"
)

RESULT_FIELDS = [
    "result_key",
    "workload",
    "protocol",
    "fold",
    "query_id",
    "sql_path",
    "sql_sha256",
    "predicted_hint",
    "pg_measured_s",
] + MEASUREMENT_COLUMNS


def import_fastgres():
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(FASTGRES_ROOT))
    os.chdir(str(FASTGRES_ROOT))
    try:
        import featurize
        import utility
        from query import Query
    finally:
        os.chdir(str(previous_cwd))
    return utility, featurize, Query


def prepare_query_dir(workload: str, output_root: Path) -> Path:
    query_dir = output_root / workload.lower() / "fastgres" / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    for query_id in workload_query_ids(workload):
        path = query_dir / "{}.sql".format(query_id)
        sql = query_sql(workload, query_id)
        if not path.is_file() or path.read_text() != sql:
            path.write_text(sql)
        expected.add(path.name)
    for path in query_dir.glob("*.sql"):
        if path.name not in expected:
            path.unlink()
    return query_dir


def catalog_types(connection):
    type_names = {
        "int2": "smallint",
        "int4": "integer",
        "int8": "bigint",
        "float4": "real",
        "float8": "double precision",
        "numeric": "numeric",
        "date": "date",
        "timestamp": "timestamp without time zone",
        "timestamptz": "timestamp with time zone",
        "varchar": "character varying",
        "bpchar": "character",
        "text": "text",
    }
    query = """
        SELECT c.relname, a.attname, t.typname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        JOIN pg_type t ON t.oid = a.atttypid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY c.relname, a.attnum
    """
    cursor = connection.cursor()
    cursor.execute(query)
    result = {}
    for table, column, type_name in cursor.fetchall():
        result.setdefault(table, {})[column] = type_names.get(type_name, type_name)
    cursor.close()
    return result


def _walk_from(entry, aliases):
    if isinstance(entry, list):
        for item in entry:
            _walk_from(item, aliases)
        return
    if isinstance(entry, str):
        aliases.setdefault(entry.split(".")[-1], entry)
        return
    if not isinstance(entry, dict):
        return

    value = entry.get("value")
    alias = entry.get("name")
    if isinstance(value, str):
        aliases[alias or value.split(".")[-1]] = value
    elif isinstance(value, dict):
        _collect_aliases(value, aliases)

    for key, child in entry.items():
        if "join" in key.lower():
            _walk_from(child, aliases)
        elif key == "on":
            continue


def _collect_aliases(node, aliases):
    if isinstance(node, list):
        for item in node:
            _collect_aliases(item, aliases)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "from":
            _walk_from(value, aliases)
        _collect_aliases(value, aliases)


def _literal(value):
    if isinstance(value, dict) and "literal" in value:
        return value["literal"]
    if isinstance(value, (str, int, float, list, tuple)):
        return value
    return value


def _column_reference(value, aliases, known_tables):
    if not isinstance(value, str) or "." not in value:
        return None
    alias, column = value.split(".", 1)
    table = aliases.get(alias)
    if table not in known_tables or column not in known_tables[table]:
        return None
    return table, column


def _collect_predicates(node, aliases, known_tables, attributes):
    if isinstance(node, list):
        for item in node:
            _collect_predicates(item, aliases, known_tables, attributes)
        return
    if not isinstance(node, dict):
        return

    supported = {
        "eq",
        "gt",
        "lt",
        "gte",
        "lte",
        "neq",
        "like",
        "not_like",
        "in",
    }
    for operator, operands in node.items():
        if operator in ("and", "or"):
            _collect_predicates(operands, aliases, known_tables, attributes)
            continue
        if operator not in supported:
            _collect_predicates(operands, aliases, known_tables, attributes)
            continue
        if not isinstance(operands, (list, tuple)) or len(operands) < 2:
            continue
        left, right = operands[0], operands[1]
        reference = _column_reference(left, aliases, known_tables)
        value = _literal(right)
        if reference is None:
            reference = _column_reference(right, aliases, known_tables)
            value = _literal(left)
        if reference is None or _column_reference(value, aliases, known_tables):
            continue
        table, column = reference
        if (
            operator == "in"
            and isinstance(value, (list, tuple))
            and known_tables[table][column] == "integer"
        ):
            numeric_values = [
                item for item in value if isinstance(item, (int, float))
            ]
            if not numeric_values:
                continue
            value = sum(numeric_values) / len(numeric_values)
        attributes[table][column][operator] = value


class GenericQuery:
    """FASTgres-compatible query representation for unsupported TPC-H SQL."""

    def __init__(self, name, parsed, known_tables):
        self.name = name
        self.parsed = parsed
        aliases = {}
        _collect_aliases(parsed, aliases)
        self.tables = {
            alias: table for alias, table in aliases.items() if table in known_tables
        }
        self.context = frozenset(sorted(set(self.tables.values())))
        self.attributes = defaultdict(lambda: defaultdict(dict))

        def visit(node):
            if isinstance(node, list):
                for item in node:
                    visit(item)
                return
            if not isinstance(node, dict):
                return
            for key, value in node.items():
                if key in ("where", "on", "having"):
                    _collect_predicates(
                        value, self.tables, known_tables, self.attributes
                    )
                visit(value)

        visit(parsed)


def build_query_objects(workload, query_dir, known_tables, NativeQuery):
    objects = {}
    if workload == "TPCH":
        from mo_sql_parsing import parse

        for query_id in workload_query_ids(workload):
            name = "{}.sql".format(query_id)
            objects[name] = GenericQuery(name, parse((query_dir / name).read_text()), known_tables)
        return objects

    with contextlib.redirect_stdout(io.StringIO()):
        for query_id in workload_query_ids(workload):
            name = "{}.sql".format(query_id)
            objects[name] = NativeQuery(name, str(query_dir) + "/")
    return objects


def _flatten_string_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def build_minmax(connection, query_objects, type_dict):
    needed = set()
    for query in query_objects.values():
        for table, columns in query.attributes.items():
            for column in columns:
                if type_dict.get(table, {}).get(column) in (
                    "integer",
                    "timestamp without time zone",
                ):
                    needed.add((table, column))
    cursor = connection.cursor()
    result = {}
    for table, column in sorted(needed):
        statement = pg_sql.SQL("SELECT MIN({column}), MAX({column}) FROM {table}").format(
            column=pg_sql.Identifier(column),
            table=pg_sql.Identifier(table),
        )
        cursor.execute(statement)
        minimum, maximum = cursor.fetchone()
        if minimum is None or maximum is None:
            minimum, maximum = 0, 1
        if minimum == maximum:
            maximum = minimum + 1
        result.setdefault(table, {})[column] = [minimum, maximum]
    cursor.close()
    return result


def build_label_encoders(
    connection, query_objects, type_dict, skipped, MyLabelEncoder
):
    values_by_column = defaultdict(set)
    for query in query_objects.values():
        for table, columns in query.attributes.items():
            for column, operators in columns.items():
                if type_dict.get(table, {}).get(column) != "character varying":
                    continue
                if column in skipped.get(table, {}).get("columns", []):
                    continue
                for operator, value in operators.items():
                    if operator in ("eq", "lt", "gt", "neq", "in"):
                        values_by_column[(table, column)].update(
                            _flatten_string_values(value)
                        )

    cursor = connection.cursor()
    result = defaultdict(dict)
    for (table, column), values in sorted(values_by_column.items()):
        ordered_values = sorted(values)
        cardinalities = []
        statement = pg_sql.SQL(
            "SELECT COUNT({column}) FROM {table} WHERE {column} = %s"
        ).format(
            column=pg_sql.Identifier(column),
            table=pg_sql.Identifier(table),
        )
        for value in ordered_values:
            cursor.execute(statement, (value,))
            cardinalities.append(cursor.fetchone()[0])
        encoder = MyLabelEncoder()
        encoder.fit(ordered_values, cardinalities)
        result[table][column] = encoder
    cursor.close()
    return result


def build_wildcards(connection, query_objects, type_dict, skipped):
    patterns = defaultdict(set)
    for query in query_objects.values():
        for table, columns in query.attributes.items():
            for column, operators in columns.items():
                if type_dict.get(table, {}).get(column) != "character varying":
                    continue
                if column in skipped.get(table, {}).get("columns", []):
                    continue
                for operator, value in operators.items():
                    if operator in ("like", "not_like") and isinstance(value, str):
                        patterns[(table, column)].add((operator, value))

    cursor = connection.cursor()
    result = defaultdict(dict)
    table_rows = {}
    for (table, column), entries in sorted(patterns.items()):
        if table not in table_rows:
            cursor.execute(
                pg_sql.SQL("SELECT COUNT(*) FROM {}").format(pg_sql.Identifier(table))
            )
            table_rows[table] = cursor.fetchone()[0]
        result[table]["max"] = table_rows[table]
        result[table].setdefault(column, {})
        for operator, value in sorted(entries):
            sql_operator = "LIKE" if operator == "like" else "NOT LIKE"
            statement = pg_sql.SQL(
                "SELECT COUNT(*) FROM {table} WHERE {column} "
                + sql_operator
                + " %s"
            ).format(
                table=pg_sql.Identifier(table),
                column=pg_sql.Identifier(column),
            )
            cursor.execute(statement, (value,))
            result[table][column][value] = cursor.fetchone()[0]
    cursor.close()
    return result


def load_or_build_metadata(
    workload,
    connection,
    query_objects,
    type_dict,
    utility,
    output_root,
):
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
            str(
                ROOT
                / "results"
                / "rebuttal"
                / "r1_o2_baselines"
                / "fastgres_job"
                / "wildcard_dict.json"
            )
        )
        return minmax, encoders, wildcards, {}

    metadata_dir = ARTIFACT_ROOT / workload.lower()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "metadata.pkl"
    if metadata_path.is_file():
        with metadata_path.open("rb") as handle:
            return pickle.load(handle)

    skipped = {}
    if workload == "STACK":
        skipped = utility.load_json(
            str(
                FASTGRES_ROOT
                / "db_info"
                / "stack"
                / "skipped_table_columns_stack.json"
            )
        )
    minmax = build_minmax(connection, query_objects, type_dict)
    encoders = build_label_encoders(
        connection, query_objects, type_dict, skipped, utility.MyLabelEncoder
    )
    if workload == "STACK":
        wildcards = utility.load_json(
            str(FASTGRES_ROOT / "db_info" / "stack" / "wildcard_dict.json")
        )
    else:
        wildcards = build_wildcards(
            connection, query_objects, type_dict, skipped
        )
    metadata = (minmax, encoders, wildcards, skipped)
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
    return metadata


def build_features(
    query_objects,
    connection_string,
    type_dict,
    minmax,
    encoders,
    wildcards,
    skipped,
    utility,
    featurize,
):
    utility.build_db_type_dict = lambda _connection_string: type_dict
    features = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for query_name, query in query_objects.items():
            feature_dict = featurize.build_feature_dict(
                query,
                connection_string,
                minmax,
                encoders,
                wildcards,
                set(),
                set(),
                skipped,
            )
            features[query_name] = featurize.encode_query(
                query.context, feature_dict, type_dict
            )
    return features


def predict_fold(fold_spec, query_objects, features, archive):
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier

    training_started = time.perf_counter()
    train = ["{}.sql".format(query_id) for query_id in fold_spec["train"]]
    test = ["{}.sql".format(query_id) for query_id in fold_spec["test"]]
    context_train = defaultdict(list)
    for query_name in train:
        context_train[query_objects[query_name].context].append(query_name)

    models = {}
    for context, query_names in context_train.items():
        labels = [int(archive[name]["opt"]) for name in query_names]
        unique = np.unique(labels)
        if len(unique) == 1:
            models[context] = int(unique[0])
            continue
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=1000,
            random_state=29,
        )
        model.fit([features[name] for name in query_names], labels)
        models[context] = model
    training_s = time.perf_counter() - training_started

    prediction_started = time.perf_counter()
    predictions = {}
    for query_name in test:
        context = query_objects[query_name].context
        model = models.get(context)
        if model is None:
            prediction = 63
        elif isinstance(model, int):
            prediction = model
        else:
            prediction = int(model.predict([features[query_name]])[0])
        predictions[query_name[:-4]] = prediction
    prediction_s = time.perf_counter() - prediction_started
    audit = {
        "training_query_ids": [name[:-4] for name in train],
        "test_query_ids": [name[:-4] for name in test],
        "contexts": len(models),
        "fitted_models": sum(not isinstance(model, int) for model in models.values()),
        "constant_contexts": sum(isinstance(model, int) for model in models.values()),
        "training_s": training_s,
        "prediction_s": prediction_s,
    }
    return predictions, audit


def execute_protocol(
    args,
    workload,
    protocol,
    predictions_by_fold,
    pg_times,
):
    result_path = (
        args.output_root
        / workload.lower()
        / "fastgres"
        / "{}_results.csv".format(protocol)
    )
    store = CsvResultStore(result_path, RESULT_FIELDS)
    connection = connect(
        workload, host=args.host, port=args.port, user=args.user
    )
    cursor = connection.cursor()
    try:
        fold_items = list(predictions_by_fold.items())
        total = sum(len(predictions) for _, predictions in fold_items)
        idx = 0
        for fold, predictions in fold_items:
            for query_id, prediction in predictions.items():
                idx += 1
                sql = query_sql(workload, query_id)
                digest = sql_sha256(sql)
                timeout_s = dynamic_timeout_s(
                    pg_times[query_id], args.timeout_factor
                )
                result_key = "{}:fastgres-test:{}:{}:{}".format(
                    workload, protocol, fold, query_id
                )
                previous = store.get(result_key)
                if (
                    previous
                    and previous.get("sql_sha256") == digest
                    and int(previous["predicted_hint"]) == prediction
                    and abs(float(previous["timeout_s"]) - timeout_s) < 1e-9
                ):
                    print(
                        "[{} FASTgres {} {}/{}] {} {}: resume {}s".format(
                            workload,
                            protocol,
                            idx,
                            total,
                            fold,
                            query_id,
                            previous["measured_s"],
                        ),
                        flush=True,
                    )
                    continue

                reset_session(cursor)
                set_fastgres_hint(cursor, prediction)
                result = execute_once(cursor, sql, timeout_s=timeout_s)
                row = {
                    "result_key": result_key,
                    "workload": workload,
                    "protocol": protocol,
                    "fold": fold,
                    "query_id": query_id,
                    "sql_path": str(query_path(workload, query_id).relative_to(Path.cwd())),
                    "sql_sha256": digest,
                    "predicted_hint": prediction,
                    "pg_measured_s": pg_times[query_id],
                }
                row.update(one_result_to_row(result))
                store.append(row)
                print(
                    "[{} FASTgres {} {}/{}] {} {} hint={}: {:.6f}s{}".format(
                        workload,
                        protocol,
                        idx,
                        total,
                        fold,
                        query_id,
                        prediction,
                        result.charged_runtime_s,
                        " timeout/error" if result.error else "",
                    ),
                    flush=True,
                )
    finally:
        cursor.close()
        connection.close()


def run_workload(args, workload):
    utility, featurize, NativeQuery = import_fastgres()
    query_dir = prepare_query_dir(workload, args.output_root)
    connection = connect(
        workload, host=args.host, port=args.port, user=args.user
    )
    try:
        type_dict = catalog_types(connection)
        query_objects = build_query_objects(
            workload, query_dir, type_dict, NativeQuery
        )
        if any(not query.context for query in query_objects.values()):
            empty = [
                name for name, query in query_objects.items() if not query.context
            ]
            raise RuntimeError(
                "{} FASTgres queries have empty contexts: {}".format(
                    workload, empty[:5]
                )
            )
        minmax, encoders, wildcards, skipped = load_or_build_metadata(
            workload,
            connection,
            query_objects,
            type_dict,
            utility,
            args.output_root,
        )
    finally:
        connection.close()

    dbname = {
        "JOB": "imdb_ori",
        "STACK": "so",
        "TPCH": "tpch",
    }[workload]
    connection_string = "dbname={} user={} host={} port={}".format(
        dbname, args.user, args.host, args.port
    )
    feature_started = time.perf_counter()
    features = build_features(
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
    feature_preparation_s = time.perf_counter() - feature_started

    archive_path = (
        args.output_root / workload.lower() / "fastgres" / "archive.json"
    )
    archive = json.loads(archive_path.read_text())
    expected_archive = {"{}.sql".format(query_id) for query_id in workload_query_ids(workload)}
    if set(archive) != expected_archive:
        raise RuntimeError(
            "{} FASTgres archive is incomplete: expected {}, got {}".format(
                workload, len(expected_archive), len(archive)
            )
        )
    pg_times = load_postgres_times(
        args.output_root / workload.lower() / "postgres.csv"
    )

    for protocol in SPLIT_PROTOCOLS[workload]:
        predictions_by_fold = {}
        audit_by_fold = {}
        for fold, spec in split_folds(workload, protocol).items():
            predictions, audit = predict_fold(
                spec,
                query_objects,
                features,
                archive,
            )
            predictions_by_fold[fold] = predictions
            audit_by_fold[fold] = audit
            print(
                "{} FASTgres {} trained on {} and predicted {} queries for {} "
                "in {:.3f}s + {:.3f}s".format(
                    workload,
                    protocol,
                    len(audit["training_query_ids"]),
                    len(predictions),
                    fold,
                    audit["training_s"],
                    audit["prediction_s"],
                ),
                flush=True,
            )
        prediction_path = (
            args.output_root
            / workload.lower()
            / "fastgres"
            / "{}_predictions.json".format(protocol)
        )
        prediction_path.write_text(
            json.dumps(
                {
                    "predictions": predictions_by_fold,
                    "fold_audit": audit_by_fold,
                    "feature_preparation_s": feature_preparation_s,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        execute_protocol(
            args,
            workload,
            protocol,
            predictions_by_fold,
            pg_times,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=["JOB", "STACK", "TPCH", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_BASELINE_WORK_ROOT
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="pgdb")
    parser.add_argument("--timeout-factor", type=float, default=5.0)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    workloads = ("JOB", "STACK", "TPCH") if args.workload == "all" else (args.workload,)
    for workload in workloads:
        run_workload(args, workload)


if __name__ == "__main__":
    main()
