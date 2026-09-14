from __future__ import annotations

import math
import re
from pathlib import Path


RECORD_PATH = Path(__file__).resolve().with_name("all_record.md")


def _record_text() -> str:
    return RECORD_PATH.read_text()


def _between(text: str, start: str, end: str | None = None) -> str:
    begin = text.index(start)
    finish = text.index(end, begin) if end is not None else len(text)
    return text[begin:finish]


def _tables(text: str) -> list[tuple[list[str], list[dict[str, str]]]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    parsed = []
    for block in blocks:
        if len(block) < 3:
            continue
        headers = [cell.strip() for cell in block[0].strip("|").split("|")]
        rows = []
        for line in block[2:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(headers):
                raise ValueError(f"malformed Markdown table row: {line}")
            rows.append(dict(zip(headers, cells)))
        parsed.append((headers, rows))
    return parsed


def _number(value: str) -> float:
    cleaned = value.replace("`", "").replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"no numeric value in {value!r}")
    return float(match.group(0))


def _optional_number(value: str) -> float | None:
    try:
        return _number(value)
    except ValueError:
        return None


def _required_number(
    row: dict[str, str], column: str, qid: str, *, integral: bool = False
) -> float:
    if column not in row:
        raise ValueError(f"missing {column!r} for query {qid!r}")
    cleaned = row[column].replace("`", "").replace(",", "").strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        raise ValueError(
            f"invalid {column!r} value {row[column]!r} for query {qid!r}"
        )
    value = float(cleaned)
    if not math.isfinite(value) or value < 0 or (integral and not value.is_integer()):
        raise ValueError(
            f"invalid {column!r} value {row[column]!r} for query {qid!r}"
        )
    return value


def _validate_per_query_rows(workload: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no per-query records found for {workload}")
    seen: set[str] = set()
    for row in rows:
        qid = row["qid"]
        if qid in seen:
            raise ValueError(f"duplicate query ID {qid!r} in {workload}")
        seen.add(qid)


def _curve_points(
    rows: list[dict[str, str]], elapsed_column: str, runtime_column: str
) -> list[dict]:
    points = []
    for row in rows:
        runtime = _optional_number(row[runtime_column])
        if runtime is None:
            continue
        points.append(
            {
                "stage": row["Training iteration"],
                "elapsed_min": _number(row[elapsed_column]),
                "norm_runtime": runtime,
            }
        )
    return points


def load_learning_curves() -> dict[str, dict[str, dict]]:
    text = _between(_record_text(), "# 1. Learning efficiency", "# 2. Action Importance")
    out: dict[str, dict[str, dict]] = {"JOB": {}, "STACK": {}, "TPC-H": {}}

    tpch = _between(text, "## TPC-H", "## JOB")
    _, rows = _tables(tpch)[0]
    q_match = re.search(r"#SubQ\s*=\s*(\d+)", tpch)
    out["TPC-H"]["Random"] = {
        "subq": int(q_match.group(1)) if q_match else None,
        "points": _curve_points(
            rows, "Elapsed Time (min)", "Best-so-far 1/WS"
        ),
    }

    job = _between(text, "## JOB", "## STACK")
    job_tables = _tables(job)
    _, final_rows = job_tables[0]
    _, curve_rows = job_tables[1]
    subq = {row["Protocol"]: int(_number(row["#SubQ"])) for row in final_rows}
    for protocol, prefix in (
        ("Base Query", "Base"),
        ("Leave-One-Out", "LOO"),
        ("Random", "Random"),
    ):
        final_name = {
            "Base Query": "Base-query",
            "Leave-One-Out": "Leave-one-out",
            "Random": "Random",
        }[protocol]
        out["JOB"][protocol] = {
            "subq": subq[final_name],
            "points": _curve_points(
                curve_rows,
                f"{prefix} elapsed (min)",
                f"{prefix} best-so-far 1/WS",
            ),
        }

    stack = _between(text, "## STACK")
    stack_tables = _tables(stack)
    _, final_rows = stack_tables[0]
    _, curve_rows = stack_tables[1]
    subq = {row["Protocol"]: int(_number(row["#SubQ"])) for row in final_rows}
    for protocol, prefix in (
        ("Base Query", "Base"),
        ("Leave-One-Out", "LOO"),
        ("Random", "Random"),
    ):
        final_name = {
            "Base Query": "Base-query",
            "Leave-One-Out": "Leave-one-out",
            "Random": "Random",
        }[protocol]
        out["STACK"][protocol] = {
            "subq": subq[final_name],
            "points": _curve_points(
                curve_rows,
                f"{prefix} elapsed (min)",
                f"{prefix} best-so-far 1/WS",
            ),
        }
    return out


def load_action_importance() -> dict[str, dict[str, float | None]]:
    text = _between(_record_text(), "# 2. Action Importance", "# 3. Action Frequence")
    _, rows = _tables(text)[0]
    workload_names = {"JOB": "job", "STACK": "stack", "TPC-H": "tpch"}
    columns = {
        "nqo": "NQO WS",
        "wo_split": "w/o Query Split WS",
        "wo_search": "w/o TOPK WS",
        "wo_lip": "w/o filter WS",
        "wo_aja": "w/o Ajoin WS",
    }
    out: dict[str, dict[str, float | None]] = {}
    for row in rows:
        workload = workload_names[row["Dataset"]]
        out[workload] = {
            setting: _optional_number(row[column])
            for setting, column in columns.items()
        }
    return out


def load_action_distribution_means() -> list[dict]:
    text = _between(_record_text(), "# 3. Action Frequence", "# 4. Per-Query Performance")
    blocks = {
        "TPCH": _between(text, "## TPC-H (random)", "## JOB"),
        "JOB": _between(text, "## JOB", "## STACK"),
        "STACK": _between(text, "## STACK"),
    }
    out = []
    for workload, block in blocks.items():
        headers, rows = _tables(block)[0]
        value_col = "Frequency" if workload == "TPCH" else "Mean frequency"
        values = {
            (row["Head"], row["Action"]): 100.0 * _number(row[value_col])
            for row in rows
        }

        def value(head: str, action: str) -> float:
            for (row_head, row_action), frequency in values.items():
                if row_head == head and row_action == action:
                    return frequency
            raise KeyError((workload, head, action, headers))

        def select_value(action_suffix: str) -> float:
            for (row_head, row_action), frequency in values.items():
                if row_head == "Select" and row_action.endswith(action_suffix):
                    return frequency
            raise KeyError((workload, "Select", action_suffix, headers))

        select_freqs = {
            "0.0": float("nan"),
            "0.5": float("nan"),
            "1.0": float("nan"),
        }
        if workload != "TPCH":
            select_freqs = {
                "0.0": select_value("0.0"),
                "0.5": select_value("0.5"),
                "1.0": select_value("1.0"),
            }

        out.append(
            {
                "workload": workload,
                "high_level_freqs": {
                    "Split": value("High", "Split"),
                    "Non-split": value("High", "Stop"),
                },
                "select_level_freqs": select_freqs,
                "medium_level_freqs": {
                    "Default": value("Search", "Default"),
                    "Split-search": 0.0,
                    "Top-5": value("Search", "TOPK (top5)"),
                    "Top-10": 0.0,
                },
                "low_level_freqs": {
                    "None": value("Low", "None"),
                    "Filter": value("Low", "Filter (LIP)"),
                    "Filter+AJoin": value("Low", "Filter + Ajoin"),
                    "AJoin": value("Low", "Ajoin (AJA)"),
                },
            }
        )
    return out


def load_per_query_panels() -> dict[str, list[dict]]:
    text = _between(
        _record_text(), "# 4. Per-Query Performance", "# 5. RL Formulation"
    )
    blocks = {
        "TPC-H": _between(text, "## TPC-H (random)", "## JOB"),
        "JOB": _between(text, "## JOB", "## STACK"),
        "STACK": _between(text, "## STACK"),
    }
    out: dict[str, list[dict]] = {}

    _, rows = _tables(blocks["TPC-H"])[0]
    out["TPC-H"] = [
        {
            "qid": row["Query"][1:] if row["Query"].startswith("Q") else row["Query"],
            "pg_s": _number(row["PG (s)"]),
            "nqo_s": _number(row["NQO (s)"]),
            "delta_s": _number(row["Δt (s)"]),
            "delta_min": _number(row["Δt (s)"]),
            "delta_max": _number(row["Δt (s)"]),
            "join_count": _required_number(
                row, "# Joins", row["Query"], integral=True
            ),
            "intermediate_rows": _required_number(
                row, "Intermediate Rows", row["Query"]
            ),
        }
        for row in rows
    ]

    for workload in ("JOB", "STACK"):
        _, rows = _tables(blocks[workload])[1]
        out[workload] = [
            {
                "qid": row["Query"],
                "pg_s": _number(row["Mean PG (s)"]),
                "nqo_s": _number(row["Mean NQO (s)"]),
                "delta_s": _number(row["Mean Δt (s)"]),
                "delta_min": _number(row["Min Δt (s)"]),
                "delta_max": _number(row["Max Δt (s)"]),
                "join_count": _required_number(
                    row, "# Joins", row["Query"], integral=True
                ),
                "intermediate_rows": _required_number(
                    row, "Intermediate Rows", row["Query"]
                ),
            }
            for row in rows
        ]

    for workload, rows in out.items():
        _validate_per_query_rows(workload, rows)
        rows.sort(key=lambda row: (row["pg_s"], row["qid"]))
    return out


def load_transferability() -> dict[str, dict[str, float]]:
    text = _between(_record_text(), "# 6. Cross-workload Transferability")
    _, rows = _tables(text)[0]
    targets = {"JOB": "job", "STACK": "stack", "TPC-H": "tpch"}
    columns = {
        "same": "Same-workload",
        "mixed": "Mixed-workload",
        "job": "Cross: source=JOB",
        "stack": "Cross: source=STACK",
        "tpch": "Cross: source=TPC-H",
    }
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        target = targets[row["Target workload"]]
        values = {}
        for name, column in columns.items():
            value = _optional_number(row[column])
            if value is not None:
                values[name] = value
        result[target] = values
    return result
