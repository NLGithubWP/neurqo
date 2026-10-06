import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_reader():
    spec = importlib.util.spec_from_file_location(
        "revision_record", ROOT / "results/paper/revision_record.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_readers_accept_current_and_legacy_headings(tmp_path):
    reader = load_reader()
    current = reader.RECORD_PATH.read_text()
    assert "NeurQO WS" in current
    expected_importance = reader.load_action_importance()
    expected_panels = reader.load_per_query_panels()
    legacy = tmp_path / "old_record.md"
    legacy.write_text(current.replace("NeurQO", "NQO"))
    reader.RECORD_PATH = legacy
    assert reader.load_action_importance() == expected_importance
    assert reader.load_per_query_panels() == expected_panels
    assert legacy.read_text() == current.replace("NeurQO", "NQO")


def test_record_keeps_all_queries():
    panels = load_reader().load_per_query_panels()
    assert {name: len(rows) for name, rows in panels.items()} == {
        "JOB": 113,
        "STACK": 112,
        "TPC-H": 22,
    }


def test_overhead_plot_reads_legacy_csv_without_mutating_it(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location(
        "plot_overhead_ratio", ROOT / "results/paper/plot_overhead_ratio.py"
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    original = plot.OVERALL_RESULTS.read_bytes()
    legacy = plot.build_ratio_df()
    assert "NeurQO" in set(legacy["method"])
    assert "NQO" not in set(legacy["method"])
    assert plot.OVERALL_RESULTS.read_bytes() == original
    renamed = tmp_path / "current.csv"
    renamed.write_text(original.decode().replace("NQO", "NeurQO"))
    plot.OVERALL_RESULTS = renamed
    assert legacy.equals(plot.build_ratio_df())
