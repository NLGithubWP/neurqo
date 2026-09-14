#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullLocator

from common import FONT_CFG
from revision_record import load_learning_curves


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper" / "figs"
WORKLOADS = ["JOB", "STACK", "TPC-H"]
OUTPUT_SLOT = {"JOB": "random", "STACK": "leave_one_out", "TPC-H": "base_query"}
PROTOCOLS = ["Base Query", "Random", "Leave-One-Out"]
PROTOCOL_LABEL = {"Base Query": "Base", "Leave-One-Out": "LOO", "Random": "Random"}
PROTOCOL_COLOR = {"Base Query": "#e36a00", "Leave-One-Out": "#12a777", "Random": "#1f77b4"}
PROTOCOL_STYLE = {"Base Query": "-", "Leave-One-Out": "--", "Random": "-."}
FIGSIZE = (2.96, 2.0)
LINE_WIDTH = 4.2
BASELINE_WIDTH = 2.8
Y_TICKS = [0.5, 1.0, 2.0]
PLOT_FONT = {
    "base": FONT_CFG["base"] + 1,
    "label": FONT_CFG["label"] + 3,
    "title": FONT_CFG["title"] + 1,
    "legend": FONT_CFG["legend"] + 0.3,
    "xtick": FONT_CFG["xtick"] + 2.5,
    "ytick": FONT_CFG["ytick"] + 2.5,
}


def _style_axis(ax, workload: str, xlabel: str) -> None:
    ax.axhline(1.0, color="#8a8a8a", lw=BASELINE_WIDTH, alpha=0.9)
    ax.set_yscale("log")
    if workload == "TPC-H":
        y_ticks = [0.9, 1.0, 1.1, 1.2, 1.3]
        ax.set_ylim(0.9, 1.3)
        y_labels = ["0.9", "1", "1.1", "1.2", "1.3"]
    else:
        y_ticks = Y_TICKS
        ax.set_ylim(0.5, 2.0)
        y_labels = ["0.5", "1", "2"]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(y_labels))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "Norm. End-to-End\nTime [log scale]",
        fontsize=PLOT_FONT["label"],
    )
    ax.yaxis.label.set_y(0.30)
    ax.set_title(workload, pad=4)
    ax.grid(axis="y", color="#dddddd", linewidth=1.2)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.8)
    ax.tick_params(axis="x", length=0, labelsize=PLOT_FONT["xtick"], pad=8)
    ax.tick_params(axis="y", which="major", length=0, labelsize=PLOT_FONT["ytick"])
    ax.tick_params(axis="y", which="minor", length=0, left=False)


def _compact_count(value: int) -> str:
    if value >= 1000:
        return f"{value / 1000.0:.1f}K"
    return str(value)


def _legend_label(protocol: str, subq: int | None, show_subq: bool) -> str:
    label = PROTOCOL_LABEL[protocol]
    if show_subq and subq is not None:
        return f"{label} ({_compact_count(subq)} SubQ)"
    return label


def _training_iteration_x(points: list[dict]) -> np.ndarray:
    stages = [point["stage"] for point in points]
    numeric_stages = [int(stage) for stage in stages if stage.isdigit()]
    last_numeric = max(numeric_stages, default=0)
    values = []
    for stage in stages:
        if stage == "-2":
            values.append(0.0)
        elif stage == "-1":
            values.append(1.0)
        elif stage.startswith("R"):
            values.append(float(last_numeric + 6 + int(stage[1:])))
        else:
            values.append(float(int(stage) + 2))
    return np.array(values, dtype=float)


def _plot_workload(workload: str, curves: dict[str, dict], kind: str) -> Path:
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT["base"],
            "axes.labelsize": PLOT_FONT["label"],
            "axes.titlesize": PLOT_FONT["title"],
            "legend.fontsize": PLOT_FONT["legend"],
        }
    )
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    for protocol in PROTOCOLS:
        if protocol not in curves:
            continue
        entry = curves[protocol]
        points = entry["points"]
        y = np.array([point["norm_runtime"] for point in points], dtype=float)
        if kind == "wallclock":
            x = np.array([point["elapsed_min"] / 60.0 for point in points], dtype=float)
        else:
            x = _training_iteration_x(points)
        ax.plot(
            x,
            y,
            color=PROTOCOL_COLOR[protocol],
            linestyle=PROTOCOL_STYLE[protocol],
            lw=LINE_WIDTH,
            label=_legend_label(protocol, entry["subq"], kind == "data"),
        )

    xlabel = "Elapsed Time (h)" if kind == "wallclock" else "Training Iteration"
    _style_axis(ax, workload, xlabel)
    ax.set_xlim(left=0.0)
    if kind == "data":
        if workload in {"STACK", "TPC-H"}:
            ax.set_xlim(0.0, 34.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    slot = OUTPUT_SLOT[workload]
    suffix = "puretime_curve_smooth" if kind == "wallclock" else "train_curve_smooth"
    output = OUT_DIR / f"task1_standardmdp_rl_{slot}_{suffix}.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _save_shared_legend() -> Path:
    handles = [
        Line2D(
            [0],
            [0],
            color=PROTOCOL_COLOR[protocol],
            linestyle=PROTOCOL_STYLE[protocol],
            linewidth=LINE_WIDTH,
        )
        for protocol in PROTOCOLS
    ]
    labels = [PROTOCOL_LABEL[protocol] for protocol in PROTOCOLS]
    fig = plt.figure(figsize=(5.4, 0.55))
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="center",
        ncol=3,
        mode="expand",
        bbox_to_anchor=(0.04, 0.0, 0.92, 1.0),
        handlelength=3.0,
        handletextpad=0.6,
        columnspacing=2.0,
        fontsize=PLOT_FONT["legend"] + 2,
    )
    output = OUT_DIR / "learning_efficiency_legend.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_learning_curves()
    for workload in WORKLOADS:
        print(_plot_workload(workload, data[workload], "wallclock"))
    for workload in WORKLOADS:
        print(_plot_workload(workload, data[workload], "data"))
    print(_save_shared_legend())


if __name__ == "__main__":
    main()
