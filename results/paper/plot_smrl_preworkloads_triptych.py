"""Render JOB / STACK / TPCH average delta plots side-by-side in one PDF.

The three panels share the same physical bar width (and bar spacing) by setting
each subplot's axes width proportional to its query count via gridspec
``width_ratios``. TPCH ends up as a narrow panel rather than a stretched-out
one with sparse bars.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG = ROOT / "results" / "paper"

from common import FONT_CFG  # noqa: E402
from revision_record import load_per_query_panels  # noqa: E402

WINDOW = 9
RIGHT_AXIS_OUTER_POS = 1.14
JOIN_LABELPAD = 0
INTER_LABELPAD = 4
OUT_DIR = PAPER_FIG / "figs"
WORKLOADS = ["JOB", "STACK", "TPCH"]

# Tuning knobs --------------------------------------------------------------
PER_QUERY_IN = 0.045   # physical inches per query (== bar+gap stride)
MIN_AXES_IN = 0.1      # keep TPCH readable without making its bars look too wide
LEFT_PAD_IN = 0.55     # space for left ylabel + ticks per panel
RIGHT_PAD_INNER = 0.36
RIGHT_PAD_OUTER = 0.5
FIG_HEIGHT_IN = 2.65
BAR_WIDTH = 0.85       # axes-fraction; with PER_QUERY_IN this fixes physical width
WSPACE = 0.15          # subplot horizontal gap (relative to mean axes width)
OUTER_RIGHT_POS = 1.5
DELTA_TICK_COLOR = "#1f6f43"
LEGEND_HANDLES = [
    Patch(facecolor="#2ecc71", edgecolor="none", label="Faster"),
    Patch(facecolor="#e74c3c", edgecolor="none", label="Slower"),
    Line2D(
        [0],
        [0],
        color="#3498db",
        linestyle="-",
        marker="o",
        markerfacecolor="white",
        linewidth=1.8,
        markersize=3.2,
        label="# joins",
    ),
    Line2D(
        [0],
        [0],
        color="#e67e22",
        linestyle=(0, (5, 2.5)),
        marker="^",
        markerfacecolor="white",
        linewidth=1.8,
        markersize=3.4,
        label="Inter. rows",
    ),
]


def smooth(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


def collect_panel(workload: str) -> dict:
    record_workload = "TPC-H" if workload == "TPCH" else workload
    rows = load_per_query_panels()[record_workload]
    qids = [row["qid"] for row in rows]
    delta_s = np.array([row["delta_s"] for row in rows], dtype=float)
    delta_min = np.array([row["delta_min"] for row in rows], dtype=float)
    delta_max = np.array([row["delta_max"] for row in rows], dtype=float)

    join_counts = np.array([row["join_count"] for row in rows], dtype=float)
    inter_rows = np.array([row["intermediate_rows"] for row in rows], dtype=float)
    join_smooth = smooth(join_counts)
    inter_safe = np.where(inter_rows > 0, inter_rows, 1.0)
    inter_smooth = np.where(smooth(inter_safe) < 1, 1.0, smooth(inter_safe))

    return {
        "workload": workload,
        "qids": qids,
        "delta_s": delta_s,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "join_smooth": join_smooth,
        "inter_smooth": inter_smooth,
    }


def main() -> None:
    panels = [collect_panel(w) for w in WORKLOADS]
    n_queries = [len(p["qids"]) for p in panels]

    # Use a shared y-range so the y=0 baseline aligns across all panels.
    global_ymin = min(float(np.min(p["delta_min"])) for p in panels)
    global_ymax = max(float(np.max(p["delta_max"])) for p in panels)
    y_pad = max(0.05 * (global_ymax - global_ymin), 0.5)
    global_ymin -= y_pad
    global_ymax += y_pad

    # Each panel's *axes* width is proportional to its query count, so a single
    # BAR_WIDTH (axes-fraction) renders to the same physical thickness in all
    # three subplots. Then add fixed left/right pads for labels.
    # axes_widths = [max(MIN_AXES_IN, n * PER_QUERY_IN) for n in n_queries]
    axes_widths = [n * PER_QUERY_IN for n in n_queries]

    panel_widths = [
        w + LEFT_PAD_IN + (RIGHT_PAD_OUTER if i == len(axes_widths) - 1 else RIGHT_PAD_INNER)
        for i, w in enumerate(axes_widths)
    ]
    fig_w = sum(panel_widths) + 0.4

    fig = plt.figure(figsize=(fig_w, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(1, len(panels), width_ratios=axes_widths, wspace=WSPACE)

    axes = []

    for idx, data in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        axes.append(ax)
        x = np.arange(len(data["qids"]))
        delta_s = data["delta_s"]
        colors = np.where(delta_s >= 0, "#2ecc71", "#e74c3c")
        bars = ax.bar(
            x,
            delta_s,
            color=colors,
            edgecolor="none",
            width=BAR_WIDTH,
            zorder=2,
        )
        yerr = np.vstack(
            [
                np.maximum(delta_s - data["delta_min"], 0.0),
                np.maximum(data["delta_max"] - delta_s, 0.0),
            ]
        )
        ax.errorbar(
            x,
            delta_s,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=1.8,
            capthick=0.9,
            zorder=3,
        )
        ax.axhline(y=0, color="black", linewidth=0.8, zorder=3)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
        if len(x) > 0:
            ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_xlabel("Query Index", fontsize=FONT_CFG["label"])
        ax.set_title(
            "TPC-H" if data["workload"] == "TPCH" else data["workload"],
            fontsize=FONT_CFG["title"],
            pad=4,
        )
        if idx == 0:
            ax.set_ylabel(
                "End-to-End Time\nReduction vs. PG (s)",
                fontsize=FONT_CFG["label"],
            )
        ax.tick_params(axis="x", labelsize=FONT_CFG["xtick"])
        ax.tick_params(axis="y", labelsize=FONT_CFG["ytick"], labelcolor=DELTA_TICK_COLOR)
        if data["workload"] == "TPCH":
            ax.set_yticks([0, 10, 20])
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax2 = ax.twinx()
        line1 = ax2.plot(
            x,
            data["join_smooth"],
            color="#3498db",
            linestyle="-",
            marker="o",
            markevery=WINDOW,
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
        )[0]
        if idx == len(panels) - 1:
            ax2.set_ylabel(
                "# joins",
                fontsize=FONT_CFG["label"],
                color="#3498db",
                labelpad=JOIN_LABELPAD,
            )
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.tick_params(axis="y", labelcolor="#3498db", labelsize=FONT_CFG["ytick"], pad=1)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", OUTER_RIGHT_POS))
        line2 = ax3.plot(
            x,
            data["inter_smooth"],
            color="#e67e22",
            linestyle=(0, (5, 2.5)),
            marker="^",
            markevery=WINDOW,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
        )[0]
        ax3.set_yscale("log")
        ax3.set_ylim(1e2, 1e7)
        if idx == len(panels) - 1:
            ax3.set_ylabel(
                "Intermediate rows (log)",
                fontsize=FONT_CFG["label"],
                color="#e67e22",
                labelpad=INTER_LABELPAD,
            )
            ax3.tick_params(axis="y", labelcolor="#e67e22", labelsize=FONT_CFG["ytick"], pad=2)
        else:
            ax3.tick_params(axis="y", which="both", right=False, labelright=False)
            ax3.set_yticks([])
            ax3.set_yticks([], minor=True)
        ax3.spines["top"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax3.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.30, top=0.85, wspace=WSPACE)
    fig.legend(
        handles=LEGEND_HANDLES,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
        fontsize=FONT_CFG["legend"] + 2,
        columnspacing=1.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "smrl_action_delta_all_avg.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"WINDOW={WINDOW} (smoothing window from main script)")


if __name__ == "__main__":
    main()
# cd ./results/paper
# python3 plot_smrl_preworkloads_triptych.py
