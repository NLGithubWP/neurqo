import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_RESULTS = ROOT / "results" / "benchmark" / "overall_performance_comparison.csv"
OUT_PDF = OUT_DIR / "learning_inference_ratio.pdf"

FONT = {
    "title": 14.5,
    "label": 14.5,
    "tick": 11.2,
}

METHODS = ["FASTgres", "TONIC", "GenJoin", "HybridQO", "AutoSteer", "NQO"]
WORKLOADS = ["JOB", "STACK", "TPCH"]
WORKLOAD_LABELS = {"JOB": "JOB", "STACK": "STACK", "TPCH": "TPC-H"}

COLORS = {
    "FASTgres": "#8ecae6",
    "TONIC": "#f4a261",
    "GenJoin": "#f7f26b",
    "HybridQO": "#e7a3d1",
    "AutoSteer": "#98ee99",
    "NQO": "#b691e5",
}

HATCHES = {
    "FASTgres": "--",
    "TONIC": "++",
    "GenJoin": "//",
    "HybridQO": "..",
    "AutoSteer": "xx",
    "NQO": "\\\\",
}

COLUMNS = (
    ("JOB", "base_query", "JOB Base-query"),
    ("JOB", "leave_one_out", "JOB Leave-one-out"),
    ("JOB", "random", "JOB Random"),
    ("STACK", "base_query", "STACK Base-query"),
    ("STACK", "leave_one_out", "STACK Leave-one-out"),
    ("STACK", "random", "STACK Random"),
    ("TPCH", "random", "TPC-H Random"),
)


def _ws(cell: str) -> float:
    return float(cell.split("/", 1)[0].strip())


def build_ratio_df() -> pd.DataFrame:
    table = pd.read_csv(OVERALL_RESULTS).set_index("Method")
    rows = []
    for method in METHODS:
        without_method = f"{method} w/o inference time"
        for workload, protocol, column in COLUMNS:
            ws_with = _ws(table.loc[method, column])
            ws_without = _ws(table.loc[without_method, column])
            rows.append(
                {
                    "workload": workload,
                    "protocol": protocol,
                    "method": method,
                    "ratio": 100.0 * (1.0 - ws_with / ws_without),
                }
            )
    result = pd.DataFrame(rows)
    expected = {
        (workload, method): 1 if workload == "TPCH" else 3
        for workload in WORKLOADS
        for method in METHODS
    }
    actual = result.groupby(["workload", "method"]).size().to_dict()
    if actual != expected:
        raise ValueError(f"unexpected inference-ratio coverage: {actual}")
    return result


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["workload", "method"], as_index=False)["ratio"]
        .agg(mean_ratio="mean", min_ratio="min", max_ratio="max", splits="size")
    )


def style_lower_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT["tick"])
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20])
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def style_upper_axes(ax, workload):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT["tick"], bottom=False, labelbottom=False)
    if workload == "TPCH":
        ax.set_ylim(35, 42)
        ax.set_yticks([40])
    else:
        ax.set_ylim(80, 100)
        ax.set_yticks([90])
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def add_break_marks(ax_top, ax_bot):
    d = 0.012
    top_style = dict(
        transform=ax_top.transAxes,
        color="black",
        clip_on=False,
        linewidth=1.0,
    )
    bottom_style = dict(
        transform=ax_bot.transAxes,
        color="black",
        clip_on=False,
        linewidth=1.0,
    )
    ax_top.plot((-d, +d), (-d, +d), **top_style)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **top_style)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **bottom_style)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **bottom_style)


def main():
    raw = build_ratio_df()
    summary = summarize(raw)

    fig = plt.figure(figsize=(7.0, 2.65))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.0, 2.25],
        wspace=0.16,
        hspace=0.06,
    )
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_bot = [
        fig.add_subplot(gs[1, i], sharex=axes_top[i]) for i in range(3)
    ]
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.31, top=0.90)

    x = np.arange(len(METHODS))
    for panel, (ax_top, ax_bot, workload) in enumerate(
        zip(axes_top, axes_bot, WORKLOADS)
    ):
        indexed = summary[summary["workload"] == workload].set_index("method")
        means = np.array([indexed.loc[method, "mean_ratio"] for method in METHODS])
        mins = np.array([indexed.loc[method, "min_ratio"] for method in METHODS])
        maxes = np.array([indexed.loc[method, "max_ratio"] for method in METHODS])
        yerr = np.vstack([means - mins, maxes - means])

        for method_index, method in enumerate(METHODS):
            common = dict(
                width=0.68,
                color=COLORS[method],
                edgecolor="black",
                linewidth=1.0,
                hatch=HATCHES[method],
                zorder=3,
            )
            error = yerr[:, method_index : method_index + 1]
            ax_bot.bar(
                x[method_index],
                means[method_index],
                yerr=error,
                capsize=2.0,
                error_kw={"elinewidth": 0.9, "capthick": 0.9},
                **common,
            )
            ax_top.bar(
                x[method_index],
                means[method_index],
                yerr=error,
                capsize=2.0,
                error_kw={"elinewidth": 0.9, "capthick": 0.9},
                **common,
            )

        ax_top.set_title(WORKLOAD_LABELS[workload], fontsize=FONT["title"], pad=3)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(
            METHODS,
            rotation=36,
            ha="right",
            rotation_mode="anchor",
            fontsize=FONT["tick"],
        )
        style_upper_axes(ax_top, workload)
        style_lower_axes(ax_bot)
        add_break_marks(ax_top, ax_bot)
        if panel > 0:
            ax_bot.tick_params(labelleft=False)

    fig.text(
        0.0,
        0.59,
        "Inf. Overhead (%)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=FONT["label"],
    )

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    print(f"saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
