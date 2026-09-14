#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "benchmark" / "nqo" / "nqo_decomposition_depth.csv"
OUTPUT_DIR = ROOT / "results" / "paper" / "figs"

PANELS = {
    "JOB": ["29a", "22c", "33a"],
    "STACK": ["q3_q3-099", "q2_q2-012", "q13_935e2051"],
}
COLORS = ["#e36a00", "#1f77b4", "#12a777"]
MARKERS = ["o", "s", "^"]
LINESTYLES = ["-", "--", "-."]
LEGEND_LABELS = {
    "q3_q3-099": "q3_099",
    "q2_q2-012": "q2_012",
    "q13_935e2051": "q13_93",
}
FONT = {
    "xlabel": 14.5,
    "ylabel": 12.5,
    "legend": 11.2,
    "xtick": 12.0,
    "ytick": 10.8,
}
Y_AXIS = {
    "JOB": ([0.05, 0.1, 0.3, 1.0, 3.0], ["0.05", "0.1", "0.3", "1", "3"]),
    "STACK": (
        [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        ["0.01", "0.03", "0.1", "0.3", "1", "3"],
    ),
}


def load_curves() -> dict[tuple[str, str], list[tuple[int, float]]]:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    curves: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        if row["method"] != "Fixed-depth":
            continue
        dataset = row["dataset"]
        query_id = row["query_id"]
        if dataset not in PANELS or query_id not in PANELS[dataset]:
            continue
        if row["status"] != "ok":
            raise ValueError(f"non-ok result for {dataset}/{query_id}: {row['status']}")
        speedup = float(row["speedup"])
        if speedup <= 0:
            raise ValueError(f"non-positive speedup for {dataset}/{query_id}")
        curves.setdefault((dataset, query_id), []).append(
            (int(row["executed_units"]), 1.0 / speedup)
        )

    expected = {(dataset, query_id) for dataset, queries in PANELS.items() for query_id in queries}
    if set(curves) != expected:
        missing = sorted(expected - set(curves))
        extra = sorted(set(curves) - expected)
        raise ValueError(f"unexpected curve coverage; missing={missing}, extra={extra}")

    for points in curves.values():
        points.sort()
        units = [unit for unit, _ in points]
        if len(units) != len(set(units)):
            raise ValueError(f"duplicate executed_units values: {units}")
    return curves


def style_axis(ax: plt.Axes, dataset: str, max_units: int) -> None:
    y_ticks, y_labels = Y_AXIS[dataset]
    ax.set_yscale("log")
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(y_labels))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axhline(1.0, color="#777777", linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.set_xlim(0.75, max_units + 0.25)
    ax.set_xticks(range(1, max_units + 1))
    ax.set_xlabel("# Executed Units", fontsize=FONT["xlabel"])
    ax.set_ylabel("Norm. End-to-End\nTime [log scale]", fontsize=FONT["ylabel"])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.9, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", length=0, labelsize=FONT["xtick"], pad=4)
    ax.tick_params(axis="y", which="both", length=0, labelsize=FONT["ytick"])


def main() -> None:
    curves = load_curves()
    plt.rcParams.update(
        {
            "font.size": FONT["ytick"],
            "axes.labelsize": FONT["ylabel"],
            "legend.fontsize": FONT["legend"],
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[plt.Figure, Path]] = []
    for dataset, query_ids in PANELS.items():
        fig, ax = plt.subplots(figsize=(3.9, 2.15))
        max_units = 0
        for query_index, query_id in enumerate(query_ids):
            points = curves[(dataset, query_id)]
            x = [unit for unit, _ in points]
            y = [runtime for _, runtime in points]
            max_units = max(max_units, max(x))
            ax.plot(
                x,
                y,
                color=COLORS[query_index],
                linestyle=LINESTYLES[query_index],
                marker=MARKERS[query_index],
                markersize=6.2,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linewidth=3.5,
                label=LEGEND_LABELS.get(query_id, query_id),
                zorder=3,
            )
            best_index = min(range(len(y)), key=y.__getitem__)
            ax.plot(
                [x[best_index]],
                [y[best_index]],
                linestyle="none",
                marker=MARKERS[query_index],
                markersize=7.4,
                markerfacecolor="#d62728",
                markeredgecolor="black",
                markeredgewidth=0.9,
                zorder=4,
            )
        style_axis(ax, dataset, max_units)
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            columnspacing=0.7,
            handlelength=2.0,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.27, right=0.92, bottom=0.21, top=0.78)
        output = OUTPUT_DIR / f"decomposition_depth_{dataset.lower()}.pdf"
        figures.append((fig, output))

    tight_boxes = []
    for fig, _ in figures:
        fig.canvas.draw()
        tight_boxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    common_bbox = Bbox.union(tight_boxes).padded(0.02)

    for fig, output in figures:
        fig.savefig(output, bbox_inches=common_bbox)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
