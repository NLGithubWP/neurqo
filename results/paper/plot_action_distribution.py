from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import FONT_CFG
from revision_record import load_action_distribution_means


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOADS = ["JOB", "STACK", "TPCH"]
FONT_LOCAL = {
    "label": FONT_CFG["label"] + 6,
    "title": FONT_CFG["title"] + 6,
    "xtick": FONT_CFG["xtick"] + 6,
    "ytick": FONT_CFG["ytick"] + 6,
    "annot": FONT_CFG["annot"] + 9,
}
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "soft_teal",
    ["#ffffff", "#eef9fb", "#d7f0ee", "#abdccf", "#74c3ae"],
)
HEAT_CMAP.set_bad("#f3f3f3")


def _round_percent_distribution(freqs):
    keys = list(freqs)
    values = np.array([freqs[key] for key in keys], dtype=float)
    if not np.all(np.isfinite(values)) or float(np.sum(values)) <= 0:
        return freqs
    values = values * (100.0 / float(np.sum(values)))
    rounded = np.floor(values).astype(int)
    remainder = 100 - int(np.sum(rounded))
    order = np.argsort(-(values - rounded), kind="stable")
    for idx in order[:remainder]:
        rounded[idx] += 1
    return {key: float(value) for key, value in zip(keys, rounded)}


def plot_layered_heatmaps():
    revision_rows = {row["workload"]: row for row in load_action_distribution_means()}
    avg_rows = [revision_rows[workload] for workload in WORKLOADS]
    for row in avg_rows:
        for key in (
            "high_level_freqs",
            "select_level_freqs",
            "medium_level_freqs",
            "low_level_freqs",
        ):
            row[key] = _round_percent_distribution(row[key])

    if not avg_rows:
        return

    row_labels = ["TPC-H" if row["workload"] == "TPCH" else row["workload"] for row in avg_rows]

    # Combined hierarchy view.
    configs = [
        ("Dec", [("Apply", "Split"), ("Skip", "Non-split")], "high_level_freqs"),
        (
            "Sched",
            [(r"$\alpha=0$", "0.0"), (r"$\alpha=0.5$", "0.5"), (r"$\alpha=1$", "1.0")],
            "select_level_freqs",
        ),
        ("Enum", [("Native", "Default"), ("TOP-K", "TOPK")], "medium_level_freqs"),
        (
            "Execution",
            [("None", "None"), ("Filter", "Filter"), ("AJoin", "AJoin"), ("Filter+AJoin", "Filter+AJoin")],
            "low_level_freqs",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(8.8, 2.55),
        gridspec_kw={"width_ratios": [2, 3, 2, 4]},
    )
    last_im = None
    for idx, (ax, (title, columns, key)) in enumerate(zip(axes, configs)):
        mat_rows = []
        for row in avg_rows:
            vals = []
            for _, source_key in columns:
                if source_key == "TOPK":
                    vals.append(
                        row[key]["Split-search"] + row[key]["Top-5"] + row[key]["Top-10"]
                    )
                else:
                    vals.append(row[key][source_key])
            mat_rows.append(vals)
        mat = np.array(mat_rows, dtype=float)
        last_im = ax.imshow(mat, cmap=HEAT_CMAP, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(
            [label for label, _ in columns],
            fontsize=FONT_LOCAL["xtick"],
            rotation=21,
            ha="right",
            rotation_mode="anchor",
        )
        for tick in ax.get_xticklabels():
            tick.set_color("black")
        ax.set_yticks(np.arange(len(row_labels)))
        if idx == 0:
            ax.set_yticklabels(row_labels, fontsize=FONT_LOCAL["ytick"], color="black")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=FONT_LOCAL["label"], pad=6, color="black")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                label = f"{int(val)}" if np.isfinite(val) else "--"
                ax.text(j, i, label, ha="center", va="center", fontsize=FONT_LOCAL["annot"], color="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
    fig.subplots_adjust(left=0.045, right=0.925, bottom=0.18, top=0.84, wspace=0.18)
    cax = fig.add_axes([0.94, 0.20, 0.010, 0.58])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_LOCAL["ytick"])
    cbar.set_label("Frequency (%)", fontsize=FONT_LOCAL["label"], color="black")
    fig.savefig(OUT_DIR / "action_distribution_hml_mean_triptych.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_layered_heatmaps()


if __name__ == "__main__":
    main()
