from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from revision_record import load_transferability


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "results" / "paper" / "figs" / "task21_single_source_zero_shot_speedup.pdf"

WORKLOADS = ["job", "stack", "tpch"]
LABELS = {"job": "JOB", "stack": "STACK", "tpch": "TPC-H"}
FONT = {
    "label": 10.5,
    "legend": 7.6,
    "xtick": 8.8,
    "ytick": 8.8,
    "annot": 6.8,
}


def plot():
    bars = load_transferability()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "same": "#fff97a",
        "mixed": "#8fd3ff",
        "job": "#f2a9d4",
        "stack": "#98f59a",
        "tpch": "#b58be8",
    }
    hatches = {
        "same": "//",
        "mixed": "oo",
        "job": ".",
        "stack": "xx",
        "tpch": "\\\\",
    }

    handles = {
        "same": Patch(facecolor=colors["same"], edgecolor="black", linewidth=1.4, hatch=hatches["same"], label="Same-wld"),
        "mixed": Patch(facecolor=colors["mixed"], edgecolor="black", linewidth=1.4, hatch=hatches["mixed"], label="Mixed-wld"),
        "job": Patch(facecolor=colors["job"], edgecolor="black", linewidth=1.4, hatch=hatches["job"], label="Zero-shot, src=JOB"),
        "stack": Patch(facecolor=colors["stack"], edgecolor="black", linewidth=1.4, hatch=hatches["stack"], label="Zero-shot, src=STACK"),
        "tpch": Patch(facecolor=colors["tpch"], edgecolor="black", linewidth=1.4, hatch=hatches["tpch"], label="Zero-shot, src=TPC-H"),
        "blank": Line2D([], [], linestyle="", linewidth=0, label=""),
    }

    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    x = np.array([0.0, 0.90, 1.80])
    width = 0.15

    order_map = {
        "job": ["same", "mixed", "stack", "tpch"],
        "stack": ["same", "mixed", "job", "tpch"],
        "tpch": ["same", "mixed", "job", "stack"],
    }

    max_val = 0.0
    for i, tgt in enumerate(WORKLOADS):
        keys = order_map[tgt]
        offsets = (np.arange(len(keys)) - (len(keys) - 1) / 2.0) * width
        for off, key in zip(offsets, keys):
            val = float(bars[tgt][key])
            max_val = max(max_val, val)
            rects = ax.bar(
                x[i] + off,
                val,
                width,
                color=colors[key],
                edgecolor="black",
                linewidth=1.4,
                hatch=hatches[key],
            )

    ax.axhline(1.0, color="#6f6f6f", linewidth=1.2)
    ax.set_ylabel("Workload Speedup", fontsize=FONT["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[w] for w in WORKLOADS], fontsize=FONT["xtick"])
    ax.tick_params(axis="y", labelsize=FONT["ytick"])
    ax.set_ylim(0, max(2.1, max_val + 0.15))
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.grid(axis="y", color="#d9d9d9", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[handles["same"], handles["mixed"], handles["blank"], handles["job"], handles["stack"], handles["tpch"]],
        loc="upper center",
        bbox_to_anchor=(0.50, 1.35),
        ncol=2,
        fontsize=FONT["legend"],
        frameon=False,
        columnspacing=0.9,
        handlelength=1.3,
    )

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.15, top=0.68)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    plot()
