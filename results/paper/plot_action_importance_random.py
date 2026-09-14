#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from revision_record import load_action_importance

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUT = OUT_DIR / "action_importance_random_speedup.pdf"

WORKLOADS = ["job", "stack", "tpch"]
WORKLOAD_LABEL = {"job": "JOB", "stack": "STACK", "tpch": "TPC-H"}
SETTINGS = ["nqo", "wo_split", "wo_search", "wo_lip", "wo_aja"]
SETTING_LABEL = {
    "nqo": "NQO",
    "wo_split": "w/o Dec",
    "wo_search": "w/o TOP-K",
    "wo_lip": "w/o Filter",
    "wo_aja": "w/o AJoin",
}
COLORS = {
    "nqo": "#f4e66a",
    "wo_split": "#f4a261",
    "wo_search": "#6ab7ff",
    "wo_lip": "#8ed081",
    "wo_aja": "#b69ae6",
}
HATCHES = {
    "nqo": "//",
    "wo_split": "\\\\",
    "wo_search": "oo",
    "wo_lip": "xx",
    "wo_aja": "..",
}
FONT = {
    "label": 10.5,
    "legend": 7.6,
    "xtick": 8.8,
    "ytick": 8.8,
}

def aggregate() -> list[dict]:
    data = load_action_importance()
    rows: list[dict] = []
    for workload in WORKLOADS:
        for setting in SETTINGS:
            rows.append(
                {
                    "workload": workload,
                    "setting": setting,
                    "baseline_ms": "",
                    "total_ms": "",
                    "speedup": data[workload][setting],
                }
            )
    return rows


def plot(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.size": FONT["ytick"],
            "axes.labelsize": FONT["label"],
            "legend.fontsize": FONT["legend"],
        }
    )

    fig, ax = plt.subplots(figsize=(3.5, 1.8))

    centers = np.array([0.0, 0.82, 1.64], dtype=float)
    width = 0.13

    row_map = {(r["workload"], r["setting"]): r for r in rows}
    offsets = {}
    for workload in WORKLOADS:
        active = [
            setting
            for setting in SETTINGS
            if row_map[(workload, setting)]["speedup"] is not None
        ]
        offsets[workload] = {
            setting: (index - (len(active) - 1) / 2) * width
            for index, setting in enumerate(active)
        }

    for i, setting in enumerate(SETTINGS):
        positions = []
        vals = []
        for j, workload in enumerate(WORKLOADS):
            value = row_map[(workload, setting)]["speedup"]
            if value is None:
                continue
            positions.append(centers[j] + offsets[workload][setting])
            vals.append(value)
        ax.bar(
            positions,
            vals,
            width=width,
            color=COLORS[setting],
            edgecolor="black",
            linewidth=1.4,
            hatch=HATCHES[setting],
            label=SETTING_LABEL[setting],
            zorder=3,
        )

    ax.axhline(1.0, color="#6f6f6f", linewidth=1.2, zorder=5)
    ax.set_xticks(centers)
    ax.set_xticklabels([WORKLOAD_LABEL[w] for w in WORKLOADS], fontsize=FONT["xtick"])
    ax.set_ylabel("Workload Speedup", fontsize=FONT["label"])
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.grid(axis="y", color="#d9d9d9", linewidth=1.0, zorder=1)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=FONT["xtick"], pad=4)
    ax.tick_params(axis="y", length=0, labelsize=FONT["ytick"])
    ax.legend(
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.24),
        columnspacing=0.8,
        handlelength=1.3,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.14, top=0.73)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = aggregate()
    plot(rows)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
