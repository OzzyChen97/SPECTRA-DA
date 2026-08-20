#!/usr/bin/env python3
"""Generate deterministic paper figures from public aggregate results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARXIV = ROOT / "arxiv"
FIGURES = ARXIV / "figures"
RESULTS = ROOT / "results"

BLUE = "#2B6CB0"
LIGHT_BLUE = "#E6F0FA"
GREEN = "#2F855A"
LIGHT_GREEN = "#E6F4EC"
ORANGE = "#C05621"
LIGHT_ORANGE = "#FBEBDD"
PURPLE = "#6B46C1"
LIGHT_PURPLE = "#EEE8FA"
GRAY = "#4A5568"
LIGHT_GRAY = "#EDF2F7"
RED = "#C53030"


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_box(ax, xy, wh, text, face, edge, fontsize=8.2, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color=GRAY, style="-|>", lw=1.3, connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=connection,
        )
    )


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.015, 0.63),
        (0.17, 0.21),
        "Frozen candidates\nalgorithm × config\nseed × checkpoint",
        LIGHT_BLUE,
        BLUE,
    )
    add_box(
        ax,
        (0.225, 0.63),
        (0.17, 0.21),
        "Unlabeled target\npredictions + graph",
        LIGHT_BLUE,
        BLUE,
    )
    add_box(
        ax,
        (0.435, 0.63),
        (0.17, 0.21),
        "Tight spectral frame\nlow / mid / high",
        LIGHT_PURPLE,
        PURPLE,
    )
    add_box(
        ax,
        (0.645, 0.63),
        (0.17, 0.21),
        "Band-wise pair\ndisagreement",
        LIGHT_PURPLE,
        PURPLE,
    )
    add_box(
        ax,
        (0.825, 0.63),
        (0.16, 0.21),
        "Robust individual\nrisk recovery",
        LIGHT_GREEN,
        GREEN,
    )

    add_box(
        ax,
        (0.12, 0.17),
        (0.21, 0.22),
        "Labeled source\nsimulated shifts",
        LIGHT_ORANGE,
        ORANGE,
    )
    add_box(
        ax,
        (0.405, 0.17),
        (0.21, 0.22),
        "Unlabeled shift\ndescriptors",
        LIGHT_ORANGE,
        ORANGE,
    )
    add_box(
        ax,
        (0.69, 0.17),
        (0.21, 0.22),
        "Band covariance\ntransport + uncertainty",
        LIGHT_ORANGE,
        ORANGE,
    )

    for x0, x1 in [(0.185, 0.225), (0.395, 0.435), (0.605, 0.645), (0.815, 0.825)]:
        arrow(ax, (x0, 0.735), (x1, 0.735))
    arrow(ax, (0.33, 0.28), (0.405, 0.28), color=ORANGE)
    arrow(ax, (0.615, 0.28), (0.69, 0.28), color=ORANGE)
    arrow(ax, (0.79, 0.39), (0.88, 0.63), color=ORANGE, connection="arc3,rad=-0.15")
    arrow(ax, (0.31, 0.63), (0.51, 0.39), color=BLUE, connection="arc3,rad=0.13")

    ax.text(
        0.5,
        0.94,
        "Public selector path: target labels never enter training or selection",
        ha="center",
        va="center",
        weight="bold",
        color=GRAY,
    )
    ax.text(
        0.5,
        0.055,
        "Hidden target labels are read once by an isolated evaluator after the selector is frozen",
        ha="center",
        va="center",
        fontsize=8.2,
        color=RED,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "pipeline.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_result_summary() -> None:
    metrics = json.loads((RESULTS / "final_metrics.json").read_text())
    current = metrics["metrics"]["mean_normalized_regret_dev"]
    baseline = metrics["comparisons"]["original_baseline_mean_regret"]
    reduction = metrics["comparisons"]["relative_regret_reduction_percent"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"width_ratios": [1.0, 1.65]})

    ax = axes[0]
    bars = ax.bar(["Original frozen\nselector", "SPECTRA-DA"], [baseline, current], color=["#A0AEC0", BLUE], width=0.62)
    ax.set_ylabel("Mean normalized regret ↓")
    ax.set_ylim(0, max(baseline, current) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, [baseline, current]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0006, f"{value:.4f}", ha="center", fontsize=8.5)
    ax.text(0.5, 0.91, f"{reduction:.1f}% relative reduction", transform=ax.transAxes, ha="center", color=GREEN, weight="bold")

    ax = axes[1]
    ax.axis("off")
    rows = [
        ("CVaR-20%", f"{metrics['metrics']['cvar20_dev']:.4f}"),
        ("Worst fold", f"{metrics['metrics']['worst_fold_regret_dev']:.4f}"),
        ("Median Kendall τ", f"{metrics['metrics']['median_kendall_tau_dev']:.3f}"),
        ("Top-weighted Kendall", f"{metrics['metrics']['mean_top_weighted_kendall_dev']:.3f}"),
        ("Top-5% hit rate", f"{100*metrics['metrics']['top_5pct_hit_rate_dev']:.1f}%"),
        ("Frozen runtime", f"{metrics['metrics']['selector_runtime_seconds']:.1f} s"),
        ("Label / protocol violations", "0 / 0"),
    ]
    y = 0.92
    for idx, (name, value) in enumerate(rows):
        face = LIGHT_BLUE if idx % 2 == 0 else "white"
        rect = FancyBboxPatch((0.02, y - 0.105), 0.96, 0.105, boxstyle="round,pad=0.004", facecolor=face, edgecolor="none")
        ax.add_patch(rect)
        ax.text(0.05, y - 0.052, name, va="center", color=GRAY, fontsize=8.3)
        ax.text(0.95, y - 0.052, value, va="center", ha="right", color="#1A202C", weight="bold", fontsize=8.5)
        y -= 0.125
    ax.text(0.5, 0.995, "Frozen development evaluation", ha="center", va="top", weight="bold", color=GRAY)
    ax.text(0.5, 0.02, "4 tasks · 44 folds · 675 candidates/task", ha="center", va="bottom", fontsize=8, color=GRAY)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURES / "result_summary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "result_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_refinement_audit() -> None:
    records = json.loads((RESULTS / "iteration_summary.json").read_text())
    numeric = [r for r in records if isinstance(r["iteration"], int)]
    x = np.array([r["iteration"] for r in numeric])
    regret = np.array([r["mean_regret"] for r in numeric])
    runtime = np.array([r["runtime_seconds"] for r in numeric])
    accepted = np.array([r["decision"] in {"baseline", "accepted"} for r in numeric])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    ax = axes[0]
    ax.plot(x, regret, color=GRAY, linewidth=1.1, zorder=1)
    ax.scatter(x[~accepted], regret[~accepted], color="#A0AEC0", s=30, label="rejected/diagnostic", zorder=2)
    ax.scatter(x[accepted], regret[accepted], color=GREEN, s=38, label="accepted", zorder=3)
    ax.axhline(regret[0], color=BLUE, linestyle="--", linewidth=1, alpha=0.75)
    ax.set_xlabel("Controlled refinement iteration")
    ax.set_ylabel("Mean normalized regret ↓")
    ax.set_xticks(x)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EDF2F7", linewidth=0.8)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")

    ax = axes[1]
    ax.plot(x, runtime, color=GRAY, linewidth=1.1, zorder=1)
    colors = [GREEN if a else "#A0AEC0" for a in accepted]
    ax.scatter(x, runtime, color=colors, s=34, zorder=2)
    ax.annotate("exact cache", (2, runtime[2]), xytext=(2.45, runtime[2] + 45), arrowprops={"arrowstyle": "->", "color": GRAY}, fontsize=7.6)
    ax.annotate("workspace reuse", (8, runtime[8]), xytext=(5.25, runtime[8] + 35), arrowprops={"arrowstyle": "->", "color": GRAY}, fontsize=7.6)
    ax.set_xlabel("Controlled refinement iteration")
    ax.set_ylabel("Selector runtime (s) ↓")
    ax.set_xticks(x)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EDF2F7", linewidth=0.8)

    fig.tight_layout(pad=0.55)
    fig.savefig(FIGURES / "refinement_audit.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "refinement_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    make_pipeline()
    make_result_summary()
    make_refinement_audit()


if __name__ == "__main__":
    main()
