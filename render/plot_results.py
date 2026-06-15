"""Render the README result charts from a leaderboard JSON.

Produces docs/assets/results.png (base vs tuned on Read F1 + Reason accuracy).
Warm palette matching the site; legend placed above the plot so it never
collides with bar labels.

Usage:
  .venv/bin/python render/plot_results.py data/results/leaderboard_v4_synth.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "results.png"

PAPER = "#FBF7F0"
INK = "#2B2521"
GRAY = "#CABF AC".replace(" ", "")  # warm stone, the base model
RED = "#C1452A"                      # terracotta, the tuned model
MUTED = "#9C9286"

# prefer a warm system serif/sans if present, else default
for fam in ("Inter", "Helvetica Neue", "Arial"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break


def load(path: str) -> dict:
    rows = {r["model"]: r for r in json.loads(Path(path).read_text())}
    return rows


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/results/leaderboard_v4_synth.json")
    tuned_key = sys.argv[2] if len(sys.argv) > 2 else "tuned_v4"
    rows = load(src)
    base = rows["qwen3b"]
    tuned = rows[tuned_key]

    metrics = ["Read F1", "Reason accuracy"]
    base_vals = [base["read_f1"], base["reason_e2e"]]
    tuned_vals = [tuned["read_f1"], tuned["reason_e2e"]]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    x = [0, 1.25]
    w = 0.42
    b1 = ax.bar([i - w / 2 for i in x], base_vals, w, label="base Qwen2.5-VL-3B",
                color=GRAY, edgecolor="none", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], tuned_vals, w, label="tuned (QLoRA)",
                color=RED, edgecolor="none", zorder=3)

    # random-verdict baseline (4-way guess) — only meaningful for reasoning
    ax.axhline(0.25, ls=(0, (5, 4)), lw=1.2, color=MUTED, zorder=2)
    ax.text((x[0] + x[1]) / 2, 0.275, "random verdict (0.25)", va="bottom", ha="center",
            fontsize=9, color=MUTED)

    for bars, vals in ((b1, base_vals), (b2, tuned_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.025, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)

    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13, color=INK)
    ax.set_ylabel("score", fontsize=11, color=INK)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color="#E7DECF", lw=1, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Reading SF parking signs: base vs QLoRA-tuned Qwen2.5-VL-3B",
                 fontsize=13, fontweight="bold", color=INK, pad=34)

    # legend ABOVE the axes — never overlaps a bar label
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=11, handlelength=1.3, columnspacing=2.2)

    fig.tight_layout()
    fig.savefig(OUT, facecolor=PAPER, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"base  read={base_vals[0]:.3f} reason={base_vals[1]:.3f}")
    print(f"tuned read={tuned_vals[0]:.3f} reason={tuned_vals[1]:.3f}")


if __name__ == "__main__":
    main()
