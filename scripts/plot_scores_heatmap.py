#!/usr/bin/env python3
"""
plot_score_heatmap.py

Heatmap B: Score landscape over hyperparameters (radius × max_nodes).

Input: a CSV produced by your sweep script, with columns:
  focus, radius, max_nodes, n_nodes, n_edges, score, lcm_json (optional)

Outputs:
  - aggregate heatmaps (mean/median/max score per (radius,max_nodes))
  - optional per-focus heatmaps for top-K focuses by best score

Usage:
  python scripts/plot_score_heatmap.py \
    --csv figs/sweep_wapo/scores.csv \
    --outdir figs/sweep_wapo/heatmaps \
    --topk-focus 12
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _heatmap(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    """
    pivot: index = radius, columns = max_nodes, values = score statistic
    """
    # Ensure sorted axes
    pivot = pivot.sort_index().sort_index(axis=1)

    data = pivot.to_numpy(dtype=float)
    fig = plt.figure(figsize=(7, 4.8))
    ax = plt.gca()

    im = ax.imshow(data, aspect="auto", origin="lower")

    # ticks/labels
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticklabels(list(pivot.index))

    ax.set_xlabel("max_nodes")
    ax.set_ylabel("radius")
    ax.set_title(title)

    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("score")

    # annotate cells with values (optional, but useful for papers)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="scores.csv from sweep_lcms.py")
    ap.add_argument("--outdir", required=True, help="output directory for heatmaps")
    ap.add_argument("--topk-focus", type=int, default=0,
                    help="also save per-focus heatmaps for top-K focuses by best score (0 disables)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required = {"focus", "radius", "max_nodes", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    # Coerce types
    df["radius"] = df["radius"].astype(int)
    df["max_nodes"] = df["max_nodes"].astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # ---------- Aggregate heatmaps ----------
    # Mean
    pivot_mean = df.pivot_table(
        index="radius", columns="max_nodes", values="score", aggfunc="mean"
    )
    _heatmap(pivot_mean, "Mean score over (radius × max_nodes)", outdir / "heatmap_mean.png")

    # Median
    pivot_median = df.pivot_table(
        index="radius", columns="max_nodes", values="score", aggfunc="median"
    )
    _heatmap(pivot_median, "Median score over (radius × max_nodes)", outdir / "heatmap_median.png")

    # Max (shows best-case performance of a setting)
    pivot_max = df.pivot_table(
        index="radius", columns="max_nodes", values="score", aggfunc="max"
    )
    _heatmap(pivot_max, "Max score over (radius × max_nodes)", outdir / "heatmap_max.png")

    # Also useful: fraction of NEGATIVE scores (or below threshold) as a 'gibberish indicator'
    # You can tune threshold; for now use <= 0.0 as "very weak"
    df["bad"] = (df["score"] <= 0.0).astype(int)
    pivot_bad = df.pivot_table(
        index="radius", columns="max_nodes", values="bad", aggfunc="mean"
    )
    _heatmap(pivot_bad, "Fraction of low-score models (score ≤ 0.0)", outdir / "heatmap_bad_fraction.png")

    # ---------- Per-focus heatmaps ----------
    if args.topk_focus and args.topk_focus > 0:
        best_by_focus = (
            df.groupby("focus")["score"].max().sort_values(ascending=False)
        )
        top_focuses = list(best_by_focus.head(args.topk_focus).index)

        per_dir = outdir / "per_focus"
        per_dir.mkdir(parents=True, exist_ok=True)

        for focus in top_focuses:
            sub = df[df["focus"] == focus]
            pivot = sub.pivot_table(index="radius", columns="max_nodes", values="score", aggfunc="max")
            safe = "".join(ch for ch in focus.lower() if ch.isalnum() or ch in (" ", "_", "-")).strip().replace(" ", "_")
            _heatmap(pivot, f"Max score for focus: {focus}", per_dir / f"{safe}.png")

    print(f"Wrote heatmaps to: {outdir}")


if __name__ == "__main__":
    main()
