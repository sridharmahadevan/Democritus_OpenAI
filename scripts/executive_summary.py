#!/usr/bin/env python3
"""
executive_summary.py

Generate a single-page "Executive Credibility Summary" with tiers
from:
  (1) scores.csv (top-K models)
  (2) credibility_claims_ranked.csv (ranked claims with credibility + evidence)

This is the final "smoke test" output (right-adjoint style presentation).

Usage:
  python -m scripts.executive_summary \
    --scores figs/antartica/scores.csv \
    --claims credibility_claims_ranked.csv \
    --topk-models 5 \
    --tier1 0.60 \
    --tier2 0.30 \
    --max-per-tier 10 \
    --out executive_credibility_summary_antarctica.md
"""

import argparse
from pathlib import Path
import pandas as pd


def truncate(s: str, n: int = 180) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def make_tier_block(df: pd.DataFrame, title: str, max_items: int) -> list[str]:
    lines = []
    lines.append(f"## {title}")
    lines.append("")
    if df.empty:
        lines.append("_No items in this tier._")
        lines.append("")
        return lines

    for i, row in enumerate(df.head(max_items).itertuples(index=False), start=1):
        # row fields are column names
        cred = float(getattr(row, "credibility"))
        src = getattr(row, "src")
        rel = getattr(row, "rel")
        dst = getattr(row, "dst")
        ex1 = getattr(row, "example_1") if "example_1" in df.columns else ""
        models = getattr(row, "models_supporting") if "models_supporting" in df.columns else ""

        lines.append(f"**{i}. ({cred:.2f}) {src} —{rel}→ {dst}**")

        ev = truncate(ex1, 180) if ex1 else ""
        if ev:
            lines.append(f"> {ev}")

        if isinstance(models, str) and models.strip():
            lines.append(f"- Supported by: {models}")

        lines.append("")

    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores.csv containing top scored LCMs")
    ap.add_argument("--claims", required=True, help="credibility_claims_ranked.csv from credibility_report.py")
    ap.add_argument("--topk-models", type=int, default=5, help="Number of top models to list")
    ap.add_argument("--tier1", type=float, default=0.60, help="Tier-1 credibility threshold (default 0.60)")
    ap.add_argument("--tier2", type=float, default=0.30, help="Tier-2 credibility threshold (default 0.30)")
    ap.add_argument("--max-per-tier", type=int, default=10, help="Max claims to show per tier")
    ap.add_argument("--title", default="Executive Credibility Summary — Antarctica Article (Democritus v2.0 smoke test)",
                    help="Report title")
    ap.add_argument("--out", required=True, help="Output markdown file (.md)")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    claims_path = Path(args.claims)
    out_path = Path(args.out)

    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if not claims_path.exists():
        raise FileNotFoundError(claims_path)

    scores_df = pd.read_csv(scores_path)
    claims_df = pd.read_csv(claims_path)

    # Validate required columns
    for col in ("focus", "score"):
        if col not in scores_df.columns:
            raise ValueError(f"scores.csv missing required column: {col}")
    for col in ("credibility", "src", "rel", "dst"):
        if col not in claims_df.columns:
            raise ValueError(f"claims CSV missing required column: {col}")

    scores_df["score"] = pd.to_numeric(scores_df["score"], errors="coerce")
    scores_df = scores_df.dropna(subset=["score"]).sort_values("score", ascending=False).head(args.topk_models)

    claims_df["credibility"] = pd.to_numeric(claims_df["credibility"], errors="coerce")
    claims_df = claims_df.dropna(subset=["credibility"]).sort_values(["credibility", "support_count"], ascending=False)

    # Tiering
    t1, t2 = args.tier1, args.tier2
    tier1 = claims_df[claims_df["credibility"] >= t1].copy()
    tier2 = claims_df[(claims_df["credibility"] < t1) & (claims_df["credibility"] >= t2)].copy()
    tier3 = claims_df[claims_df["credibility"] < t2].copy()

    md = []
    md.extend(make_tier_block(tier1, "Tier 1 Claims", args.max_per_tier))
    md.extend(make_tier_block(tier2, "Tier 2 Claims", args.max_per_tier))
    md.extend(make_tier_block(tier3, "Tier 3 Claims", args.max_per_tier))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[executive_summary] wrote: {out_path}")


if __name__ == "__main__":
    main()
