#!/usr/bin/env python3
"""
score_lcms_dir.py

Score a directory full of LCM JSON files using the text-evidence evaluator.
Designed to run AFTER you manually delete obvious LLM junk.

Inputs:
  - A directory containing LCM JSON files (from sweep_lcms.py)
  - A triples JSONL file used as evidence (relational_triples.jsonl)

Outputs:
  - scores.csv (default): one row per LCM with score + metadata
  - optionally: top.csv with top-K rows

Usage:
  python -m scripts.score_lcms_dir \
    --indir figs/sweep_wapo \
    --triples relational_triples.jsonl \
    --out scores.csv \
    --local-only \
    --lambda-edge 0.25 \
    --topk 50
"""

import argparse
import json
from pathlib import Path
import pandas as pd

# Import your scorer
from scripts.lcm_score_text import score_lcm_text


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory of LCM JSON files (post-cleanup)")
    ap.add_argument("--triples", required=True, help="relational_triples.jsonl evidence file")
    ap.add_argument("--pattern", default="*_r*_m*.json", help="Glob pattern for LCM JSON files")
    ap.add_argument("--out", default="scores.csv", help="Output CSV file (written in indir if relative)")
    ap.add_argument("--local-only", action="store_true",
                    help="Score using only evidence triples whose endpoints are inside the LCM nodes")
    ap.add_argument("--lambda-edge", type=float, default=0.25, help="Complexity penalty per edge")
    ap.add_argument("--topk", type=int, default=0, help="Also write top-K rows to top.csv (0 disables)")
    ap.add_argument("--min-edges", type=int, default=0,
                    help="Skip LCMs with fewer than this many edges (post-export edges)")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.exists():
        raise FileNotFoundError(indir)

    triples_path = Path(args.triples)
    if not triples_path.exists():
        raise FileNotFoundError(triples_path)

    # Resolve output path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = indir / out_path

    files = sorted(indir.glob(args.pattern))
    # avoid scoring index.csv etc.
    files = [p for p in files if p.is_file() and p.suffix == ".json"]

    if not files:
        print(f"[score_lcms_dir] No JSON files matched: {indir / args.pattern}")
        return

    rows = []
    n_skipped = 0
    for p in files:
        try:
            lcm = load_json(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            n_skipped += 1
            continue

        focus = lcm.get("focus", "")
        radius = lcm.get("radius", None)

        n_nodes = len(lcm.get("nodes", []))
        n_edges = len(lcm.get("edges", []))

        if args.min_edges and n_edges < args.min_edges:
            n_skipped += 1
            continue

        try:
            s = score_lcm_text(
                lcm,
                triples_path,
                lambda_edge=args.lambda_edge,
                local_only=args.local_only,
            )
        except TypeError:
            # If your score_lcm_text doesn't yet support local_only, fall back gracefully.
            s = score_lcm_text(
                lcm,
                triples_path,
                lambda_edge=args.lambda_edge,
            )

        rows.append({
            "file": p.name,
            "focus": focus,
            "radius": radius,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "score": float(s),
            "lcm_json": str(p),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[score_lcms_dir] No models scored (everything skipped or failed).")
        return

    df = df.sort_values("score", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[score_lcms_dir] Scored {len(df)} models ({n_skipped} skipped).")
    print(f"[score_lcms_dir] Wrote: {out_path}")

    if args.topk and args.topk > 0:
        top_path = out_path.with_name("top.csv")
        df.head(args.topk).to_csv(top_path, index=False)
        print(f"[score_lcms_dir] Wrote top-{args.topk}: {top_path}")


if __name__ == "__main__":
    main()
