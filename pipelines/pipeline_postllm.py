#!/usr/bin/env python3
"""
pipeline.py

End-to-end Democritus v2.0 runner (from triples to reports):

  triples -> sweep LCMs -> score -> credibility bundle (reports)

This is the "whole kahuna" orchestrator.

Usage:
  python -m scripts.pipeline \
    --name WaPoChocolate \
    --triples relational_triples.jsonl \
    --outdir figs/wapo_chocolate \
    --topk 200 \
    --radii 1,2,3 \
    --maxnodes 10,20,30,40,60 \
    --topk-models 5 \
    --topk-claims 30 \
    --alpha 1.0 \
    --tier1 0.60 --tier2 0.30 \
    --anchors "chocolate,cocoa,theobromine,polyphenol,methylation,epigenetic,clock,lead,cadmium,sugar"
"""

import argparse
import shutil
from pathlib import Path
import subprocess

def run(cmd: list[str]) -> None:
    print("\n[PIPELINE] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Run name used in report filenames")
    ap.add_argument("--triples", required=True, help="relational_triples.jsonl (canonical input)")
    ap.add_argument("--outdir", required=True, help="Output directory for this run")

    # sweep settings
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--radii", default="1,2,3")
    ap.add_argument("--maxnodes", default="10,20,30,40,60")

    # scoring settings
    ap.add_argument("--lambda-edge", type=float, default=0.25)
    ap.add_argument("--local-only", action="store_true", help="Use local-only scoring if supported by scorer")

    # credibility bundle settings
    ap.add_argument("--topk-models", type=int, default=5)
    ap.add_argument("--topk-claims", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tier1", type=float, default=0.60)
    ap.add_argument("--tier2", type=float, default=0.30)
    ap.add_argument("--anchors", default="", help="Comma-separated keyword anchors (optional)")
    ap.add_argument("--title", default="", help="Optional executive summary title override")
    ap.add_argument("--dedupe-focus", action="store_true",
                help="Select at most one top model per unique focus (passed to make_credibility_bundle)")
    ap.add_argument("--require-anchor-in-focus", action="store_true",
                help="Only keep top models whose focus matches anchor keywords (passed to make_credibility_bundle)")
    ap.add_argument("--focus-blacklist-regex", default="",
                help="Regex; exclude models whose focus matches this pattern")
    # Phase II report UX enhancements (forwarded to make_credibility_bundle)
    ap.add_argument("--render-topk-pngs", action="store_true",
                    help="Render PNG diagrams for top-K selected LCMs into reports assets/")
    ap.add_argument("--assets-dir", default="assets",
                    help="Subdirectory under reports/ to store PNG assets (default: assets)")
    ap.add_argument("--png-dpi", type=int, default=200,
                    help="DPI for rendered PNGs (default: 200)")

    ap.add_argument("--write-deep-dive", action="store_true",
                    help="Write TS-WM Causal Deep Dive v1 markdown file")
    ap.add_argument("--deep-dive-max-bullets", type=int, default=8,
                    help="Max Tier-1 bullets shown in deep dive (default: 8)")
    return ap.parse_args()

def main():
    args = parse_args()
    triples_path = Path(args.triples)
    if not triples_path.exists():
        raise FileNotFoundError(triples_path)

    outdir = Path(args.outdir)
    sweep_dir = outdir / "sweep"
    reports_dir = outdir / "reports"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Copy triples into outdir as the canonical experiment artifact
    canon_triples = outdir / "relational_triples.jsonl"
    if canon_triples.resolve() != triples_path.resolve():
        shutil.copy2(triples_path, canon_triples)
        print(f"[PIPELINE] Copied triples -> {canon_triples}")

    # 1) Generate LCM JSONs (no scoring)
    run([
        "python", "-m", "scripts.sweep_lcm",
        "--triples", str(canon_triples),
        "--outdir", str(sweep_dir),
        "--topk", str(args.topk),
        "--radii", args.radii,
        "--maxnodes", args.maxnodes,
    ])

    # 2) Score all LCMs into sweep/scores.csv
    score_cmd = [
        "python", "-m", "scripts.score_lcms_dir",
        "--indir", str(sweep_dir),
        "--triples", str(canon_triples),
        "--out", "scores.csv",
        "--lambda-edge", str(args.lambda_edge),
    ]
    if args.local_only:
        score_cmd.append("--local-only")
    run(score_cmd)

    # 3) Produce reports bundle into reports/
    bundle_cmd = [
        "python", "-m", "scripts.make_credibility_bundle",
        "--scores", str(sweep_dir / "scores.csv"),
        "--triples", str(canon_triples),
        "--lcm-dir", str(sweep_dir),
        "--topk-models", str(args.topk_models),
        "--topk-claims", str(args.topk_claims),
        "--alpha", str(args.alpha),
        "--tier1", str(args.tier1),
        "--tier2", str(args.tier2),
        "--outdir", str(reports_dir),
        "--name", args.name,
    ]
    if args.dedupe_focus:
        bundle_cmd.append("--dedupe-focus")
    if args.require_anchor_in_focus:
        bundle_cmd.append("--require-anchor-in-focus")
    if args.anchors.strip():
        bundle_cmd += ["--keyword-anchors", args.anchors]
    if args.title.strip():
        bundle_cmd += ["--title", args.title]
    if args.focus_blacklist_regex.strip():
        bundle_cmd += ["--focus-blacklist-regex", args.focus_blacklist_regex]
        
    # New bundle options
    if args.render_topk_pngs:
        bundle_cmd.append("--render-topk-pngs")
        bundle_cmd += ["--assets-dir", args.assets_dir]
        bundle_cmd += ["--png-dpi", str(args.png_dpi)]

    if args.write_deep_dive:
        bundle_cmd.append("--write-deep-dive")
        bundle_cmd += ["--deep-dive-max-bullets", str(args.deep_dive_max_bullets)]

    run(bundle_cmd)

    print("\n[PIPELINE] DONE")
    print(f"  Outdir: {outdir}")
    print(f"  Triples: {canon_triples}")
    print(f"  Scores:  {sweep_dir / 'scores.csv'}")
    print(f"  Report:  {reports_dir / (args.name + '_credibility_report.md')}")
    print(f"  Exec:    {reports_dir / (args.name + '_executive_summary.md')}")

if __name__ == "__main__":
    main()
