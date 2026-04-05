#!/usr/bin/env python3
"""
credibility_report.py

Right-adjoint prototype: (scored LCMs) -> (ranked causal claim credibility report).

Inputs:
  - scores.csv              : output from score_lcms_dir.py (or any similar scorer)
  - relational_triples.jsonl: Democritus triples as evidence (subj/obj/rel/statement)
  - LCM JSONs               : local causal models referenced by scores.csv

Outputs:
  - credibility_report.md
  - credibility_claims_ranked.csv

Example:
  python -m scripts.credibility_report \
    --scores figs/antartica/scores.csv \
    --triples relational_triples.jsonl \
    --lcm-dir figs/antartica \
    --topk-models 5 \
    --topk-claims 30 \
    --alpha 1.0 \
    --out-md credibility_report_antarctica.md \
    --out-csv credibility_claims_ranked.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


Edge = Tuple[str, str, str]  # (src, dst, rel)


def softmax(scores: List[float], alpha: float = 1.0) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(alpha * (s - m)) for s in scores]
    Z = sum(exps) if exps else 1.0
    return [e / Z for e in exps]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_triple_index(triples_path: Path, max_examples_per_edge: int = 3) -> Tuple[Dict[Edge, int], Dict[Edge, List[str]]]:
    """
    Build (edge->count) and (edge->example statements) from triples.jsonl.
    """
    counts: Dict[Edge, int] = {}
    examples: Dict[Edge, List[str]] = {}

    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("subj") or r.get("source")
            o = r.get("obj") or r.get("target")
            rel = r.get("rel") or r.get("relation") or ""
            stmt = r.get("statement") or ""

            if not s or not o:
                continue

            key: Edge = (s, o, rel)
            counts[key] = counts.get(key, 0) + 1

            if stmt:
                lst = examples.setdefault(key, [])
                if len(lst) < max_examples_per_edge:
                    lst.append(stmt)

    return counts, examples


def canonical_edge(e: dict) -> Edge:
    return (e["src"], e["dst"], e.get("rel", "") or "")


def summarize_lcm_hub(lcm: dict, k: int = 8) -> Tuple[str, str, List[dict]]:
    """
    Choose a "hub" as node with max outgoing edges; return hub and its outgoing edges.
    """
    focus = lcm.get("focus", "")
    edges = lcm.get("edges", [])

    outdeg: Dict[str, int] = {}
    for e in edges:
        outdeg[e["src"]] = outdeg.get(e["src"], 0) + 1

    hub = max(outdeg, key=outdeg.get) if outdeg else focus
    hub_edges = [e for e in edges if e["src"] == hub][:k]
    return focus, hub, hub_edges


def truncate(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores.csv (must include columns: file, focus, score)")
    ap.add_argument("--triples", required=True, help="relational_triples.jsonl evidence file")
    ap.add_argument("--lcm-dir", required=True, help="Directory containing the LCM JSON files")
    ap.add_argument("--topk-models", type=int, default=5, help="Use top-K models by score")
    ap.add_argument("--topk-claims", type=int, default=30, help="Show top-K ranked claims in report")
    ap.add_argument("--alpha", type=float, default=1.0, help="Softmax temperature for model weights")
    ap.add_argument("--out-md", default="credibility_report.md", help="Output markdown report")
    ap.add_argument("--out-csv", default="credibility_claims_ranked.csv", help="Output CSV ranked claims")
    ap.add_argument("--drop-anchor-rels", action="store_true",
                    help="Drop has_subj/has_obj edges if they somehow exist in LCM JSONs")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    triples_path = Path(args.triples)
    lcm_dir = Path(args.lcm_dir)

    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if not triples_path.exists():
        raise FileNotFoundError(triples_path)
    if not lcm_dir.exists():
        raise FileNotFoundError(lcm_dir)

    df = pd.read_csv(scores_path)
    for col in ("file", "focus", "score"):
        if col not in df.columns:
            raise ValueError(f"scores.csv missing required column: {col}")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).sort_values("score", ascending=False)

    top_df = df.head(args.topk_models).copy()
    if len(top_df) == 0:
        raise ValueError("No models found in scores.csv after filtering.")

    # Load selected LCMs
    models = []
    for _, row in top_df.iterrows():
        fname = str(row["file"])
        lcm_path = (lcm_dir / fname) if not Path(fname).is_absolute() else Path(fname)
        if not lcm_path.exists():
            # try lcm_json column if present
            if "lcm_json" in df.columns:
                cand = Path(row.get("lcm_json", ""))
                if cand.exists():
                    lcm_path = cand
                else:
                    raise FileNotFoundError(f"LCM file not found: {fname} (and lcm_json missing/invalid)")
            else:
                raise FileNotFoundError(f"LCM file not found: {fname} (and lcm_json column not present)")

        lcm = load_json(lcm_path)
        score = float(row["score"])
        models.append({"file": fname, "path": lcm_path, "lcm": lcm, "score": score})

    # Compute model weights
    weights = softmax([m["score"] for m in models], alpha=args.alpha)
    for m, w in zip(models, weights):
        m["weight"] = float(w)

    # Evidence index
    edge_counts, edge_examples = load_triple_index(triples_path, max_examples_per_edge=3)

    # Aggregate claim credibility across models
    anchor_rels = {"has_subj", "has_obj"}
    edge_info: Dict[Edge, Dict[str, Any]] = {}

    for m in models:
        lcm = m["lcm"]
        w = m["weight"]
        focus = lcm.get("focus", "")

        for e in lcm.get("edges", []):
            if args.drop_anchor_rels and (e.get("rel", "") in anchor_rels):
                continue
            key = canonical_edge(e)
            if key not in edge_info:
                edge_info[key] = {"cred": 0.0, "models": set()}
            edge_info[key]["cred"] += w
            edge_info[key]["models"].add(focus)

    claim_rows = []
    for (s, o, rel), info in edge_info.items():
        ex_list = edge_examples.get((s, o, rel), [])
        claim_rows.append({
            "credibility": float(info["cred"]),
            "src": s,
            "rel": rel,
            "dst": o,
            "support_count": int(edge_counts.get((s, o, rel), 0)),
            "models_supporting": "; ".join(sorted(info["models"])),
            "example_1": ex_list[0] if len(ex_list) > 0 else "",
            "example_2": ex_list[1] if len(ex_list) > 1 else "",
            "example_3": ex_list[2] if len(ex_list) > 2 else "",
        })

    claims_df = pd.DataFrame(claim_rows).sort_values(
        ["credibility", "support_count"], ascending=False
    ).reset_index(drop=True)

    # Write ranked claims CSV
    out_csv = Path(args.out_csv)
    out_csv.write_text("", encoding="utf-8")  # ensure writable location
    claims_df.to_csv(out_csv, index=False)

    # Write markdown report
    out_md = Path(args.out_md)

    lines: List[str] = []
    lines.append("# Democritus Credibility Report")
    lines.append("")
    lines.append("This report aggregates the **top-scoring local causal models (LCMs)** and produces a **ranked list of causal claims** (directed edges) supported by those models.")
    lines.append("")
    lines.append("## Models used")
    lines.append("")
    lines.append("| Rank | Focus | Score | Weight | Nodes | Edges | File |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")

    for i, m in enumerate(sorted(models, key=lambda x: x["score"], reverse=True), start=1):
        lcm = m["lcm"]
        lines.append(
            f"| {i} | {lcm.get('focus','')} | {m['score']:.3f} | {m['weight']:.3f} | "
            f"{len(lcm.get('nodes',[]))} | {len(lcm.get('edges',[]))} | `{m['path'].name}` |"
        )

    lines.append("")
    lines.append("## Ranked causal claims")
    lines.append("")
    lines.append("Credibility is computed as the **sum of model weights** over models that contain the edge.")
    lines.append("")
    topN = min(args.topk_claims, len(claims_df))
    lines.append(f"Top {topN} claims:")
    lines.append("")
    lines.append("| Rank | Credibility | Support | Claim | Supported by | Example evidence |")
    lines.append("|---:|---:|---:|---|---|---|")

    for i, row in claims_df.head(topN).iterrows():
        claim = f"**{row['src']}** —{row['rel']}→ **{row['dst']}**"
        ex = truncate(row["example_1"], 140)
        lines.append(
            f"| {i+1} | {row['credibility']:.3f} | {int(row['support_count'])} | "
            f"{claim} | {row['models_supporting']} | {ex} |"
        )

    lines.append("")
    lines.append("## Per-model natural language summaries (right-adjoint prototypes)")
    lines.append("")
    for m in sorted(models, key=lambda x: x["score"], reverse=True):
        lcm = m["lcm"]
        focus, hub, hub_edges = summarize_lcm_hub(lcm, k=8)
        lines.append(f"### {focus}  (score={m['score']:.3f}, weight={m['weight']:.3f})")
        lines.append("")
        lines.append(f"**Central hub node:** `{hub}`")
        lines.append("")
        lines.append("**Key outgoing claims from hub:**")
        for e in hub_edges:
            key = canonical_edge(e)
            ex = ""
            if key in edge_examples and edge_examples[key]:
                ex = truncate(edge_examples[key][0], 180)
            lines.append(f"- {e['src']} —{e.get('rel','')}→ {e['dst']}" + (f"  \n  _Evidence:_ {ex}" if ex else ""))
        lines.append("")
        lines.append("**Interpretation (template):**")
        lines.append(f"- This model centers on **{focus}** and organizes a local causal neighborhood in which `{hub}` connects to downstream physical, ecological, and/or policy consequences.")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[credibility_report] wrote: {out_md}")
    print(f"[credibility_report] wrote: {out_csv}")


if __name__ == "__main__":
    main()
