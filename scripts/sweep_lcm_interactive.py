#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import pandas as pd
import networkx as nx

from scripts.local_causal_dag import load_graph_from_triples, make_local_figure, save_lcm_json
from scripts.lcm_score_text import score_lcm_text

import re
import networkx as nx

JUNK_PAT = re.compile(r"\b(def|class|import|return|print)\b|get_gpt_response|prompt\):|\bno other text\b|#")

ANCHOR_RELS = {"has_subj", "has_obj"}

def count_nonanchor_edges_lcm(lcm):
    return sum(1 for e in lcm.get("edges", []) if e.get("rel","") not in ANCHOR_RELS)

def focus_incident_nonanchor_edges_lcm(lcm, focus):
    return sum(
        1 for e in lcm.get("edges", [])
        if e.get("rel","") not in ANCHOR_RELS and (e["src"] == focus or e["dst"] == focus)
    )

def is_degenerate_lcm(lcm, focus):
    ce = count_nonanchor_edges_lcm(lcm)
    fi = focus_incident_nonanchor_edges_lcm(lcm, focus)
    # tune thresholds later; keep them loose
    return (ce < 2) or (fi < 1)

def is_codey_focus(focus: str) -> bool:
    return bool(JUNK_PAT.search(focus.lower()))

JUNK_PAT = re.compile(
    r"\b(def|class|import|return|print)\b|"
    r"get_gpt_response|prompt\):|"
    r"\bno other text\b|"
    r"#|\{|\}|\[|\]|::|;|=\s*"
)

def pick_focus_nodes(G, topk=40):
    degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    out = []
    for n, _ in degs:
        s = str(n).strip().lower()
        if JUNK_PAT.search(s):
            continue
        out.append(n)
        if len(out) >= topk:
            break
    return out
    
def compute_node_support(triples_path):
    c = Counter()
    import json
    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            s = r.get("subj") or r.get("source")
            o = r.get("obj")  or r.get("target")
            if s: c[s] += 1
            if o: c[o] += 1
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True)
    ap.add_argument("--outdir", default="figs/sweep")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--radii", type=str, default="1,2")
    ap.add_argument("--maxnodes", type=str, default="30,60")
    ap.add_argument("--lambda-edge", type=float, default=0.25)
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    triples_path = Path(args.triples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    G = load_graph_from_triples(triples_path)
    focuses = pick_focus_nodes(G, topk=args.topk)

    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    maxnodes = [int(x) for x in args.maxnodes.split(",") if x.strip()]

    rows = []
    for focus in focuses:
        if focus not in G:
            continue
        for r in radii:
            for m in maxnodes:
                # extract local neighborhood graph (but don’t draw/save PNG here)
                H = make_local_figure(
                    G, focus=focus, radius=r, max_nodes=m,
                    out_file=None, title_prefix="",
                )
                ANCHOR_RELS = {"has_subj", "has_obj"}

            
                # write LCM JSON
                slug = "".join(ch for ch in focus.lower() if ch.isalnum() or ch in (" ", "_")).strip().replace(" ", "_")
                lcm_path = outdir / f"{slug}_r{r}_m{m}.json"
                save_lcm_json(H, focus, r, triples_path, lcm_path)
                
                lcm = json.loads(lcm_path.read_text(encoding="utf-8"))

                # HARD GUARD: reject degenerate exported models
                if len(lcm.get("edges", [])) < 2:
                    lcm_path.unlink(missing_ok=True)
                    continue

                # focus must actually participate in exported causal edges
                focus_incident = sum(
                    1 for e in lcm["edges"]
                    if e["src"] == focus or e["dst"] == focus
                )
                if focus_incident < 1:
                    lcm_path.unlink(missing_ok=True)
                continue

                # score (text evidence)
                lcm = json.loads(lcm_path.read_text(encoding="utf-8"))
                s = score_lcm_text(
                    lcm, triples_path,
                    lambda_edge=args.lambda_edge,
                    local_only=args.local_only,
                )
                
                ce = count_nonanchor_edges_lcm(lcm)
                fi = focus_incident_nonanchor_edges_lcm(lcm, focus)
                degenerate = is_degenerate_lcm(lcm, focus)
                codey = is_codey_focus(focus)

                rows.append({
                    "focus": focus,
                    "radius": r,
                    "max_nodes": m,
                    "n_nodes": len(lcm["nodes"]),
                    "n_edges": len(lcm["edges"]),
                    "causal_edges": ce,
                    "focus_incident_edges": fi,
                    "degenerate": int(degenerate),
                    "codey_focus": int(codey),
                    "score": float(s),
                    "lcm_json": str(lcm_path),
                })

              

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_csv = outdir / "scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    main()
