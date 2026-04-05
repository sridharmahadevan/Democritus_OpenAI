#!/usr/bin/env python3
"""
sweep_lcms.py

Generate (and ONLY generate) local causal models (LCMs) as JSON files
from a Democritus relational triples JSONL file.

- No plotting
- No scoring
- No filtering
- No prompts

It sweeps over:
  focus nodes (top-k by degree by default)
  radii (comma-separated list)
  max_nodes (comma-separated list)

Outputs:
  outdir/*.json  (one per LCM)
  outdir/index.csv  (metadata index)

Usage:
  python -m scripts.sweep_lcms \
    --triples relational_triples.jsonl \
    --outdir figs/sweep_wapo \
    --topk 200 \
    --radii 1,2,3 \
    --maxnodes 20,40,60

Notes:
- This script intentionally writes *everything* (including junk).
- You can delete masses of junk later by filename, as you proposed.
"""

import argparse
import csv
import json
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------
# Global graph construction (robust to different triple schemas)
# ---------------------------------------------------------------------
def load_graph_from_triples(path: Path) -> nx.DiGraph:
    """
    Build a directed graph from a Democritus triples JSONL file.

    Supports:
      {subj, obj, rel, topic/domain}
      {source, target, relation, topic/domain}

    Adds:
      subj -> obj  edges with rel
      topic -> subj / topic -> obj anchor edges with rel=has_subj/has_obj (optional)
    """
    G = nx.DiGraph()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            subj = rec.get("subj") or rec.get("source")
            obj  = rec.get("obj")  or rec.get("target")
            rel  = rec.get("rel")  or rec.get("relation") or ""
            topic = rec.get("topic") or rec.get("domain") or ""

            if not subj or not obj:
                continue

            # nodes
            G.add_node(subj)
            G.add_node(obj)
            if topic:
                G.add_node(topic)

            # causal-ish edge
            G.add_edge(subj, obj, rel=rel)

            # optional anchors
            if topic:
                G.add_edge(topic, subj, rel="has_subj")
                G.add_edge(topic, obj,  rel="has_obj")

    return G


# ---------------------------------------------------------------------
# Local neighborhood extraction (NO PLOTTING)
# ---------------------------------------------------------------------
def extract_local_subgraph(
    G: nx.DiGraph,
    focus: str,
    radius: int,
    max_nodes: int,
) -> nx.DiGraph:
    """
    Extract a local directed ego neighborhood around `focus`.

    - Uses directed ego graph (as in your visualizer).
    - If too many nodes, keeps the top-degree nodes (always includes focus).
    """
    H = nx.ego_graph(G, focus, radius=radius, undirected=False)

    if H.number_of_nodes() > max_nodes:
        degs = sorted(H.degree, key=lambda x: x[1], reverse=True)
        keep = {focus} | {n for n, _ in degs[: max_nodes - 1]}
        H = H.subgraph(keep).copy()

    return H


# ---------------------------------------------------------------------
# LCM JSON writer
# ---------------------------------------------------------------------
ANCHOR_RELS = {"has_subj", "has_obj"}

def save_lcm_json(
    H: nx.DiGraph,
    focus: str,
    radius: int,
    triples_path: Path,
    out_json: Path,
    *,
    drop_anchor_edges: bool = True,
) -> dict:
    """
    Save local graph H as an LCM JSON file.

    Returns the payload dict (useful for indexing).
    """
    edges = []
    for u, v, d in H.edges(data=True):
        rel = (d.get("rel") or "")
        if drop_anchor_edges and rel in ANCHOR_RELS:
            continue
        edges.append({"src": u, "dst": v, "rel": rel})

    payload = {
        "focus": focus,
        "radius": radius,
        "nodes": list(H.nodes()),
        "edges": edges,
        "meta": {"triples_file": str(triples_path)},
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------
# Focus selection
# ---------------------------------------------------------------------
def pick_focus_nodes_by_degree(G: nx.DiGraph, topk: int) -> list[str]:
    """
    Pick top-k nodes by total degree. (No filtering by design.)
    """
    degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [n for n, _ in degs[:topk]]


def slugify(s: str, maxlen: int = 80) -> str:
    """
    Filename-safe slug.
    """
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in (" ", "_", "-"):
            out.append(ch)
    slug = "".join(out).strip().replace(" ", "_")
    if len(slug) > maxlen:
        slug = slug[:maxlen]
    return slug or "focus"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True, help="relational_triples.jsonl")
    ap.add_argument("--outdir", default="figs/sweep", help="where to write JSONs")
    ap.add_argument("--topk", type=int, default=40, help="top-k focus nodes by degree")
    ap.add_argument("--radii", type=str, default="1,2", help="comma-separated radii, e.g. 1,2,3")
    ap.add_argument("--maxnodes", type=str, default="30,60", help="comma-separated max_nodes, e.g. 20,40,60")
    ap.add_argument("--keep-anchors", action="store_true",
                    help="keep topic anchor edges (has_subj/has_obj) in exported JSON")
    ap.add_argument("--index", default="index.csv", help="index filename (written inside outdir)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    triples_path = Path(args.triples)
    if not triples_path.exists():
        raise FileNotFoundError(triples_path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    maxnodes = [int(x) for x in args.maxnodes.split(",") if x.strip()]

    G = load_graph_from_triples(triples_path)
    focuses = pick_focus_nodes_by_degree(G, topk=args.topk)

    index_path = outdir / args.index
    with index_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "focus", "radius", "max_nodes", "n_nodes", "n_edges"],
        )
        w.writeheader()

        n_written = 0
        for focus in focuses:
            if focus not in G:
                continue

            focus_slug = slugify(focus)

            for r in radii:
                for m in maxnodes:
                    H = extract_local_subgraph(G, focus=focus, radius=r, max_nodes=m)

                    # Write JSON
                    out_json = outdir / f"{focus_slug}_r{r}_m{m}.json"
                    payload = save_lcm_json(
                        H,
                        focus=focus,
                        radius=r,
                        triples_path=triples_path,
                        out_json=out_json,
                        drop_anchor_edges=(not args.keep_anchors),
                    )

                    w.writerow({
                        "file": out_json.name,
                        "focus": focus,
                        "radius": r,
                        "max_nodes": m,
                        "n_nodes": len(payload["nodes"]),
                        "n_edges": len(payload["edges"]),
                    })
                    n_written += 1

    print(f"[sweep_lcms] Global graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[sweep_lcms] Wrote {n_written} LCM JSON files to: {outdir}")
    print(f"[sweep_lcms] Index: {index_path}")


if __name__ == "__main__":
    main()
