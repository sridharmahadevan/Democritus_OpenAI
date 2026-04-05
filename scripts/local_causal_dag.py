#!/usr/bin/env python3
"""
local_causal_dag.py

Generate a static local causal graph figure from a Democritus
relational triples file for a chosen focus node.

- Works with multiple triple schemas:
    {subj, obj, rel, topic}   (older)
    {source, target, relation, topic} (newer)
- Domain-agnostic: any set of triples is fine.

Usage examples
--------------
# default file + focus
python -m scripts.local_causal_dag \
    --triples relational_triples.jsonl \
    --focus "Chicxulub asteroid impact"

# larger neighborhood
python -m scripts.local_causal_dag \
    --triples relational_triples.jsonl \
    --focus "Holocene monsoon variability in South Asia" \
    --radius 2 --max-nodes 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


#----------------------------------
# Output local causal model as JSON file
#----------------------------------

# --- add near imports ---
from dataclasses import dataclass
from typing import Any

# --- add helper ---
def save_lcm_json(
    H: nx.DiGraph,
    focus: str,
    radius: int,
    triples_path: Path,
    out_json: Path,
    meta: dict[str, Any] | None = None,
) -> None:
    # capture topic nodes via outgoing "has_subj/has_obj" edges, if present
    topic_nodes = set()
    for u, v, d in H.edges(data=True):
        if d.get("rel") in ("has_subj", "has_obj"):
            topic_nodes.add(u)

    payload = {
        "focus": focus,
        "radius": radius,
        "nodes": list(H.nodes()),
        "topics": sorted(topic_nodes),
        "edges": [
            {"src": u, "dst": v, "rel": (d.get("rel") or "")}
            for u, v, d in H.edges(data=True)
            # optionally exclude anchor edges from topic→node if you want a pure causal LCM:
            # if d.get("rel") not in ("has_subj","has_obj")
        ],
        "meta": {
            "triples_file": str(triples_path),
            **(meta or {}),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved LCM JSON to {out_json}")


def load_graph_from_triples(path: Path) -> nx.DiGraph:
    """
    Build a directed graph from a Democritus triples JSONL file.

    Tries to be robust to field names:
      - subj/obj/rel vs source/target/relation
      - optional 'topic' anchor.
    """
    G = nx.DiGraph()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            # Try multiple key spellings
            subj = rec.get("subj") or rec.get("source")
            obj  = rec.get("obj") or rec.get("target")
            rel  = rec.get("rel") or rec.get("relation") or ""
            topic = rec.get("topic") or rec.get("domain") or ""

            if not subj or not obj:
                continue

            # Add nodes
            for node in (subj, obj, topic):
                if node:
                    G.add_node(node)

            # subj -> obj causal edge
            G.add_edge(subj, obj, rel=rel)

            # Optional anchoring edges from topic to subj/obj
            if topic:
                G.add_edge(topic, subj, rel="has_subj")
                G.add_edge(topic, obj,  rel="has_obj")

    return G


def make_local_figure(
    G: nx.DiGraph,
    focus: str,
    radius: int = 1,
    max_nodes: int = 40,
    out_file: Path | None = None,
    title_prefix: str = "Democritus WhyGraph: Local Causal Neighborhood",
) -> nx.DiGraph:
    """
    Extract a local ego-graph around `focus` and draw it.

    Parameters
    ----------
    G : nx.DiGraph
        Global causal graph.
    focus : str
        Node label to center the neighborhood on.
    radius : int
        Ego radius in hop distance.
    max_nodes : int
        Hard cap on neighborhood size (keep highest-degree nodes).
    out_file : Path | None
        If provided, save PNG there; otherwise show interactively.
    """
    if focus not in G:
        raise ValueError(f"Focus node not in graph: {focus!r}")

    # Ego neighborhood
    H = nx.ego_graph(G, focus, radius=radius, undirected=False)
    
    # Keep only the weakly-connected component containing the focus node
    if H.number_of_nodes() > 0:
        und = H.to_undirected()
        comp = next(c for c in nx.connected_components(und) if focus in c)
        H = H.subgraph(comp).copy()

    if H.number_of_nodes() > max_nodes:
        degs = sorted(H.degree, key=lambda x: x[1], reverse=True)
        keep = {focus} | {n for n, _ in degs[: max_nodes - 1]}
        H = H.subgraph(keep).copy()

    # Layout
    pos = nx.spring_layout(H, seed=0, k=0.8)

    # Styling
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.axis("off")

    # Edges with red arrows
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="red",
                lw=1.5,
                shrinkA=5,
                shrinkB=5,
            ),
        )

    # Nodes: gold circles with red outlines, larger for focus node
    node_colors = ["gold" for _ in H.nodes()]
    node_sizes = []
    for n in H.nodes():
        node_sizes.append(700 if n == focus else 400)

    nx.draw_networkx_nodes(
        H,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="red",
        linewidths=1.5,
    )

    # Labels in blue
    labels = {n: n for n in H.nodes()}
    nx.draw_networkx_labels(
        H,
        pos,
        labels,
        font_size=9,
        font_color="blue",
    )

    plt.title(
        f"{title_prefix}\nFocus node: {focus}",
        color="blue",
        fontsize=11,
    )

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        print(f"Saved figure to {out_file}")
    else:
        plt.show()
    return H


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local causal DAG viewer for Democritus triples.")
    p.add_argument(
        "--triples",
        type=str,
        default="relational_triples.jsonl",
        help="Path to relational triples JSONL file (default: relational_triples.jsonl)",
    )
    p.add_argument(
        "--focus",
        type=str,
        required=True,
        help='Focus node label, e.g. "Chicxulub asteroid impact"',
    )
    p.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Ego radius (hops) around focus node (default: 1)",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=40,
        help="Maximum number of nodes in local neighborhood (default: 40)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="figs/local_causal_dag.png",
        help="Output PNG (default: figs/local_causal_dag.png). If empty, show interactively.",
    )
    p.add_argument(
        "--title-prefix",
        type=str,
        default="Democritus WhyGraph: Local Causal Neighborhood",
        help="Title prefix for the figure.",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="figs/local_causal_dag.json",
        help="Output JSON for the extracted local causal model (default: figs/local_causal_dag.json)",
    )
    return p.parse_args()

#--------------------------------------
# write out JSON file for local causal graph
#--------------------------------------

def save_lcm_json(
    H: nx.DiGraph,
    focus: str,
    radius: int,
    triples_path: Path,
    out_json: Path,
    *,
    drop_anchor_edges: bool = True,
) -> None:
    edges = []
    for u, v, d in H.edges(data=True):
        rel = (d.get("rel") or "")
        if drop_anchor_edges and rel in ("has_subj", "has_obj"):
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
    print(f"Saved LCM JSON to {out_json}")

def main() -> None:
    args = parse_args()
    triples_path = Path(args.triples)
    if not triples_path.exists():
        raise FileNotFoundError(f"Triples file not found: {triples_path}")

    G = load_graph_from_triples(triples_path)
    print(f"[local_causal_dag] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    out_file = Path(args.out) if args.out else None

    H = make_local_figure(
        G,
        focus=args.focus,
        radius=args.radius,
        max_nodes=args.max_nodes,
        out_file=out_file,
        title_prefix=args.title_prefix,
    )

    save_lcm_json(
        H,
        focus=args.focus,
        radius=args.radius,
        triples_path=triples_path,
        out_json=Path(args.out_json),
    )


if __name__ == "__main__":
    main()
