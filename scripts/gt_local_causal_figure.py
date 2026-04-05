#!/usr/bin/env python3
"""
gt_local_causal_figure.py

Generate a static, publication-style local causal graph figure
from relational_triples_*.jsonl for a chosen focus node.

- White background
- Blue text
- Yellow nodes with red outlines
- Red arrows
"""

import json
import networkx as nx
import matplotlib.pyplot as plt

TRIPLES_PATH = "relational_triples.jsonl"  # adjust as needed
FOCUS_NODE   = "Indus River discharge and river droughts" # or any node text
OUT_FIG      = "figs/democritus-indus-valley.png"     # output PNG


def load_graph_from_triples(path):
    G = nx.DiGraph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            subj = rec.get("subj")
            obj  = rec.get("obj")
            rel  = rec.get("rel", "")
            topic = rec.get("topic", "")

            if not subj or not obj:
                continue

            # Add nodes
            for node in (subj, obj, topic):
                if node:
                    G.add_node(node)

            # subj -> obj causal edge
            G.add_edge(subj, obj, rel=rel)

            # optional anchor edges
            if topic:
                G.add_edge(topic, subj, rel="has_subj")
                G.add_edge(topic, obj,  rel="has_obj")
    return G


def make_local_figure(G, focus, radius=1, max_nodes=40, out_file=None):
    if focus not in G:
        raise ValueError(f"Focus node not in graph: {focus}")

    # get ego graph
    H = nx.ego_graph(G, focus, radius=radius, undirected=False)
    if H.number_of_nodes() > max_nodes:
        degs = sorted(H.degree, key=lambda x: x[1], reverse=True)
        keep = {focus} | {n for n, _ in degs[: max_nodes - 1]}
        H = H.subgraph(keep).copy()

    # layout
    pos = nx.spring_layout(H, seed=0, k=0.8)

    # styling
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.axis("off")

    # draw edges with red arrows
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

    # draw nodes as yellow circles with red outlines
    node_colors = ["gold" if n != focus else "gold" for n in H.nodes()]
    node_sizes  = [400 if n != focus else 700 for n in H.nodes()]
    nx.draw_networkx_nodes(
        H,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="red",
        linewidths=1.5,
    )

    # draw labels in blue
    labels = {n: n for n in H.nodes()}
    nx.draw_networkx_labels(
        H,
        pos,
        labels,
        font_size=10,
        font_color="blue",
    )

    plt.title(
        f"Democritus: Causality from Large Language Models\n"
        f"Local causal neighborhood of: {focus}",
        color="blue",
        fontsize=12,
    )

    if out_file:
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        print(f"Saved figure to {out_file}")
    else:
        plt.show()


def main():
    G = load_graph_from_triples(TRIPLES_PATH)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    make_local_figure(G, FOCUS_NODE, radius=1, max_nodes=40, out_file=OUT_FIG)


if __name__ == "__main__":
    main()
