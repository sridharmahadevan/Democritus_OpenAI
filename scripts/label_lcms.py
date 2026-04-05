#!/usr/bin/env python3
"""
label_lcms.py

Human-in-the-loop filter for Democritus LCMs.
- Iterates through LCM JSON files
- (Optionally) renders each LCM as a quick matplotlib graph
- Prompts user: [y] keep, [n] reject, [s] skip, [q] quit
- Moves files into keep/ or reject/ subfolders
- Writes labels.csv with decisions

Usage:
  python scripts/label_lcms.py --indir figs/sweep_wapo --show

Notes:
- Requires: networkx, matplotlib (already in your env)
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt


ANCHOR_RELS = {"has_subj", "has_obj"}


def load_lcm(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lcm_to_graph(lcm: dict, drop_anchor_edges: bool = True) -> nx.DiGraph:
    G = nx.DiGraph()
    focus = lcm.get("focus", "")
    for n in lcm.get("nodes", []):
        G.add_node(n)
    for e in lcm.get("edges", []):
        rel = e.get("rel", "")
        if drop_anchor_edges and rel in ANCHOR_RELS:
            continue
        u, v = e["src"], e["dst"]
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)
        G.add_edge(u, v, rel=rel)
    # ensure focus exists if present
    if focus and focus not in G:
        G.add_node(focus)
    return G


def show_lcm(lcm: dict, title_prefix: str = "LCM") -> None:
    """Quick visualization in a popup window."""
    focus = lcm.get("focus", "")
    G = lcm_to_graph(lcm, drop_anchor_edges=True)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.axis("off")

    # Small guard: if graph is empty, just show text
    if G.number_of_nodes() == 0:
        plt.title(f"{title_prefix}: (empty)")
        plt.show()
        return

    pos = nx.spring_layout(G, seed=0, k=0.9)

    # edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", width=1.5)

    # nodes
    sizes = [900 if n == focus else 450 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, linewidths=1.5)

    # labels (truncate long labels for readability)
    def short(s: str, n: int = 42) -> str:
        s = " ".join(str(s).split())
        return s if len(s) <= n else s[: n - 3] + "..."

    labels = {n: short(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(f"{title_prefix} | focus: {short(focus, 60)}\n"
              f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}",
              fontsize=11)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing LCM JSON files (e.g., figs/sweep_wapo)")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern for LCM json files (default: *.json)")
    ap.add_argument("--show", action="store_true", help="Pop up a matplotlib visualization for each LCM")
    ap.add_argument("--keepdir", default="keep", help="Subfolder name for kept models")
    ap.add_argument("--rejectdir", default="reject", help="Subfolder name for rejected models")
    ap.add_argument("--labels", default="labels.csv", help="Output CSV of labels (written inside indir)")
    ap.add_argument("--resume", action="store_true", help="Skip files already listed in labels.csv")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.exists():
        raise FileNotFoundError(indir)

    keepdir = indir / args.keepdir
    rejectdir = indir / args.rejectdir
    keepdir.mkdir(parents=True, exist_ok=True)
    rejectdir.mkdir(parents=True, exist_ok=True)

    labels_path = indir / args.labels

    done = set()
    if args.resume and labels_path.exists():
        with labels_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["file"])

    json_files = sorted(indir.glob(args.pattern))
    # ignore keep/reject subfolders
    json_files = [p for p in json_files if p.parent == indir]
    if not json_files:
        print(f"No files matched {indir / args.pattern}")
        return

    write_header = not labels_path.exists()
    with labels_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "focus", "nodes", "edges", "decision"])
        if write_header:
            w.writeheader()

        for p in json_files:
            if args.resume and p.name in done:
                continue

            lcm = load_lcm(p)
            focus = lcm.get("focus", "")
            G = lcm_to_graph(lcm, drop_anchor_edges=True)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            print("\n" + "-" * 80)
            print(f"FILE:   {p.name}")
            print(f"FOCUS:  {focus}")
            print(f"SIZE:   nodes={n_nodes} edges={n_edges}")
            # show a tiny edge preview
            preview = list(G.edges())[:8]
            if preview:
                print(f"EDGES (preview): {preview}")
            else:
                print("EDGES (preview): (none)")

            if args.show:
                show_lcm(lcm, title_prefix="Democritus LCM")

            while True:
                ans = input("[y] keep  [n] reject  [s] skip  [q] quit > ").strip().lower()
                if ans in ("y", "n", "s", "q"):
                    break

            if ans == "q":
                print("Quitting.")
                return
            if ans == "s":
                w.writerow({"file": p.name, "focus": focus, "nodes": n_nodes, "edges": n_edges, "decision": "skip"})
                f.flush()
                continue

            if ans == "y":
                dest = keepdir / p.name
                decision = "keep"
            else:
                dest = rejectdir / p.name
                decision = "reject"

            shutil.move(str(p), str(dest))
            w.writerow({"file": p.name, "focus": focus, "nodes": n_nodes, "edges": n_edges, "decision": decision})
            f.flush()
            print(f"→ {decision.upper()}: moved to {dest.relative_to(indir)}")

    print(f"\nDone. Labels written to {labels_path}")


if __name__ == "__main__":
    main()
