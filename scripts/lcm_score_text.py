#!/usr/bin/env python3
"""
lcm_score_text.py

Score an LCM using only the triples evidence:
- reward edges that appear in the triples for this topic neighborhood
- reward mediation patterns implied by 'by/leading_to/which' (already in triples)
- penalize complexity
This is a pragmatic evaluator when no numeric dataset is available.
"""

import argparse, json, math
from pathlib import Path
from collections import Counter

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
    
def apply_map(name: str, name_map: dict[str, str]) -> str:
    return name_map.get(name, name)

def edges_to_parents(lcm: dict, name_map: dict[str, str], *, drop_anchor_edges: bool = True) -> dict[str, list[str]]:
    nodes = [apply_map(n, name_map) for n in lcm["nodes"]]
    parents: dict[str, list[str]] = {n: [] for n in nodes}

    for e in lcm["edges"]:
        rel = e.get("rel", "")
        if drop_anchor_edges and rel in ("has_subj", "has_obj"):
            continue
        src = apply_map(e["src"], name_map)
        dst = apply_map(e["dst"], name_map)
        if src in parents and dst in parents and src != dst:
            parents[dst].append(src)

    for k in parents:
        parents[k] = sorted(set(parents[k]))
    return parents

def score_lcm_text(
    lcm: dict,
    triples_jsonl: Path,
    *,
    drop_anchor_edges: bool = True,
    lambda_edge: float = 0.25,
    w_edge: float = 1.0,
    local_only: bool = False,
    ) -> float:
    """
    Very simple likelihood surrogate:
    - Treat each triple subj->obj as supporting that directed edge.
    - Score = sum_{edges in lcm} log(1 + count_support(edge)) - lambda*|E|
    """
    support = Counter()
    with triples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            s = r.get("subj") or r.get("source")
            o = r.get("obj") or r.get("target")
            lcm_nodes = set(lcm["nodes"])
            if local_only:
                if (s not in lcm_nodes) or (o not in lcm_nodes):
                    continue
            if s and o:
                support[(s, o)] += 1

    edges = []
    for e in lcm["edges"]:
        rel = e.get("rel", "")
        if drop_anchor_edges and rel in ("has_subj", "has_obj"):
            continue
        edges.append((e["src"], e["dst"]))

    score = 0.0
    for (u, v) in edges:
        c = support[(u, v)]
        score += w_edge * math.log(1.0 + c)

    score -= lambda_edge * len(edges)
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcm", required=True, help="LCM JSON produced by local_causal_dag.py")
    ap.add_argument("--triples", required=True, help="relational_triples.jsonl used to build the graph")
    ap.add_argument("--lambda-edge", type=float, default=0.25, help="Complexity penalty per edge")
    ap.add_argument("--local-only", action="store_true",
                    help="Only count triple evidence where both endpoints are in the LCM nodes")
    args = ap.parse_args()

    lcm = load_json(Path(args.lcm))
    s = score_lcm_text(
        lcm,
        Path(args.triples),
        lambda_edge=args.lambda_edge,
        local_only=args.local_only,
    )
    print(f"Text-evidence score (higher is better): {s:.3f}")

if __name__ == "__main__":
    main()
