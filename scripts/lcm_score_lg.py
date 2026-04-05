#!/usr/bin/env python3
"""
lcm_score_lg.py

Score an exported LCM (JSON) as a linear-Gaussian SEM using BIC.
Requires a CSV with columns matching node names (after normalization).
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

def load_lcm(path: Path) -> dict:
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

def bic_linear_gaussian(df: pd.DataFrame, parents: dict[str, list[str]]) -> float:
    n = len(df)
    total = 0.0
    for y, xs in parents.items():
        if y not in df.columns:
            raise KeyError(f"Missing column for node: {y}")
        Y = df[y].to_numpy()
        if xs:
            missing = [x for x in xs if x not in df.columns]
            if missing:
                raise KeyError(f"Missing parent columns for {y}: {missing}")
            X = sm.add_constant(df[xs].to_numpy(), has_constant="add")
        else:
            X = np.ones((n, 1))
        fit = sm.OLS(Y, X).fit()
        total += float(fit.bic)
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcm", required=True, help="Path to LCM JSON")
    ap.add_argument("--csv", required=True, help="CSV with data columns matching node names")
    ap.add_argument("--map-json", default="", help="Optional JSON mapping from LCM node labels to CSV column names")
    args = ap.parse_args()

    lcm = load_lcm(Path(args.lcm))
    df = pd.read_csv(args.csv)
    name_map = json.loads(Path(args.map_json).read_text()) if args.map_json else {}
    parents = edges_to_parents(lcm, name_map)
    score = bic_linear_gaussian(df, parents)
    print(f"BIC (lower is better): {score:.3f}")

if __name__ == "__main__":
    main()
