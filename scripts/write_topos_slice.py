#!/usr/bin/env python3
import os
import json
import pickle
from datetime import datetime
from typing import List, Tuple


def write_topos_slice(
    rel_state_path: str,
    domain_name: str,
    topic_roots: List[str],
    out_dir: str = "topos_slices",
) -> Tuple[str, str]:
    """
    Take a relational_state.pkl file and write a timestamped slice into
    topos_slices/, along with a small meta JSON.

    Returns: (slice_pkl_path, meta_json_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Timestamp for uniqueness and ordering
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    base = f"{ts}_{domain_name}"

    # Load the relational_state
    with open(rel_state_path, "rb") as f:
        state = pickle.load(f)

    # Save under topos_slices/
    slice_pkl = os.path.join(out_dir, base + ".pkl")
    with open(slice_pkl, "wb") as f:
        pickle.dump(state, f)

    # Minimal metadata
    meta = {
        "run_id": base,
        "domain": domain_name,
        "topic_roots": topic_roots,
        "timestamp": ts,
        "processed": False,
    }
    meta_json = os.path.join(out_dir, base + ".meta.json")
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Slice] Wrote topos slice {slice_pkl}")
    print(f"[Slice] Wrote metadata {meta_json}")
    return slice_pkl, meta_json


if __name__ == "__main__":
    # For ad-hoc testing:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rel-state", required=True, help="Path to relational_state.pkl")
    ap.add_argument("--domain", required=True, help="Domain name, e.g. 'economics'")
    ap.add_argument("--topic", action="append", default=[],
                    help="Root topic (may be repeated)")
    ap.add_argument("--out", default="topos_slices")
    args = ap.parse_args()

    write_topos_slice(args.rel_state, args.domain, args.topic, args.out)
