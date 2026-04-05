#!/usr/bin/env python3
"""
topos_update.py

Consume topos_slices/*.pkl and merge into models/relational_global_state.pkl.
For v0: if global doesn't exist, take the first slice as-is.
Later we plug in GC + proper merging.
"""

import os
import glob
import json
import pickle

TOPOS_DIR = "topos_slices"
MODELS_DIR = "models"
GLOBAL_PATH = os.path.join(MODELS_DIR, "relational_global_state.pkl")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def find_unprocessed_slices():
    metas = glob.glob(os.path.join(TOPOS_DIR, "*.meta.json"))
    to_process = []
    for meta_path in metas:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("processed", False):
            continue
        base = meta_path[:-10]  # strip ".meta.json"
        pkl_path = base + ".pkl"
        if os.path.exists(pkl_path):
            to_process.append((pkl_path, meta_path, meta))
    # sort by timestamp so we process in chronological order
    to_process.sort(key=lambda x: x[2]["timestamp"])
    return to_process


def mark_processed(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    meta["processed"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    slices = find_unprocessed_slices()
    if not slices:
        print("[ToposUpdate] No new slices to process.")
        return

    print(f"[ToposUpdate] Found {len(slices)} unprocessed slices.")
    if not os.path.exists(GLOBAL_PATH):
        # v0: initialize from first slice only
        first_pkl, first_meta_path, first_meta = slices[0]
        print(f"[ToposUpdate] Initializing global state from {first_pkl}")
        state = load_pickle(first_pkl)
        save_pickle(state, GLOBAL_PATH)
        mark_processed(first_meta_path)
        print(f"[ToposUpdate] Global state written to {GLOBAL_PATH}")
        # We leave other slices untouched for now
    else:
        print(f"[ToposUpdate] Global state already exists at {GLOBAL_PATH}")
        print("[ToposUpdate] (Merging multiple slices will be wired in next.)")


if __name__ == "__main__":
    main()
