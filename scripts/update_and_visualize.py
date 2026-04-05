
#!/usr/bin/env python3
"""
update_and_visualize.py

Convenience wrapper:
  1) Run topos_update to merge new topos_slices into the global state.
  2) Run UMAP visualizations on the global relational state.

Usage:
  python -m scripts.update_and_visualize
  # or
  python scripts/update_and_visualize.py
"""

import os
import time

from scripts import topos_update
from scripts.mumble_relational_umap import generate_umap_bundle


GLOBAL_STATE_PATH = os.path.join("models", "relational_global_state.pkl")
UMAP_OUT_DIR      = os.path.join("models", "umap_vis")
BASE_NAME         = "global"  # used in filenames, e.g. global_domain_2d.png


def main():
    t0 = time.time()

    # ----------------------------------------------------
    # Step 1: update global relational_state from slices
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP A: Updating global topos state from topos_slices")
    print("=" * 80 + "\n")

    topos_update.main()  # uses topos_update.py's existing logic

    if not os.path.exists(GLOBAL_STATE_PATH):
        print(f"[update_and_visualize] WARNING: {GLOBAL_STATE_PATH} not found "
              f"after topos_update; skipping UMAP.")
        return

    # ----------------------------------------------------
    # Step 2: generate UMAP visualizations on global state
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STEP B: Generating UMAP visualizations for {GLOBAL_STATE_PATH}")
    print("=" * 80 + "\n")

    os.makedirs(UMAP_OUT_DIR, exist_ok=True)

    generate_umap_bundle(
        relational_state_path=GLOBAL_STATE_PATH,
        out_dir=UMAP_OUT_DIR,
        base_name=BASE_NAME,
        embedding_key="emb",   # matches your relational_state
    )

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[update_and_visualize] COMPLETE in {dt:.1f} seconds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
