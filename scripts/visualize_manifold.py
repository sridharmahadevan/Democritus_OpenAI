# scripts/visualize_manifold.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

def visualize_from_state(
    state_path: str,
    out_dir: str = "viz",
    title_prefix: str = "Relational manifold",
):
    """
    Load a saved relational/manifold state and produce 2D and 3D UMAP plots.

    Parameters
    ----------
    state_path : str
        Path to the .pkl file (either relational_state.pkl or manifold_state.pkl).
    out_dir : str
        Directory where plots will be saved.
    title_prefix : str
        Text prefix to use in plot titles.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    # Prefer GT-refined embeddings if present, otherwise use base embeddings
    emb = None
    if "V_ref" in state:
        emb = np.array(state["V_ref"])
    elif "emb" in state:
        emb = np.array(state["emb"])

    umap_2 = state.get("umap_2d", None)
    umap_3 = state.get("umap_3d", None)

    # ---------- 2D plot ----------
    if umap_2 is not None and umap_2.ndim == 2 and umap_2.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_2[:, 0], umap_2[:, 1], s=10)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(f"{title_prefix} (2D)")
        out_path_2d = os.path.join(out_dir, "relational_manifold_2d.png")
        plt.tight_layout()
        plt.savefig(out_path_2d, dpi=150)
        plt.close()
        print(f"[viz] Saved 2D manifold plot to {out_path_2d}")
    else:
        print("[viz] No valid 2D embedding found; skipping 2D plot.")

    # ---------- 3D plot ----------
    if umap_3 is not None and umap_3.ndim == 2 and umap_3.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(umap_3[:, 0], umap_3[:, 1], umap_3[:, 2], s=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        ax.set_title(f"{title_prefix} (3D)")
        out_path_3d = os.path.join(out_dir, "relational_manifold_3d.png")
        plt.tight_layout()
        plt.savefig(out_path_3d, dpi=150)
        plt.close()
        print(f"[viz] Saved 3D manifold plot to {out_path_3d}")
    else:
        print("[viz] No valid 3D embedding found; skipping 3D plot.")
