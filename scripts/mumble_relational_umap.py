# mumble_relational_umap.py

import argparse
import os
import pickle
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch
import plotly.express as px
import pandas as pd

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    Axes3D = None


# ---------- core helpers ----------

def load_relational_state(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_embeddings(state, embedding_key=None):
    """
    Get an N x d embedding matrix from relational_state.

    - If embedding_key is given, use that.
    - Otherwise, prefer 'emb', then some common fallbacks.
    """
    if embedding_key is not None:
        if isinstance(state, dict):
            return to_numpy(state[embedding_key])
        else:
            return to_numpy(getattr(state, embedding_key))

    # Special-case for this repo: 'emb'
    if isinstance(state, dict) and "emb" in state:
        print("[info] Using embeddings from dict key 'emb'")
        return to_numpy(state["emb"])

    candidates = ["unified_manifold", "embeddings", "node_embeddings",
                  "var_embeddings", "latent", "Z"]

    if isinstance(state, dict):
        keys = list(state.keys())
        for k in candidates:
            if k in state:
                print(f"[info] Using embeddings from dict key '{k}'")
                return to_numpy(state[k])
        raise RuntimeError(
            "Could not find an embedding matrix in relational_state dict.\n"
            f"Available keys: {keys}\n"
            "Try re-running with --embedding-key <keyname>."
        )
    else:
        attrs = dir(state)
        for k in candidates:
            if hasattr(state, k):
                print(f"[info] Using embeddings from attribute '{k}'")
                return to_numpy(getattr(state, k))
        raise RuntimeError(
            "Could not find an embedding matrix in relational_state object.\n"
            f"Available attributes (sample): "
            f"{[a for a in attrs if not a.startswith('_')][:30]}\n"
            "Try re-running with --embedding-key <attrname>."
        )


def _map_ids_to_names(ids, mapping_dict, mapping_name):
    """
    Map integer IDs (dom_ids / rel_ids) to readable names via DOM2ID / REL2ID.

    Handles both id->name and name->id mappings.
    """
    ids = to_numpy(ids)
    ids = np.asarray(ids).astype(int)
    if ids.size == 0:
        return []

    sample_id = int(ids[0])

    # Case 1: keys are IDs
    if sample_id in mapping_dict:
        id2name = mapping_dict
    else:
        # Case 2: values are IDs -> invert
        inv = {v: k for k, v in mapping_dict.items()}
        if sample_id in inv:
            id2name = inv
        else:
            print(f"[warn] Could not interpret {mapping_name} mapping; "
                  "falling back to raw IDs as labels.")
            return [f"{mapping_name}:{int(i)}" for i in ids]

    labels = [id2name.get(int(i), f"{mapping_name}:{int(i)}") for i in ids]
    return labels


def extract_labels(state, label_mode="domain", num_points=None):
    """
    Extract labels for coloring the UMAP.

    label_mode:
      - 'domain'   -> derive per-variable domains from dom_ids + edges
      - 'relation' -> derive per-variable relation labels from rel_ids + edges
      - 'none'     -> no labels (points all same color)
    """
    if num_points is None:
        raise ValueError("num_points must be provided to extract_labels.")

    if label_mode == "none":
        print("[info] Coloring disabled (color-by=none).")
        return None

    # Convenience: convert tensors to numpy
    def _to_np(x):
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return x
        except Exception:
            pass
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        import numpy as _np
        return _np.asarray(x)

    # ------------------------------------------------------------------
    # DOMAIN mode
    # ------------------------------------------------------------------
    if label_mode == "domain":
        if "dom_ids" not in state or "DOM2ID" not in state or "edges" not in state:
            print("[info] No dom_ids/DOM2ID/edges in state; using uniform labels.")
            return ["var"] * num_points

        dom_ids = _to_np(state["dom_ids"])
        edges = state["edges"]
        try:
            edges = edges.tolist()
        except AttributeError:
            pass

        DOM2ID = state["DOM2ID"]

        # Build id -> name map (DOM2ID is usually name->id)
        sample_id = int(dom_ids[0]) if len(dom_ids) > 0 else 0
        if sample_id in DOM2ID:
            id2name = DOM2ID
        else:
            inv = {v: k for k, v in DOM2ID.items()}
            id2name = inv

        N = num_points
        E = len(dom_ids)
        if len(edges) != E:
            print(f"[warn] dom_ids length {E} != edges length {len(edges)}; "
                  "using uniform labels.")
            return ["var"] * N

        # Collect domains incident to each node
        node_dom_ids = [set() for _ in range(N)]
        for e_idx, (i, j) in enumerate(edges):
            if not (0 <= i < N and 0 <= j < N):
                continue
            d_id = int(dom_ids[e_idx])
            node_dom_ids[i].add(d_id)
            node_dom_ids[j].add(d_id)

        labels = []
        for i in range(N):
            if node_dom_ids[i]:
                first_dom = sorted(node_dom_ids[i])[0]
                labels.append(id2name.get(first_dom, "var"))
            else:
                labels.append("var")

        print("[info] Derived per-variable domain labels from per-edge dom_ids.")
        return labels

    # ------------------------------------------------------------------
    # RELATION mode
    # ------------------------------------------------------------------
    if label_mode == "relation":
        if "rel_ids" not in state or "REL2ID" not in state or "edges" not in state:
            print("[info] No rel_ids/REL2ID/edges in state; using uniform labels.")
            return ["var"] * num_points

        rel_ids = _to_np(state["rel_ids"])
        edges = state["edges"]
        try:
            edges = edges.tolist()
        except AttributeError:
            pass

        REL2ID = state["REL2ID"]

        sample_id = int(rel_ids[0]) if len(rel_ids) > 0 else 0
        if sample_id in REL2ID:
            id2name = REL2ID
        else:
            inv = {v: k for k, v in REL2ID.items()}
            id2name = inv

        N = num_points
        E = len(rel_ids)
        if len(edges) != E:
            print(f"[warn] rel_ids length {E} != edges length {len(edges)}; "
                  "using uniform labels.")
            return ["var"] * N

        node_rel_ids = [set() for _ in range(N)]
        for e_idx, (i, j) in enumerate(edges):
            if not (0 <= i < N and 0 <= j < N):
                continue
            r_id = int(rel_ids[e_idx])
            node_rel_ids[i].add(r_id)
            node_rel_ids[j].add(r_id)

        labels = []
        for i in range(N):
            if node_rel_ids[i]:
                first_rel = sorted(node_rel_ids[i])[0]
                labels.append(id2name.get(first_rel, "var"))
            else:
                labels.append("var")

        print("[info] Derived per-variable relation labels from per-edge rel_ids.")
        return labels

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    print(f"[warn] Unknown label_mode '{label_mode}', using uniform labels.")
    return ["var"] * num_points

    # Fallback ------------------------------------------------------------
    print(f"[warn] Unknown label_mode '{label_mode}', using uniform labels.")
    return ["var"] * num_points

def run_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


# ---------- high-level public API ----------

def make_umap_2d_png(
    relational_state_path: str,
    out_png: str,
    embedding_key: str = "emb",
    color_by: str = "domain",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Generate a 2D UMAP PNG (no interactive GUI)."""
    state = load_relational_state(relational_state_path)
    emb = extract_embeddings(state, embedding_key=embedding_key)
    labels = extract_labels(state, label_mode=color_by, num_points=emb.shape[0])
    coords = run_umap(emb, n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)

    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is None:
        sc = ax.scatter(x, y, s=5, alpha=0.8)
    else:
        unique = sorted(set(labels))
        label_to_int = {lab: i for i, lab in enumerate(unique)}
        colors = [label_to_int[lab] for lab in labels]
        sc = ax.scatter(x, y, c=colors, s=5, alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax, ticks=range(len(unique)))
        cbar.ax.set_yticklabels(unique)

    ax.set_title("Relational manifold (2D)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Saved 2D UMAP PNG to {out_png}")


def make_umap_3d_html(
    relational_state_path: str,
    out_html: str,
    embedding_key: str = "emb",
    color_by: str = "domain",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Generate a 3D UMAP interactive HTML (Plotly)."""
    state = load_relational_state(relational_state_path)
    emb = extract_embeddings(state, embedding_key=embedding_key)
    labels = extract_labels(state, label_mode=color_by, num_points=emb.shape[0])
    coords = run_umap(emb, n_neighbors=n_neighbors, min_dist=min_dist, n_components=3)

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "label": labels if labels is not None else ["var"] * coords.shape[0],
        }
    )

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        hover_name="label",
        title="Relational manifold (3D UMAP)",
    )
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[info] Saved 3D UMAP HTML to {out_html}")


def generate_umap_bundle(
    relational_state_path: str,
    out_dir: str,
    base_name: str,
    embedding_key: str = "emb",
):
    """
    Convenience helper: produce a small visualization bundle:

    - 2D domain-colored PNG
    - 3D domain-colored HTML
    - 2D relation-colored PNG
    - 3D relation-colored HTML
    """
    os.makedirs(out_dir, exist_ok=True)

    for color_by in ["domain", "relation"]:
        suffix = f"{base_name}_{color_by}"
        png_path = os.path.join(out_dir, f"{suffix}_2d.png")
        html_path = os.path.join(out_dir, f"{suffix}_3d.html")
        make_umap_2d_png(
            relational_state_path,
            png_path,
            embedding_key=embedding_key,
            color_by=color_by,
        )
        make_umap_3d_html(
            relational_state_path,
            html_path,
            embedding_key=embedding_key,
            color_by=color_by,
        )


# ---------- CLI for ad-hoc use ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relational-state", type=str, required=True,
                        help="Path to relational_state.pkl")
    parser.add_argument("--embedding-key", type=str, default="emb",
                        help="Key/attribute for embeddings (default: emb)")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2,
                        help="UMAP output dimension (2 or 3)")
    parser.add_argument("--title", type=str, default="Relational UMAP")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["domain", "relation", "none"],
        default="domain",
        help="Color by 'domain' (dom_ids/DOM2ID), 'relation' (rel_ids/REL2ID), or 'none'.",
    )
    args = parser.parse_args()

    # Simple CLI: just show a non-saved plot (as before).
    state = load_relational_state(args.relational_state)
    
    print("[debug] DOM2ID:", state.get("DOM2ID", {}))
    print("[debug] dom_ids shape/len:", len(state.get("dom_ids", [])))

    emb = extract_embeddings(state, embedding_key=args.embedding_key)
    print(f"[info] Embeddings shape: {emb.shape}")
    labels = extract_labels(state, label_mode=args.color_by, num_points=emb.shape[0])
    coords = run_umap(
        emb,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.dim,
    )

    if args.dim == 2:
        # quick interactive view
        x = coords[:, 0]
        y = coords[:, 1]
        plt.figure(figsize=(8, 6))
        if labels is None:
            plt.scatter(x, y, s=5, alpha=0.8)
        else:
            unique = sorted(set(labels))
            label_to_int = {lab: i for i, lab in enumerate(unique)}
            colors = [label_to_int[lab] for lab in labels]
            sc = plt.scatter(x, y, c=colors, s=5, alpha=0.8)
            cbar = plt.colorbar(sc, ticks=range(len(unique)))
            cbar.ax.set_yticklabels(unique)
        plt.title(args.title)
        plt.tight_layout()
        plt.show()
    else:
        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "label": labels if labels is not None else ["var"] * coords.shape[0],
            }
        )
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="label",
            hover_name="label",
            title=args.title,
        )
        fig.show()


if __name__ == "__main__":
    main()
