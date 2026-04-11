# scripts/visualize_manifold.py
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)


def _clean_hover_label(value: object) -> str:
    text = " ".join(str(value or "").replace("_", " ").split())
    return text[:96].strip()


def _compute_hover_clusters(state: dict, umap_2: np.ndarray) -> list[dict[str, object]]:
    vars_list = list(state.get("vars") or [])
    if umap_2 is None or getattr(umap_2, "ndim", 0) != 2 or umap_2.shape[1] != 2:
        return []
    point_count = min(len(vars_list), int(umap_2.shape[0]))
    if point_count < 2:
        return []

    coords = np.asarray(umap_2[:point_count], dtype=float)
    labels = [_clean_hover_label(item) for item in vars_list[:point_count]]
    span = np.ptp(coords, axis=0)
    span[span == 0] = 1.0

    cluster_count = max(3, min(8, int(round(np.sqrt(point_count / 18.0))) + 2))
    cluster_count = min(cluster_count, point_count)
    seed_indices = np.linspace(0, point_count - 1, num=cluster_count, dtype=int)
    centers = coords[seed_indices].copy()
    assignments = np.zeros(point_count, dtype=int)

    for _ in range(10):
        distances = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assignments = distances.argmin(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_index in range(cluster_count):
            mask = assignments == cluster_index
            if np.any(mask):
                centers[cluster_index] = coords[mask].mean(axis=0)

    x_min, y_min = coords.min(axis=0)
    x_span, y_span = span.tolist()
    clusters: list[dict[str, object]] = []
    for cluster_index in range(cluster_count):
        member_indices = np.where(assignments == cluster_index)[0]
        if len(member_indices) == 0:
            continue
        cluster_coords = coords[member_indices]
        centroid = cluster_coords.mean(axis=0)
        distances = np.sqrt(((cluster_coords - centroid) ** 2).sum(axis=1))
        order = np.argsort(distances)
        sample_labels: list[str] = []
        for member_offset in order:
            label = labels[member_indices[int(member_offset)]]
            if not label or label in sample_labels:
                continue
            sample_labels.append(label)
            if len(sample_labels) >= 4:
                break
        if not sample_labels:
            continue
        radius = float(np.quantile(distances, 0.85)) if len(distances) > 1 else 0.0
        normalized_radius = max(radius / max(x_span, y_span), 0.05)
        clusters.append(
            {
                "id": f"cluster-{cluster_index + 1}",
                "label": " / ".join(sample_labels[:3]),
                "sample_labels": sample_labels,
                "point_count": int(len(member_indices)),
                "x_norm": float((centroid[0] - x_min) / x_span) if x_span else 0.5,
                "y_norm": float(1.0 - ((centroid[1] - y_min) / y_span)) if y_span else 0.5,
                "radius_norm": float(min(0.22, max(0.055, normalized_radius * 1.35))),
            }
        )
    clusters.sort(key=lambda item: (-int(item["point_count"]), str(item["label"])))
    return clusters


def _write_hover_metadata(out_dir: str, clusters: list[dict[str, object]]) -> None:
    if not clusters:
        return
    out_path = os.path.join(out_dir, "relational_manifold_labels.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"clusters": clusters}, handle, indent=2)

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
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.margins(0.08)
        out_path_2d = os.path.join(out_dir, "relational_manifold_2d.png")
        plt.tight_layout(pad=0.25)
        plt.savefig(out_path_2d, dpi=150, bbox_inches="tight", pad_inches=0.03)
        plt.close()
        _write_hover_metadata(out_dir, _compute_hover_clusters(state, umap_2))
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
