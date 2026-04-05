"""
Module 5 (v5 — Simplicial Manifold Builder)
------------------------------------------

Input:
    relational_state.pkl   <-- output of Module 4

Output:
    manifold_state.pkl     <-- refined embeddings, simplices, domain maps
    umap_2d.npy
    umap_3d.npy

This module:
    1. Loads relational graph + SBERT embeddings.
    2. Refines embeddings using a lightweight Geometric Transformer v2.
    3. Builds multi-relational simplicial structure (0-, 1-, 2-simplices).
    4. Computes UMAP(2D) and UMAP(3D) layout embeddings.
"""

from __future__ import annotations

import json
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import umap
import os
import json

from sentence_transformers import SentenceTransformer

import argparse
from pathlib import Path

DEFAULT_TOPICS_FILE = "configs/root_topics.txt"


def load_root_topics(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Root topics file not found: {path}")
    topics = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            topics.append(line)
    if not topics:
        raise ValueError(f"No topics found in {path}")
    print(f"[Module 1] Loaded {len(topics)} root topics from {path}")
    return topics


# ============================================================
# CONFIG
# ============================================================

IN_STATE   = "relational_state.pkl"
OUT_STATE  = "manifold_state.pkl"

UMAP_2D    = "umap_2d.npy"
UMAP_3D    = "umap_3d.npy"

IN_TRIPLES = "relational_triples.jsonl"  # <-- add this line

DEVICE = "cpu"   # M1/M2/M3/M4 — CPU is actually faster for UMAP+GT here

# ============================================================
# Initialize relational state from triples (fallback)
# ============================================================

_SBERT_MODEL = None

def get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        # 384-dim model, matches the default GT dim
        print("[Module 5] Loading SentenceTransformer (all-MiniLM-L6-v2)…")
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _pick_key(rec, candidates):
    for k in candidates:
        if k in rec:
            return rec[k]
    raise KeyError(f"None of {candidates} found in record: {rec}")


def init_rel_state_from_triples():
    """
    Fallback initializer: if relational_state.pkl doesn't exist yet,
    build a minimal relational state from relational_triples.jsonl.

    - Nodes: unique heads/tails
    - Edges: (head_idx, tail_idx) per triple
    - Relations: map relation strings to integer IDs
    - Domains: map rec['domain'] or rec['topic'] to DOM2ID and per-edge dom_ids
    - Embeddings: SBERT embeddings of variable names
    """
    triples_path = Path(IN_TRIPLES)
    if not triples_path.exists():
        raise FileNotFoundError(
            f"{IN_STATE} not found and {IN_TRIPLES} not found either. "
            "Run Module 4 (triple extractor) first."
        )

    print(f"[Module 5] Initializing relational_state from {IN_TRIPLES}…")

    vars_list = []
    var2idx = {}
    idx2var = {}
    edges = []
    rel_labels = []
    dom_labels = []  # <-- domain label per triple

    with triples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # tolerant key selection
            h = _pick_key(rec, ["head", "src", "source", "subj"])
            t = _pick_key(rec, ["tail", "dst", "target", "obj"])
            r = _pick_key(rec, ["relation", "rel", "predicate"])

            # domain label: prefer 'domain', fall back to 'topic', else 'var'
            d = rec.get("domain") or rec.get("topic") or "var"

            for name in (h, t):
                if name not in var2idx:
                    idx = len(vars_list)
                    var2idx[name] = idx
                    idx2var[idx] = name
                    vars_list.append(name)

            h_idx = var2idx[h]
            t_idx = var2idx[t]
            edges.append((h_idx, t_idx))
            rel_labels.append(r)
            dom_labels.append(d)

    num_nodes = len(vars_list)
    num_edges = len(edges)
    print(f"[Module 5] Found {num_nodes} variables and {num_edges} edges.")

    # Build REL2ID mapping
    REL2ID = {}
    rel_ids = []
    for r in rel_labels:
        if r not in REL2ID:
            REL2ID[r] = len(REL2ID)
        rel_ids.append(REL2ID[r])

    # Build DOM2ID mapping from domain labels
    DOM2ID = {}
    dom_ids = []
    for d in dom_labels:
        if d not in DOM2ID:
            DOM2ID[d] = len(DOM2ID)
        dom_ids.append(DOM2ID[d])

    # id -> label map for convenience
    domains = {dom_id: label for label, dom_id in DOM2ID.items()}

    # Compute SBERT embeddings for variable names
    model = get_sbert_model()
    emb_np = model.encode(vars_list, convert_to_numpy=True)
    emb = torch.from_numpy(emb_np.astype("float32"))

    # Pack into a state dict in the format Module 5 expects
    state = {
        "vars": vars_list,
        "var2idx": var2idx,
        "idx2var": idx2var,

        "emb": emb,                                     # [N, d] torch tensor
        "edges": torch.tensor(edges, dtype=torch.long), # [E, 2]
        "rel_ids": torch.tensor(rel_ids, dtype=torch.long),  # per-edge
        "dom_ids": torch.tensor(dom_ids, dtype=torch.long),  # per-edge

        "REL2ID": REL2ID,
        "DOM2ID": DOM2ID,
        "domains": domains,
    }

    print("[Module 5] Initialized relational_state with "
          f"{len(vars_list)} vars, {len(edges)} edges, "
          f"{len(REL2ID)} relation types, {len(DOM2ID)} domains.")

    # Save it so future runs can just load
    with open(IN_STATE, "wb") as f:
        pickle.dump(state, f)
    print(f"[Module 5] Saved new relational state to {IN_STATE}")

    return state
    
# ============================================================
# Load relational state
# ============================================================

def load_rel_state():
    if Path(IN_STATE).exists():
        print(f"[Module 5] Loading existing relational state from {IN_STATE}…")
        return pickle.load(open(IN_STATE, "rb"))

    # Fallback: build from triples
    return init_rel_state_from_triples()
    
    
#==========================================
# Geometric Transformer MOE version
#=========================================

class GeomMoEBlock(nn.Module):
    """
    Simple dense MoE over node embeddings.

    - Input:  H ∈ R^{N×D}
    - Gating: softmax over E experts per node
    - Experts: small 2-layer MLPs with ReLU
    - Output: LayerNorm(H + Σ_e w_e * expert_e(H))

    This is intentionally simple & stable; it reuses the same
    dimensionality as the Democritus manifold embeddings.
    """

    def __init__(self, dim: int, n_experts: int = 4, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.n_experts = n_experts
        self.gate = nn.Linear(dim, n_experts)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(n_experts)
            ]
        )

        self.ln = nn.LayerNorm(dim)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: (N, D)
        """
        # gating weights per node
        logits = self.gate(H)              # (N, E)
        weights = F.softmax(logits, dim=-1)  # (N, E)

        # dense MoE: sum_e w_e * expert_e(H)
        out = torch.zeros_like(H)
        for e, expert in enumerate(self.experts):
            # expert(H): (N, D)
            h_e = expert(H)
            w_e = weights[:, e].unsqueeze(-1)  # (N, 1)
            out = out + w_e * h_e

        # residual + layer norm
        return self.ln(H + out)




# ============================================================
# Step 1 — Run Geometric Transformer refinement
# ============================================================


# and wherever your full GT is defined:
from gt import GeometricTransformerV2
import torch.nn.functional as F

def refine_embeddings(state, mode: str = "full"):
    """
    mode:
      - "full":    just GeometricTransformerV2 (current Democritus default)
      - "lite":    (optional) GT-lite / conv-based smoother if you had one
      - "moe":     GT-Full + GeomMoEBlock on top, over node embeddings
    """
    emb     = state["emb"].to(DEVICE)         # (N, D)
    edges   = state["edges"].to(DEVICE)       # (2, E)
    rel_ids = state["rel_ids"].to(DEVICE)     # (E,)
    dom_ids = state["dom_ids"].to(DEVICE)     # (E,) or None

    num_rel = len(state["REL2ID"])
    dim     = emb.shape[1]

    # --- Step 1: geometric transformer (graph-based) ---
    gt = GeometricTransformerV2(dim=dim, depth=2, num_rel=num_rel).to(DEVICE)

    with torch.no_grad():
        H = gt(emb, edges, rel_ids, dom_ids)  # (N, D)

    # --- Step 2: optional MoE or other variant on top ---
    if mode == "full":
        print("[Module 5] Using GT-Full manifold refinement")
        V_ref = H
    elif mode == "moe":
        print("[Module 5] Using GT-Full + MoE manifold refinement")
        moe = GeomMoEBlock(dim=dim, n_experts=4).to(DEVICE)
        with torch.no_grad():
            V_ref = moe(H)
    elif mode == "lite":
        print("[Module 5] GT-lite path not implemented here; falling back to GT-Full")
        V_ref = H
    else:
        raise ValueError(f"Unknown refinement mode: {mode}")

    return V_ref.cpu()
    
# ============================================================
# Step 2 — Build simplicial structure
# ============================================================

def build_simplicial_structure(state):
    """
    0-simplices = variables
    1-simplices = edges
    2-simplices = domain-coherent triples  (simple but effective)
    """
    vars_list = state["vars"]
    edges     = state["edges"].tolist()
    domains   = state["dom_ids"].tolist()

    two_simplices = []

    print("[Module 5] Building 2-simplices (domain-based)…")
    # naive but effective:
    # any pair (i→j, j→k) in SAME DOMAIN => 2-simplex (i,j,k)
    outgoing = {}
    for idx, (i, j) in enumerate(edges):
        outgoing.setdefault(i, []).append((j, domains[idx]))

    for i in outgoing:
        for (j, d1) in outgoing[i]:
            if j in outgoing:
                for (k, d2) in outgoing[j]:
                    if d1 == d2:
                        two_simplices.append((i, j, k, d1))

    return two_simplices


# ============================================================
# Step 3 — UMAP layouts
# ============================================================

def compute_umap_embeddings(V_ref):
    print("[Module 5] Running UMAP (2D)…")
    reducer_2d = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    U2 = reducer_2d.fit_transform(V_ref.numpy())

    print("[Module 5] Running UMAP (3D)…")
    reducer_3d = umap.UMAP(n_components=3, metric="cosine", random_state=42)
    U3 = reducer_3d.fit_transform(V_ref.numpy())

    return U2, U3


# ============================================================
# Step 4 — Save manifold
# ============================================================

def save_manifold(state, V_ref, two_simplices, U2, U3):
    M = {
        "vars": state["vars"],
        "var2idx": state["var2idx"],
        "idx2var": state["idx2var"],

        "edges": state["edges"],
        "rel_ids": state["rel_ids"],
        "dom_ids": state["dom_ids"],

        "domains": state["domains"],
        "REL2ID": state["REL2ID"],
        "DOM2ID": state["DOM2ID"],

        "emb_refined": V_ref,
        "simplices_2": two_simplices,
        "umap_2d": U2,
        "umap_3d": U3,
    }

    print("[Module 5] Saving manifold →", OUT_STATE)
    pickle.dump(M, open(OUT_STATE, "wb"))

    np.save(UMAP_2D, U2)
    np.save(UMAP_3D, U3)


# ============================================================
# MAIN
# ============================================================
def main(mode: str = "full"):
    print("[Module 5] Loading relational state…")
    state = load_rel_state()

    print(f"[Module 5] Step 1 — refinement mode = {mode}")
    V_ref = refine_embeddings(state, mode=mode)

    print("[Module 5] Step 2 — simplicial structure")
    two_simplices = build_simplicial_structure(state)

    print("[Module 5] Step 3 — UMAP")
    U2, U3 = compute_umap_embeddings(V_ref)

    print("[Module 5] Step 4 — save manifold")
    save_manifold(state, V_ref, two_simplices, U2, U3)

    print("[Module 5] COMPLETE.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "lite", "moe"],
        default="full",
        help="Refinement mode for manifold builder",
    )
    args = parser.parse_args()
    main(mode=args.mode)
