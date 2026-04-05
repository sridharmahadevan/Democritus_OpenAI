# gt/simplicial_mp.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialMessagePassing(nn.Module):
    """
    Core GAIA-style message passing over 0/1/2-simplices.

    This is the thing that knows how to:
      - aggregate neighbor messages along edges / triangles
      - push information up and down the simplicial structure
    """

    def __init__(self, dim: int, num_rel: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + num_rel, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        # Add any other layers you used (triangle MLPs, norm, etc.)

    def forward(
        self,
        V: torch.Tensor,
        edge_index: torch.Tensor,
        rel_ids: torch.Tensor,
        dom_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        edge_index:
        - either shape (2, E)  → [src, dst]
        - or shape (E, 2)      → rows [src, dst]
        We'll normalize to (2, E) internally.
        """
        if edge_index.ndim != 2:
            raise ValueError(f"edge_index must be 2D, got shape {edge_index.shape}")

        # If edges are stored as (E, 2) in the relational_state, fix that here
        if edge_index.shape[0] == 2:
            # shape (2, E): rows are src, dst
            src = edge_index[0].long()
            dst = edge_index[1].long()
        elif edge_index.shape[1] == 2:
            # shape (E, 2): columns are src, dst
            src = edge_index[:, 0].long()
            dst = edge_index[:, 1].long()
        else:
            raise ValueError(
                f"edge_index must have shape (2, E) or (E, 2), got {edge_index.shape}"
        )

        src_h = V[src]
        dst_h = V[dst]

        # One-hot encode relations (assuming rel_ids is length E)
        num_rel = int(rel_ids.max().item()) + 1 if rel_ids.numel() > 0 else 0
        if num_rel > 0:
            rel_onehot = F.one_hot(rel_ids.long(), num_classes=num_rel).float()
        else:
            rel_onehot = torch.zeros(rel_ids.size(0), 0, device=V.device)

        edge_feat = torch.cat([src_h, dst_h, rel_onehot], dim=-1)
        msg = self.edge_mlp(edge_feat)  # (E, D)

        out = torch.zeros_like(V)
        out.index_add_(0, dst, msg)

        return V + out
