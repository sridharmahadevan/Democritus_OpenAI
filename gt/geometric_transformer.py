# gt/geometric_transformer.py
from __future__ import annotations
import torch
import torch.nn as nn

from .simplicial_mp import SimplicialMessagePassing
import torch.nn.functional as F

class GeometricTransformerV2(nn.Module):
    """
    Geometric Transformer block used in Democritus:
      - takes base node embeddings
      - applies several layers of simplicial message passing
      - returns refined node embeddings
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_rel: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [SimplicialMessagePassing(dim=dim, num_rel=num_rel) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        V: torch.Tensor,         # (N, D)
        edge_index: torch.Tensor,  # (2, E)
        rel_ids: torch.Tensor,   # (E,)
        dom_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = V
        for layer in self.layers:
            h = layer(h, edge_index, rel_ids, dom_ids)
        return self.norm(h)
        
