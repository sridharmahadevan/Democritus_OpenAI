import mlx.core as mx
import mlx.nn as nn


class SimplicialMessagePassingModule(nn.Module):
    """
    MLX nn.Module version of the 2-simplices GT layer.

    Uses:
      - edge messages (1-simplices),
      - optional triangle -> node messages (2-simplices),
      - residual MLP update: x_next = x + MLP(msgs).
    """
    def __init__(self, dim, num_relations_1=1, num_relations_2=1):
        super().__init__()
        self.rel1 = nn.Embedding(num_relations_1, dim)
        self.rel2 = nn.Embedding(num_relations_2, dim)

        self.lin_msg1 = nn.Linear(dim, dim)
        self.lin_msg2 = nn.Linear(dim, dim)

        self.lin_upd1 = nn.Linear(dim, dim)
        self.lin_upd2 = nn.Linear(dim, dim)

        self.relu = nn.ReLU()

    def __call__(self, x, edge_index, triangles=None):
        # x: [N, D], edge_index: [2, E]
        N, D = x.shape
        msgs = mx.zeros_like(x)

        # --- 1-skeleton messages (edges) ---
        # edge_index: [2, E] with [dst, src] or [src, dst] depending on convention.
        # Here we assume edge_index[0] = dst, edge_index[1] = src (as in your scripts).
        dst = edge_index[0]  # [E]
        src = edge_index[1]  # [E]

        # All edges share relation id 0
        rel_ids1 = mx.zeros_like(src)
        r1 = self.rel1(rel_ids1)          # [E, D]

        x_src = x[src]                    # [E, D]
        msgs_e1 = self.relu(self.lin_msg1(x_src) + r1)  # [E, D]

        # Vectorized scatter-add: msgs[dst[e]] += msgs_e1[e]
        msgs = msgs.at[dst].add(msgs_e1)

        # --- 2-simplices messages (triangles -> nodes) ---
        if triangles is not None and triangles.size != 0:
            # triangles: [T,3] indices into nodes
            tri = triangles              # [T,3]
            tri_emb = x[tri].mean(axis=1)   # [T, D]

            rel_ids2 = mx.zeros(tri_emb.shape[0], dtype=triangles.dtype)
            r2 = self.rel2(rel_ids2)        # [T, D]
            msgs_t = self.relu(self.lin_msg2(tri_emb) + r2)  # [T, D]

            T = tri.shape[0]
            tri_rep = mx.repeat(msgs_t, 3, axis=0)    # [T*3, D]
            verts   = tri.reshape((T * 3,))           # [T*3]

            # Vectorized scatter-add for triangles
            msgs = msgs.at[verts].add(tri_rep)

        # --- residual MLP update: x_next = x + MLP(msgs) ---
        upd = self.relu(self.lin_upd1(msgs))
        upd = self.lin_upd2(upd)
        x_next = x + upd
        return x_next


class GeometricTransformerModule(nn.Module):
    """
    Geometric Transformer with triangles as a proper MLX nn.Module:

      - input_proj: in_dim -> hidden_dim
      - depth x SimplicialMessagePassingModule
      - LayerNorm + ReLU between layers
      - Graph mean pooling
      - MLP head: hidden_dim -> out_dim
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        depth=2,
        num_relations_1=1,
        num_relations_2=1,
        out_dim=1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = [
            SimplicialMessagePassingModule(
                hidden_dim,
                num_relations_1=num_relations_1,
                num_relations_2=num_relations_2,
            )
            for _ in range(depth)
        ]

        self.norms = [nn.LayerNorm(hidden_dim) for _ in range(depth)]
        self.relu  = nn.ReLU()

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def __call__(self, x, edge_index, triangles=None, batch=None):
        # x: [N, in_dim]
        h = self.input_proj(x)  # [N, hidden_dim]

        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, triangles=triangles)
            h = norm(h)
            h = self.relu(h)

        # single-graph pooling for now
        if batch is None:
            g = h.mean(axis=0, keepdims=True)  # [1, hidden_dim]
        else:
            # TODO: implement proper batching; for now assume single graph
            g = h.mean(axis=0, keepdims=True)

        out = self.mlp_out(g)  # [B, out_dim]
        return out
