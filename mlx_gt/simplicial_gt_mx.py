import mlx.core as mx
import mlx.nn as nn


class SimplicialMessagePassingMX:
    """
    1-skeleton + optional 2-simplices (triangles) in MLX.

    Plain Python class, not subclassing nn.Module. We still use MLX layers
    (nn.Linear, nn.Embedding) for params, but rely on __call__ for forward.
    """

    def __init__(self, dim, num_relations_1=1, num_relations_2=1):
        self.rel1 = nn.Embedding(num_relations_1, dim)
        self.rel2 = nn.Embedding(num_relations_2, dim)

        self.lin_msg1 = nn.Linear(dim, dim)
        self.lin_msg2 = nn.Linear(dim, dim)

        # simple MLP for update: x_next = x + MLP(msgs)
        self.lin_upd1 = nn.Linear(dim, dim)
        self.lin_upd2 = nn.Linear(dim, dim)

        self.relu = nn.ReLU()

    def __call__(self, x, edge_index, triangles=None):
        # x: [N, D], edge_index: [2, E]
        N, D = x.shape
        msgs = mx.zeros_like(x)

        # -------- 1-skeleton messages (edges) --------
        src = edge_index[1]   # [E]
        dst = edge_index[0]   # [E]

        # One relation id for all edges
        rel_ids1 = mx.zeros_like(src)
        r1 = self.rel1(rel_ids1)           # [E, D]

        x_src = x[src]                     # [E, D]
        msgs_e1 = self.relu(self.lin_msg1(x_src) + r1)  # [E, D]

        # Scatter-add into msgs[d] += msgs_e1[e] where d = dst[e]
        for e in range(msgs_e1.shape[0]):
            i = int(dst[e])
            msgs = msgs.at[i].add(msgs_e1[e])

        # -------- 2-simplices messages (triangles -> nodes) --------
        if triangles is not None and triangles.size != 0:
            tri = triangles                  # [T,3]
            tri_emb = x[tri].mean(axis=1)    # [T, D]

            rel_ids2 = mx.zeros(tri_emb.shape[0], dtype=triangles.dtype)
            r2 = self.rel2(rel_ids2)         # [T, D]
            msgs_t = self.relu(self.lin_msg2(tri_emb) + r2)  # [T, D]

            # scatter triangle messages to their 3 vertices
            T = tri.shape[0]
            tri_rep = mx.repeat(msgs_t, 3, axis=0)       # [T*3, D]
            verts   = tri.reshape((T * 3,))              # [T*3]

            for e in range(tri_rep.shape[0]):
                i = int(verts[e])
                msgs = msgs.at[i].add(tri_rep[e])

        # -------- Residual MLP update: x_next = x + MLP(msgs) --------
        upd = self.relu(self.lin_upd1(msgs))
        upd = self.lin_upd2(upd)
        x_next = x + upd
        return x_next


class GeometricTransformerMX:
    """
    Tiny Geometric Transformer in MLX:

      - project input to hidden_dim,
      - apply several SimplicialMessagePassingMX layers,
      - mean-pool to graph-level embedding,
      - MLP head to predict graph label.

    Plain Python class, callable via __call__.
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
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = [
            SimplicialMessagePassingMX(
                hidden_dim,
                num_relations_1=num_relations_1,
                num_relations_2=num_relations_2,
            )
            for _ in range(depth)
        ]

        self.norms = [nn.LayerNorm(hidden_dim) for _ in range(depth)]

        self.relu = nn.ReLU()
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def __call__(self, x, edge_index, triangles=None, batch=None):
        """
        x:          [N, in_dim]
        edge_index: [2, E]
        triangles:  [T, 3]
        batch:      [N] graph ids, or None (assume single graph)
        """
        # Project input features
        h = self.input_proj(x)  # [N, hidden_dim]

        # Stack message-passing layers + norm + ReLU
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, triangles=triangles)
            h = norm(h)
            h = self.relu(h)

        # Graph-level pooling
        if batch is None:
            g = h.mean(axis=0, keepdims=True)  # [1, hidden_dim]
        else:
            # TODO: implement proper batching; for now assume single graph
            g = h.mean(axis=0, keepdims=True)

        # Output head
        out = self.mlp_out(g)  # [B, out_dim]
        return out
