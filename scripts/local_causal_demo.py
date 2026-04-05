"""
gt_local_causal_demo.py

Local causal neighborhood + Geometric Transformer visualization.

- Loads a global causal graph from causal_statements.jsonl.
- Every few seconds, picks a focus node (or cycles through a curated list).
- Extracts a local ego-graph around the focus node.
- Builds a tiny simplicial complex (nodes, edges, triangles).
- Runs a single GeometricTransformerModule forward pass in MLX.
- Visualizes the local graph with node size/color based on GT activations.

To run:

    (.venv_mumble_mlx) python -m scripts.gt_local_causal_demo

Then open http://localhost:8051 in your browser.
"""

import json
import random

import networkx as nx
import numpy as np
import mlx.core as mx
import mlx.nn as nn

import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

from mlx_gt.gt_module_mx import GeometricTransformerModule


# --------- 1. Build global causal graph from JSONL ---------


def load_causal_graph(statements_path: str) -> nx.DiGraph:
    """
    Build a directed graph from causal_statements.jsonl.

    Each record is expected to have keys:
      - subj: cause
      - obj: effect
      - rel: relation label (e.g., 'increases', 'reduces')
      - topic: source topic
      - domain: econ domain (e.g., 'Monetary Policy')
      - path: topic path (optional)
    """
    G = nx.DiGraph()
    with open(statements_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            subj = rec.get("subj")
            obj = rec.get("obj")
            rel = rec.get("rel", "")
            topic = rec.get("topic", "")
            domain = rec.get("domain", topic or "Unknown")
            path = rec.get("path", [])

            if not subj or not obj:
                continue

            # Add nodes with attributes
            if subj not in G:
                G.add_node(subj, domain=domain)
            if obj not in G:
                G.add_node(obj, domain=domain)

            # Add directed edge with attributes
            G.add_edge(
                subj,
                obj,
                rel=rel,
                topic=topic,
                path=path,
            )

    print(f"[CausalGraph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def pick_focus_node(G: nx.DiGraph, seed_nodes=None) -> str:
    """
    Pick a focus node from a curated list if provided, else any node.
    """
    if seed_nodes:
        candidates = [n for n in seed_nodes if n in G]
        if candidates:
            return random.choice(candidates)

    # Fallback: pick a random node with out-degree > 0
    nodes = [n for n, d in G.out_degree() if d > 0]
    if not nodes:
        nodes = list(G.nodes())
    return random.choice(nodes) if nodes else None


def ego_causal_subgraph(
    G: nx.DiGraph, focus_node: str, radius: int = 1, max_nodes: int = 40
) -> nx.DiGraph:
    """
    Extract a local directed ego-graph around focus_node.

    radius=1 => immediate in- and out-neighbors.
    If the resulting subgraph is too large, prune to highest-degree nodes.
    """
    H = nx.ego_graph(G, focus_node, radius=radius, undirected=False)
    if H.number_of_nodes() > max_nodes:
        # simple heuristic: keep top-degree nodes
        degs = sorted(H.degree, key=lambda x: x[1], reverse=True)
        keep = {focus_node} | {node for node, _ in degs[: max_nodes - 1]}
        H = H.subgraph(keep).copy()
    return H


# --------- 2. Build simplicial complex inputs for GT ---------


def build_simplicial_inputs(H: nx.DiGraph):
    """
    Convert a small networkx DiGraph into:
      - nodes: list of node names
      - x: [N, in_dim] node features (degree + domain id)
      - edge_index: [2, E] directed edges
      - triangles: [T, 3] (simple motif-based 2-simplices)

    We use:
      in_dim = 2 (degree, domain_id).
    """

    nodes = list(H.nodes())
    N = len(nodes)
    index = {n: i for i, n in enumerate(nodes)}

    # Map domains to integer ids (0..D-1)
    domains = [H.nodes[n].get("domain", "Unknown") for n in nodes]
    unique_domains = sorted(set(domains))
    dom2id = {d: i for i, d in enumerate(unique_domains)}
    dom_ids = np.array([dom2id[d] for d in domains], dtype=np.float32)

    # Node degree (undirected degree used as a simple structural feature)
    degs = np.array([H.degree[n] for n in nodes], dtype=np.float32)

    # Node features: [degree, domain_id]
    x = np.stack([degs, dom_ids], axis=1)  # [N, 2]

    # Edge index: directed edges
    edges = []
    for u, v in H.edges():
        i = index[u]
        j = index[v]
        edges.append([i, j])
    if edges:
        edge_index = np.array(edges, dtype=np.int32).T  # [2, E]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)

    # Triangles as simple 2-simplices: any (u -> v -> w) path
    triangles = []
    for u, v in H.edges():
        for _, w in H.out_edges(v):
            if u != w and (u in index) and (v in index) and (w in index):
                tri = tuple(sorted([index[u], index[v], index[w]]))
                triangles.append(tri)
    triangles = sorted(set(triangles))
    if triangles:
        triangles = np.array(triangles, dtype=np.int32)
    else:
        triangles = np.zeros((0, 3), dtype=np.int32)

    return nodes, x, edge_index, triangles, domains


# --------- 3. Run GT forward to get node activations ---------


def gt_forward_activations(
    model: GeometricTransformerModule,
    x_np: np.ndarray,
    edge_index_np: np.ndarray,
    triangles_np: np.ndarray,
) -> np.ndarray:
    """
    Run a single forward pass of GeometricTransformerModule in MLX
    and return a scalar activation per node.

    For the demo, we simply take the L2 norm of the final node embeddings.
    """
    x = mx.array(x_np, dtype=mx.float32)                 # [N, 2]
    edge_index = mx.array(edge_index_np, dtype=mx.int32) # [2, E]
    triangles = mx.array(triangles_np, dtype=mx.int32)   # [T, 3]

    # For this local graph, treat it as a single graph => batch=None
    # GeometricTransformerModule expects in_dim=2
    h = model.input_proj(x)  # manually replicate __call__ to keep triangles

    for layer, norm in zip(model.layers, model.norms):
        h = layer(h, edge_index, triangles=None if triangles.shape[0] == 0 else triangles)
        h = norm(h)
        h = model.relu(h)

    # Node-level embeddings: h [N, hidden_dim]
    # Compute scalar activation as L2 norm
    activations = mx.sqrt(mx.sum(h * h, axis=1))  # [N]
    return np.array(activations)  # back to numpy for plotting


# --------- 4. Dash app for local GT visualization ---------

def layout_and_plot(H: nx.DiGraph, node_score: np.ndarray, focus_node: str) -> go.Figure:
    pos = nx.spring_layout(H, seed=0, k=0.8)
    nodes = list(H.nodes())
    N = len(nodes)
    scores = node_score

    # Normalize scores for marker size
    s_min = scores.min() if N > 0 else 0.0
    s_max = scores.max() if N > 0 else 1.0
    s_range = max(s_max - s_min, 1e-6)
    sizes = 12 + 24 * (scores - s_min) / s_range

    xs = []
    ys = []
    texts = []

    for n in nodes:
        x, y = pos[n]
        xs.append(x)
        ys.append(y)
        texts.append(n)

    fig = go.Figure()

    # Edges as lines
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines",
                showlegend=False,
            )
        )

    # Nodes
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=scores,
                colorscale="Viridis",
                opacity=0.9,
                line=dict(width=1, color="white"),
            ),
            text=texts,
            textposition="top center",
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        )
    )

    # Add arrowheads via annotations
    annotations = []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=3,
                arrowwidth=1,
                arrowcolor="red",
            )
        )

    fig.update_layout(
        title=f"Causal neighborhood of: {focus_node}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=16),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        annotations=annotations,
    )

    return fig


# --------- 5. Wire everything into a Dash loop ---------


# Adjust the path below to point to your econ causal_statements.jsonl
CAUSAL_STATEMENTS_PATH = "causal_statements.jsonl"

def load_causal_graph(statements_path: str) -> nx.DiGraph:
    """
    Build a directed graph from causal_statements.jsonl where each record is:
      {"topic": ..., "path": [...], "question": ..., "statements": [...]}

    We create:
      - a node for the topic,
      - a node for each statement string,
      - edges: topic -> statement.
    """
    G = nx.DiGraph()
    with open(statements_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            topic = rec.get("topic")
            domain = topic or "Unknown"  # for now, treat topic as domain
            stmts = rec.get("statements", [])

            if not topic:
                continue

            # Add topic node
            if topic not in G:
                G.add_node(topic, domain=domain)

            for s in stmts:
                s_clean = s.strip()
                if not s_clean:
                    continue
                # Add statement node with same domain (or some other heuristic)
                if s_clean not in G:
                    G.add_node(s_clean, domain=domain)
                # Add edge topic -> statement
                G.add_edge(topic, s_clean, rel="describes", topic=topic)

    print(f"[CausalGraph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def create_app():
    G = load_causal_graph(CAUSAL_STATEMENTS_PATH)

    # A few hand-picked seeds (if they exist in G); else we pick random nodes
    seed_nodes = [
        "an increase in interest rates",
        "inflation",
        "unemployment",
        "expansionary fiscal policy",
        "the price of bonds",
    ]

    # Tiny GT model for demo (no training; random weights)
    model = GeometricTransformerModule(
        in_dim=2,
        hidden_dim=32,
        depth=2,
        num_relations_1=1,
        num_relations_2=1,
        out_dim=1,
    )
    mx.eval(model.parameters())  # initialize lazily

    app = Dash(__name__)

    app.layout = html.Div(
        style={"backgroundColor": "black", "height": "100vh", "color": "white"},
        children=[
            html.Div(
                style={"textAlign": "center", "padding": "8px 0px"},
                children=[
                    html.H1(
                        "Local Causal Graph + Geometric Transformer",
                        style={"color": "white", "fontSize": "28px", "margin": "0px"},
                    ),
                    html.Div(
                        "Every few seconds we pick a new focus and show its causal neighborhood with GT activations.",
                        style={"color": "white", "fontSize": "16px"},
                    ),
                ],
            ),
            dcc.Graph(
                id="local-graph",
                style={"height": "85vh"},
            ),
            dcc.Interval(
                id="interval",
                interval=5000,  # 5 seconds per neighborhood
                n_intervals=0,
            ),
        ],
    )

    @app.callback(
        Output("local-graph", "figure"),
        Input("interval", "n_intervals"),
    )
    def update_local_graph(n):
        focus = pick_focus_node(G, seed_nodes=seed_nodes)
        if focus is None:
            return go.Figure()

        H = ego_causal_subgraph(G, focus, radius=1, max_nodes=30)
        if H.number_of_nodes() == 0:
            return go.Figure()

        nodes, x_np, edge_index_np, triangles_np, domains = build_simplicial_inputs(H)
        activations = gt_forward_activations(model, x_np, edge_index_np, triangles_np)
        fig = layout_and_plot(H, activations, focus)
        return fig

    return app


if __name__ == "__main__":
    app = create_app()
    # Run on a separate port so it doesn't collide with your econ manifold viewer
    app.run(debug=False, host="0.0.0.0", port=8051)
