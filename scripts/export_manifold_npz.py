import pickle
import numpy as np

# 1. Load manifold_state from Module 5 of the Qwen80B run
with open("manifold_state.pkl", "rb") as f:
    M = pickle.load(f)

coords = np.array(M["umap_3d"])   # [N1, 3]
topics = np.array(M["vars"])      # [N2]
dom_ids = np.array(M["dom_ids"])  # [N3]
DOM2ID = M["DOM2ID"]              # dict: domain_name -> id

# 2. Clamp to common length
N = min(len(coords), len(topics), len(dom_ids))
coords = coords[:N]
topics = topics[:N]
dom_ids = dom_ids[:N]

print(f"Exporting {N} manifold points (no depth filtering).")

# 3. Clean labels for display
def clean_label(s: str) -> str:
    s = s.replace("<|end|>", "")
    s = s.replace("<|assistant|>", "")
    s = s.replace("answer:", "")
    return s.strip()

clean_topics = np.array([clean_label(str(t)) for t in topics])

# 4. Convert domain ids to names
id2dom = {v: k for k, v in DOM2ID.items()}
domains = np.array([id2dom.get(int(d), "Unknown") for d in dom_ids])

# 5. Save to npz for the viewer
np.savez(
    "demo_data/manifold.npz",
    coords=coords,
    topics=clean_topics,
    domains=domains,
)

print("Saved demo_data/manifold.npz")
