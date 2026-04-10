#!/usr/bin/env python3
"""
make_credibility_bundle.py

One-command generator for:
  1) Ranked claim credibility CSV
  2) Full credibility report (Markdown)
  3) Executive tiered summary (Markdown)

Inputs:
  - scores.csv: output of your scoring pass (must include columns: file, focus, score;
               optionally lcm_json, n_nodes, n_edges)
  - relational_triples.jsonl: evidence triples (subj/obj/rel/statement)
  - lcm-dir: directory containing the LCM JSON files referenced by scores.csv

Usage examples:

# Antarctica
python -m scripts.make_credibility_bundle \
  --scores figs/antartica/scores.csv \
  --triples relational_triples.jsonl \
  --lcm-dir figs/antartica \
  --topk-models 5 \
  --topk-claims 30 \
  --alpha 1.0 \
  --tier1 0.60 --tier2 0.30 \
  --outdir figs/antartica/reports \
  --name antarctica

# WaPo chocolate
python -m scripts.make_credibility_bundle \
  --scores figs/wapo/scores.csv \
  --triples relational_triples.jsonl \
  --lcm-dir figs/wapo \
  --topk-models 5 \
  --outdir figs/wapo/reports \
  --name wapo_chocolate

Outputs (inside outdir):
  - <name>_credibility_claims_ranked.csv
  - <name>_credibility_report.md
  - <name>_executive_summary.md
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd

Edge = Tuple[str, str, str]  # (src, dst, rel)

_LEADING_PHRASE_PREFIXES = (
    "the use of ",
    "use of ",
    "treatment with ",
    "treatment using ",
    "administration of ",
    "exposure to ",
    "the discovery of ",
    "discovery of ",
    "the presence of ",
    "presence of ",
    "an increase in ",
    "increase in ",
    "a rise in ",
    "rise in ",
    "the reduction of ",
    "reduction of ",
    "the loss of ",
    "loss of ",
    "the decline in ",
    "decline in ",
)

_CLAUSE_MARKERS = (
    ", which ",
    " which ",
    " by ",
    " due to ",
    " through ",
    " allowing ",
    " because ",
    " while ",
    " when ",
)

_TEXT_REWRITES: Tuple[Tuple[str, str], ...] = (
    (r"\bglucagon[\s-]+like[\s-]+peptide[\s-]+1\b", "glp1"),
    (r"\bglp[\s-]*1\b", "glp1"),
    (r"\bglp1ras?\b", "glp1 receptor agonist"),
    (r"\bglp1 receptor agonists\b", "glp1 receptor agonist"),
    (r"\bindividuals\b", "people"),
    (r"\bpatients\b", "people"),
    (r"\bsubjects\b", "people"),
    (r"\bpersons\b", "people"),
    (r"\bmoving\b", "move"),
    (r"\bmoves\b", "move"),
)

_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "that",
    "the",
    "their",
    "this",
    "to",
    "which",
    "with",
}

_META_OBJECT_PREFIXES = (
    "our knowledge about ",
    "our knowledge of ",
    "our understanding of ",
    "the understanding of ",
    "the significance of understanding ",
    "the interpretation of ",
    "scientific theories about ",
    "scientific theories of ",
    "the timeline of ",
    "the evolutionary timeline of ",
    "knowledge about ",
    "knowledge of ",
    "understanding of ",
)

_RELATION_FAMILIES = {
    "leads_to": "causes",
    "leads to": "causes",
    "results_in": "causes",
    "results in": "causes",
    "drives": "causes",
    "causes": "causes",
    "cause": "causes",
    "influences": "affects",
    "influence": "affects",
    "affects": "affects",
    "affect": "affects",
    "increases": "increases",
    "increase": "increases",
    "raises": "increases",
    "raise": "increases",
    "boosts": "increases",
    "boost": "increases",
    "reduces": "reduces",
    "reduce": "reduces",
    "decreases": "reduces",
    "decrease": "reduces",
    "lowers": "reduces",
    "lower": "reduces",
    "supports": "supports",
    "support": "supports",
    "enables": "supports",
    "enable": "supports",
    "facilitates": "supports",
    "facilitate": "supports",
}


# ---------------------------
# Utilities
# ---------------------------
def softmax(scores: List[float], alpha: float = 1.0) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(alpha * (s - m)) for s in scores]
    Z = sum(exps) if exps else 1.0
    return [e / Z for e in exps]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def truncate(s: str, n: int = 180) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def canonical_edge(e: dict) -> Edge:
    return (e["src"], e["dst"], e.get("rel", "") or "")


def summarize_lcm_hub(lcm: dict, k: int = 8) -> Tuple[str, str, List[dict]]:
    """
    Choose hub as node with max outdegree and return its top-k outgoing edges.
    """
    focus = lcm.get("focus", "")
    edges = lcm.get("edges", [])
    outdeg: Dict[str, int] = {}
    for e in edges:
        outdeg[e["src"]] = outdeg.get(e["src"], 0) + 1
    hub = max(outdeg, key=outdeg.get) if outdeg else focus
    hub_edges = [e for e in edges if e["src"] == hub][:k]
    return focus, hub, hub_edges


def load_triple_index(triples_path: Path, max_examples_per_edge: int = 3) -> Tuple[Dict[Edge, int], Dict[Edge, List[str]]]:
    """
    Build (edge->count) and (edge->example statements) from triples.jsonl.
    """
    counts: Dict[Edge, int] = {}
    examples: Dict[Edge, List[str]] = {}

    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("subj") or r.get("source")
            o = r.get("obj") or r.get("target")
            rel = r.get("rel") or r.get("relation") or ""
            stmt = r.get("statement") or ""

            if not s or not o:
                continue

            key: Edge = (s, o, rel)
            counts[key] = counts.get(key, 0) + 1

            if stmt:
                lst = examples.setdefault(key, [])
                if len(lst) < max_examples_per_edge:
                    lst.append(stmt)

    return counts, examples


def normalize_relation_family(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    normalized = normalized.replace("-", "_")
    return _RELATION_FAMILIES.get(normalized, normalized)


def normalize_claim_surface(value: str, *, strip_meta_prefixes: bool = False) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for marker in _CLAUSE_MARKERS:
        if marker in text:
            prefix, _, _ = text.partition(marker)
            if prefix.strip():
                text = prefix.strip()
                break
    text = text.replace("_", " ")
    text = re.sub(r"[\-/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in _TEXT_REWRITES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    changed = True
    while changed and text:
        changed = False
        for prefix in _LEADING_PHRASE_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
    text = re.sub(r"^(?:the|a|an)\s+", "", text).strip()
    if strip_meta_prefixes:
        changed = True
        while changed and text:
            changed = False
            for prefix in _META_OBJECT_PREFIXES:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    changed = True
    return re.sub(r"\s+", " ", text).strip()


def claim_token_signature(value: str, *, strip_meta_prefixes: bool = False) -> Tuple[str, ...]:
    normalized = normalize_claim_surface(value, strip_meta_prefixes=strip_meta_prefixes)
    if not normalized:
        return ()
    tokens = [
        token for token in normalized.split()
        if token and token not in _TOKEN_STOPWORDS
    ]
    if not tokens:
        return ()
    return tuple(sorted(dict.fromkeys(tokens)))


def jaccard(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def quotient_claims_for_summary(claims_df: pd.DataFrame) -> pd.DataFrame:
    if claims_df.empty:
        return claims_df.copy()

    rows = claims_df.reset_index(drop=True).copy()
    rows["_canonical_rel"] = rows["rel"].apply(normalize_relation_family)
    rows["_subj_sig"] = rows["src"].apply(claim_token_signature)
    rows["_dst_sig"] = rows["dst"].apply(lambda value: claim_token_signature(value, strip_meta_prefixes=True))
    rows["_summary_rank"] = list(range(len(rows)))

    groups: List[List[int]] = []
    for idx, row in rows.iterrows():
        placed = False
        for group in groups:
            representative = rows.iloc[group[0]]
            if row["_canonical_rel"] != representative["_canonical_rel"]:
                continue
            subj_overlap = jaccard(row["_subj_sig"], representative["_subj_sig"])
            dst_overlap = jaccard(row["_dst_sig"], representative["_dst_sig"])
            if subj_overlap >= 0.85 and dst_overlap >= 0.6:
                group.append(idx)
                placed = True
                break
            if subj_overlap >= 0.75 and dst_overlap >= 0.8:
                group.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])

    kept_rows: List[dict] = []
    for group in groups:
        group_df = rows.iloc[group].copy()
        representative = group_df.sort_values(
            ["credibility", "support_count", "_summary_rank"],
            ascending=[False, False, True],
        ).iloc[0]
        merged_examples: List[str] = []
        for column in ("example_1", "example_2", "example_3"):
            for value in group_df[column].tolist() if column in group_df.columns else []:
                value = str(value or "").strip()
                if value and value not in merged_examples:
                    merged_examples.append(value)
        row_dict = dict(representative)
        row_dict["summary_variant_count"] = int(len(group_df))
        row_dict["summary_equivalence_size"] = int(len(group_df))
        row_dict["support_count"] = int(group_df["support_count"].max())
        row_dict["models_supporting"] = "; ".join(
            sorted(
                {
                    item.strip()
                    for value in group_df["models_supporting"].tolist()
                    for item in str(value or "").split(";")
                    if item.strip()
                }
            )
        )
        for offset, value in enumerate(merged_examples[:3], start=1):
            row_dict[f"example_{offset}"] = value
        kept_rows.append(row_dict)

    result = pd.DataFrame(kept_rows)
    result = result.sort_values(
        ["credibility", "support_count", "_summary_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return result.drop(columns=["_canonical_rel", "_subj_sig", "_dst_sig", "_summary_rank"], errors="ignore")


def make_tier_block(df: pd.DataFrame, title: str, max_items: int) -> List[str]:
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    if df.empty:
        lines.append("_No items in this tier._")
        lines.append("")
        return lines

    for i, row in enumerate(df.head(max_items).itertuples(index=False), start=1):
        cred = float(getattr(row, "credibility"))
        src = getattr(row, "src")
        rel = getattr(row, "rel")
        dst = getattr(row, "dst")
        ex1 = getattr(row, "example_1") if "example_1" in df.columns else ""
        models = getattr(row, "models_supporting") if "models_supporting" in df.columns else ""
        variant_count = int(getattr(row, "summary_equivalence_size", 1) or 1)

        lines.append(f"**{i}. ({cred:.2f}) {src} —{rel}→ {dst}**")

        ev = truncate(ex1, 180) if ex1 else ""
        if ev:
            lines.append(f"> {ev}")

        if isinstance(models, str) and models.strip():
            lines.append(f"- Supported by: {models}")
        if variant_count > 1:
            lines.append(f"- Consolidates {variant_count} nearby causal variants")

        lines.append("")

    return lines
    
def render_lcm_png(lcm: dict, out_png: Path, *, title: str = "", dpi: int = 200) -> None:
    """
    Render an LCM JSON (nodes + edges) to a PNG for quick inspection.
    Uses a simple spring layout (consistent with existing local_causal_dag style).
    """
    G = nx.DiGraph()
    for n in lcm.get("nodes", []):
        G.add_node(n)
    for e in lcm.get("edges", []):
        src, dst = e["src"], e["dst"]
        rel = e.get("rel", "")
        G.add_edge(src, dst, rel=rel)

    if G.number_of_nodes() == 0:
        return

    # Layout
    pos = nx.spring_layout(G, seed=0, k=0.8)

    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.axis("off")

    # edges: red arrows
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.4, shrinkA=6, shrinkB=6),
        )

    # nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color="gold",
        node_size=480,
        edgecolors="red",
        linewidths=1.2,
    )

    # labels (truncate long labels for readability)
    def short(s: str, n: int = 42) -> str:
        s = " ".join(str(s).split())
        return s if len(s) <= n else s[:n-3] + "..."

    labels = {n: short(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="blue")

    if title:
        plt.title(title, fontsize=11, color="blue")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def write_deep_dive_v1(
    out_path: Path,
    name: str,
    models: List[dict],
    tier1: pd.DataFrame,
    tier2: pd.DataFrame,
    tier3: pd.DataFrame,
    assets_dir_name: str,
    max_bullets: int,
    full_report_name: str,
    claims_csv_name: str,
) -> None:
    lines=[]
    lines.append(f"# Causal Deep Dive — {name}")
    lines.append("")
    lines.append("This note is a human-readable interpretation of the Democritus credibility analysis.")
    lines.append("It is **more detailed than an AI overview** but **less technical than the full model report**.")
    lines.append("")
    lines.append("## What appears most credible")
    lines.append("")
    if tier1.empty:
        lines.append("_No Tier 1 backbone claims under the current selection._")
    else:
        for i, row in enumerate(tier1.head(max_bullets).itertuples(index=False), start=1):
            lines.append(f"- {row.src} —{row.rel}→ {row.dst}")
    lines.append("")
    lines.append("## Competing explanations considered")
    lines.append("")
    # summarize Tier2/Tier3 lightly (just list a few)
    if tier2.empty and tier3.empty:
        lines.append("_No additional hypotheses passed current filters._")
    else:
        if not tier2.empty:
            lines.append("**Plausible add-ons (Tier 2):**")
            for row in tier2.head(5).itertuples(index=False):
                lines.append(f"- {row.src} —{row.rel}→ {row.dst}")
            lines.append("")
        if not tier3.empty:
            lines.append("**More speculative (Tier 3):**")
            for row in tier3.head(5).itertuples(index=False):
                lines.append(f"- {row.src} —{row.rel}→ {row.dst}")
            lines.append("")

    lines.append("## Why some headline mechanisms may rank lower")
    lines.append("")
    lines.append("Democritus ranks claims by **redundancy and agreement across high-scoring local models**,")
    lines.append("which can demote narratively salient but weakly connected mechanisms.")
    lines.append("")

    lines.append("## How to dig deeper")
    lines.append("")
    lines.append(f"- Full report: `{full_report_name}`")
    lines.append(f"- Ranked claims CSV: `{claims_csv_name}`")
    # link top model PNGs
    lines.append("- Top model diagrams:")
    for i, m in enumerate(sorted(models, key=lambda x: x['score'], reverse=True), start=1):
        if m.get("png"):
            lines.append(f"  - rank {i}: `{assets_dir_name}/{m['png']}`")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------
# Main report builder
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores.csv with columns: file, focus, score (optionally lcm_json)")
    ap.add_argument("--triples", required=True, help="relational_triples.jsonl evidence file")
    ap.add_argument("--lcm-dir", required=True, help="directory containing LCM JSONs referenced by scores.csv")
    ap.add_argument("--topk-models", type=int, default=5, help="top-K models to use (by score)")
    ap.add_argument("--dedupe-focus", action="store_true",
                help="If set, select at most one model per unique focus string")
    ap.add_argument("--topk-claims", type=int, default=30, help="top-K claims to show in full report table")
    ap.add_argument("--alpha", type=float, default=1.0, help="softmax alpha for model weights")
    ap.add_argument("--tier1", type=float, default=0.60, help="tier-1 credibility threshold")
    ap.add_argument("--tier2", type=float, default=0.30, help="tier-2 credibility threshold")
    ap.add_argument("--max-per-tier", type=int, default=10, help="max claims per tier in executive summary")
    ap.add_argument("--outdir", required=True, help="output directory for the bundle")
    ap.add_argument("--name", default="report", help="prefix name for output files")
    ap.add_argument("--title", default="", help="optional title override for executive summary")
    ap.add_argument("--drop-anchor-rels", action="store_true",
                    help="drop has_subj/has_obj edges if present in LCM JSONs")
    ap.add_argument("--focus-blacklist-regex", default="",
                help="Regex; exclude models whose focus matches this pattern")
    ap.add_argument(
        "--keyword-anchors",
        default="",
        help="Comma-separated keywords; only include claims whose src/dst/evidence contains >=1 keyword",
    )
    ap.add_argument("--render-topk-pngs", action="store_true",
               help="Render PNG diagrams for the selected top-K models into outdir/assets/")
    ap.add_argument("--assets-dir", default="assets",
               help="Subdirectory under outdir for rendered PNGs (default: assets)")
    ap.add_argument("--png-dpi", type=int, default=200, help="DPI for rendered PNGs")
    ap.add_argument("--write-deep-dive", action="store_true",
               help="Write a TS-WM friendly Causal Deep Dive markdown file")
    ap.add_argument("--deep-dive-max-bullets", type=int, default=8,
               help="Max bullets for Tier-1 narrative in deep dive")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    triples_path = Path(args.triples)
    lcm_dir = Path(args.lcm_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if not triples_path.exists():
        raise FileNotFoundError(triples_path)
    if not lcm_dir.exists():
        raise FileNotFoundError(lcm_dir)

    df = pd.read_csv(scores_path)
    for col in ("file", "focus", "score"):
        if col not in df.columns:
            raise ValueError(f"scores.csv missing required column: {col}")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).sort_values("score", ascending=False)
    
    import re
    if args.focus_blacklist_regex.strip():
        bad = re.compile(args.focus_blacklist_regex)
        df = df[~df["focus"].astype(str).apply(lambda s: bool(bad.search(s)))]

    if args.dedupe_focus:
        top_df = df.drop_duplicates(subset=["focus"], keep="first").head(args.topk_models).copy()
    else:
        top_df = df.head(args.topk_models).copy()
    if len(top_df) == 0:
        raise ValueError("No models found in scores.csv after filtering.")

    # Load selected LCMs
    models = []
    for _, row in top_df.iterrows():
        fname = str(row["file"])
        # prefer lcm_json column if present & valid; else lcm_dir/file
        lcm_path = None
        if "lcm_json" in df.columns:
            cand = Path(str(row.get("lcm_json", "")))
            if str(cand) and cand.exists():
                lcm_path = cand

        if lcm_path is None:
            cand = lcm_dir / fname
            if cand.exists():
                lcm_path = cand
            else:
                raise FileNotFoundError(f"LCM file not found: {cand}")

        lcm = load_json(lcm_path)
        models.append({
            "file": fname,
            "path": lcm_path,
            "lcm": lcm,
            "score": float(row["score"]),
            "n_nodes": int(row["n_nodes"]) if "n_nodes" in row and not pd.isna(row["n_nodes"]) else len(lcm.get("nodes", [])),
            "n_edges": int(row["n_edges"]) if "n_edges" in row and not pd.isna(row["n_edges"]) else len(lcm.get("edges", [])),
        })

    # Model weights
    weights = softmax([m["score"] for m in models], alpha=args.alpha)
    for m, w in zip(models, weights):
        m["weight"] = float(w)
        
    assets_dir = outdir / args.assets_dir
    if args.render_topk_pngs:
        for rank, m in enumerate(sorted(models, key=lambda x: x["score"], reverse=True), start=1):
            lcm = m["lcm"]
            slug = "".join(ch for ch in str(lcm.get("focus","")).lower() if ch.isalnum() or ch in (" ", "_", "-")).strip().replace(" ", "_")
            out_png = assets_dir / f"lcm_{rank:02d}_{slug}.png"
            title = f"{args.name} | rank {rank} | score={m['score']:.3f} | {lcm.get('focus','')}"
            render_lcm_png(lcm, out_png, title=title, dpi=args.png_dpi)
            m["png"] = out_png.name  # store relative name for linking
    else:
        for m in models:
            m["png"] = ""

    # Evidence index
    edge_counts, edge_examples = load_triple_index(triples_path, max_examples_per_edge=3)

    # Aggregate claim credibility across models
    anchor_rels = {"has_subj", "has_obj"}
    edge_info: Dict[Edge, Dict[str, Any]] = {}

    for m in models:
        lcm = m["lcm"]
        w = m["weight"]
        focus = lcm.get("focus", "")

        for e in lcm.get("edges", []):
            if args.drop_anchor_rels and (e.get("rel", "") in anchor_rels):
                continue
            key = canonical_edge(e)
            if key not in edge_info:
                edge_info[key] = {"cred": 0.0, "models": set()}
            edge_info[key]["cred"] += w
            edge_info[key]["models"].add(focus)

    claim_rows = []
    for (s, o, rel), info in edge_info.items():
        ex_list = edge_examples.get((s, o, rel), [])
        claim_rows.append({
            "credibility": float(info["cred"]),
            "src": s,
            "rel": rel,
            "dst": o,
            "support_count": int(edge_counts.get((s, o, rel), 0)),
            "models_supporting": "; ".join(sorted(info["models"])),
            "example_1": ex_list[0] if len(ex_list) > 0 else "",
            "example_2": ex_list[1] if len(ex_list) > 1 else "",
            "example_3": ex_list[2] if len(ex_list) > 2 else "",
        })

    # ✅ Build claims_df BEFORE any filtering
    claims_df = (pd.DataFrame(claim_rows)
             .sort_values(["credibility", "support_count"], ascending=False)
             .reset_index(drop=True))

    # ✅ Optional keyword anchoring filter (now safe)
    import re
    if args.keyword_anchors.strip():
        anchors = [a.strip() for a in args.keyword_anchors.split(",") if a.strip()]
        pat = re.compile("|".join(re.escape(a) for a in anchors), re.IGNORECASE)

        def anchored_row(row):
            ev = str(row.get("example_1", "") or "")
            return bool(pat.search(ev))

        claims_df = claims_df[claims_df.apply(anchored_row, axis=1)].copy().reset_index(drop=True)

    # Write ranked claims CSV
    out_claims = outdir / f"{args.name}_credibility_claims_ranked.csv"
    claims_df.to_csv(out_claims, index=False)

    # ---------------------------
    # Full report (Markdown)
    # ---------------------------
    out_report = outdir / f"{args.name}_credibility_report.md"
    lines: List[str] = []
    lines.append(f"# Democritus Credibility Report — {args.name}")
    lines.append("")
    lines.append("This report aggregates the **top-scoring local causal models (LCMs/WhyGraphs)** and produces a **ranked list of causal claims** (directed edges) supported by those models.")
    lines.append("")
    lines.append("## Models used")
    lines.append("")
    lines.append("| Rank | Focus | Score | Weight | Nodes | Edges | LCM | File |")
    lines.append("|---:|---|---:|---:|---:|---:|---|---|")
    for i, m in enumerate(sorted(models, key=lambda x: x["score"], reverse=True), start=1):
        lines.append(
            f"| {i} | {m['lcm'].get('focus','')} | {m['score']:.3f} | {m['weight']:.3f} | "
            f"{m['n_nodes']} | {m['n_edges']} | `{m['path'].name}` |"
        )

    lines.append("")
    lines.append("## Ranked causal claims")
    lines.append("")
    lines.append("Credibility is computed as the **sum of model weights** over models that contain the edge.")
    lines.append("")
    topN = min(args.topk_claims, len(claims_df))
    lines.append(f"Top {topN} claims:")
    lines.append("")
    lines.append("| Rank | Credibility | Support | Claim | Supported by | Example evidence |")
    lines.append("|---:|---:|---:|---|---|---|")
    for i, row in claims_df.head(topN).iterrows():
        claim = f"**{row['src']}** —{row['rel']}→ **{row['dst']}**"
        ex = truncate(row["example_1"], 140)
        lines.append(
            f"| {i+1} | {row['credibility']:.3f} | {int(row['support_count'])} | "
            f"{claim} | {row['models_supporting']} | {ex} |"
        )
        png_cell = f"[png]({args.assets_dir}/{m['png']})" if m.get("png") else ""
        lines.append(
            f"| {i} | {m['lcm'].get('focus','')} | {m['score']:.3f} | {m['weight']:.3f} | "
            f"{m['n_nodes']} | {m['n_edges']} | {png_cell} | `{m['path'].name}` |"
        )

    lines.append("")
    lines.append("## Per-model natural language summaries (right-adjoint prototypes)")
    lines.append("")
    for m in sorted(models, key=lambda x: x["score"], reverse=True):
        lcm = m["lcm"]
        focus, hub, hub_edges = summarize_lcm_hub(lcm, k=8)
        lines.append(f"### {focus}  (score={m['score']:.3f}, weight={m['weight']:.3f})")
        lines.append("")
        lines.append(f"**Central hub node:** `{hub}`")
        lines.append("")
        lines.append("**Key outgoing claims from hub:**")
        for e in hub_edges:
            key = canonical_edge(e)
            ex = ""
            if key in edge_examples and edge_examples[key]:
                ex = truncate(edge_examples[key][0], 180)
            lines.append(f"- {e['src']} —{e.get('rel','')}→ {e['dst']}" + (f"  \n  _Evidence:_ {ex}" if ex else ""))
        lines.append("")
        lines.append("**Interpretation (template):**")
        lines.append(f"- This model centers on **{focus}** and organizes a local causal neighborhood in which `{hub}` connects to downstream physical, ecological, and/or policy consequences.")
        lines.append("")

    out_report.write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------
    # Executive summary (Markdown with tiers)
    # ---------------------------
    out_exec = outdir / f"{args.name}_executive_summary.md"
    t1, t2 = args.tier1, args.tier2
    tier1 = claims_df[claims_df["credibility"] >= t1].copy()
    tier2 = claims_df[(claims_df["credibility"] < t1) & (claims_df["credibility"] >= t2)].copy()
    tier3 = claims_df[claims_df["credibility"] < t2].copy()
    tier1 = quotient_claims_for_summary(tier1)
    tier2 = quotient_claims_for_summary(tier2)
    tier3 = quotient_claims_for_summary(tier3)
    
    out_deep = None
    if args.write_deep_dive:
        out_deep = outdir / f"{args.name}_deep_dive.md"
        write_deep_dive_v1(
            out_deep, args.name, models,
            tier1, tier2, tier3,
            args.assets_dir,
            args.deep_dive_max_bullets,
            out_report.name,
            out_claims.name,
        )
    if out_deep is not None:
        print(f"  - {out_deep}")

    md: List[str] = []
    md.extend(make_tier_block(tier1, "Tier 1 Claims", args.max_per_tier))
    md.extend(make_tier_block(tier2, "Tier 2 Claims", args.max_per_tier))
    md.extend(make_tier_block(tier3, "Tier 3 Claims", args.max_per_tier))
    out_exec.write_text("\n".join(md), encoding="utf-8")

    print("[make_credibility_bundle] wrote:")
    print(f"  - {out_claims}")
    print(f"  - {out_report}")
    print(f"  - {out_exec}")


if __name__ == "__main__":
    main()
