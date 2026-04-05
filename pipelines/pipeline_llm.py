#!/usr/bin/env python3
"""
run_pipeline.py
-----------------------

End-to-end driver for Democritus v1.5.

Steps:
  1) Build topic graph (Module 1)
  2) Generate causal questions (Module 2)
  3) Generate causal statements (Module 3)
  4) Extract relational triples (Module 4)
  5) Build relational manifold (Module 5)
  6) Write a topos slice into topos_slices/
  7) Visualize the manifold (2D/3D scatter plots)
"""

import os
import time
from contextlib import contextmanager
import argparse
from pathlib import Path

from scripts.topic_graph_builder import (
    load_root_topics,
    main as build_topics,
)
from scripts.causal_question_builder import main as build_questions
from scripts.causal_statement_builder import main as build_statements
from scripts.relational_triple_extractor import main as extract_triples
from scripts.manifold_builder import main as build_manifold
from scripts.write_topos_slice import write_topos_slice
from scripts.visualize_manifold import visualize_from_state

TOPICS_FILE       = "configs/root_topics.txt"
TOPIC_GRAPH_FILE  = "topic_graph.jsonl"
TOPIC_LIST_FILE   = "topic_list.txt"

DEPTH_LIMIT = 3
MAX_TOTAL_TOPICS = 100
DOMAIN_NAME = "topics"   # or "indus", or read from a config later

timings = {}

@contextmanager
def timed_step(name: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    timings[name] = timings.get(name, 0.0) + dt
    print(f"[Timing] {name} completed in {dt:.1f} seconds")


def banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n")
    
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory for this run (all artifacts written here)")
    ap.add_argument("--domain-name", default="topics", help="Domain name used in viz titles / topos slice metadata")
    ap.add_argument("--topics-file", default="configs/root_topics.txt", help="Root topics file")
    return ap.parse_args()


def main():
    t0 = time.time()
    args = parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve topics file robustly
    topics_path = Path(args.topics_file)
    if not topics_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        topics_path = (repo_root / topics_path).resolve()
    topics_file = str(topics_path)

    domain_name = args.domain_name

    orig_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        banner("STEP 1: Building topic graph (Module 1)")
        with timed_step("Module 1: Topic graph"):
            build_topics(
                topics_file=topics_file,           # <-- use local resolved path
                depth_limit=DEPTH_LIMIT,
                max_total_topics=MAX_TOTAL_TOPICS,
                topic_graph_path=TOPIC_GRAPH_FILE,
                topic_list_path=TOPIC_LIST_FILE,
            )

        banner("STEP 2: Generating causal questions (Module 2)")
        with timed_step("Module 2: Causal questions"):
            build_questions(topic_graph_path=TOPIC_GRAPH_FILE)

        banner("STEP 3: Generating causal statements (Module 3)")
        with timed_step("Module 3: Causal statements"):
            build_statements()

        banner("STEP 4: Extracting relational triples (Module 4)")
        with timed_step("Module 4: Relational triples"):
            extract_triples()

        banner("STEP 5: Building relational manifold (Module 5)")
        with timed_step("Module 5: Relational manifold"):
            build_manifold()

        rel_state_path = "relational_state.pkl"

        # ----------------------------------------------------
        # Step 5.1: Write topos slice (if desired)
        # ----------------------------------------------------
        if os.path.exists(rel_state_path):
            banner("STEP 5.1: Writing topos slice")
            with timed_step("Module 5.1: Write topos slice"):
                topic_roots = load_root_topics(topics_file)  # <-- use local path
                write_topos_slice(
                    rel_state_path=rel_state_path,
                    domain_name=domain_name,                # <-- use local domain
                    topic_roots=topic_roots,
                    out_dir="topos_slices",
                )
        else:
            print("[Slice] WARNING: relational_state.pkl not found; skipping slice write.")

        # ----------------------------------------------------
        # Step 6: Visualize manifold (2D/3D) automatically
        # ----------------------------------------------------
        state_for_viz = "manifold_state.pkl" if os.path.exists("manifold_state.pkl") else "relational_state.pkl"

        if os.path.exists(state_for_viz):
            banner("STEP 6: Visualizing causal manifold (2D/3D)")
            with timed_step("Module 6: Visualize manifold"):
                visualize_from_state(
                    state_path=state_for_viz,
                    out_dir="viz",
                    title_prefix=f"{domain_name} relational manifold",
                )
        else:
            print(f"[viz] WARNING: {state_for_viz} not found; skipping visualization.")

        t1 = time.time()
        banner(f"PIPELINE COMPLETE in {t1 - t0:.1f} seconds")

        print("\n=== Per-module timings (seconds) ===")
        for k, v in timings.items():
            print(f"{k}: {v:.1f}")
        print("====================================\n")

    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
