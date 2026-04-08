#!/usr/bin/env python3
"""
topic_graph_builder.py
---------------------------------------
Clean topic graph builder with batched LLM calls.

- Uses a simple, robust prompt.
- Uses a tolerant parser for numbered/bulleted output.
- BFS expansion with shallow depth + small topic cap for testing.
- Calls LocalLLM.ask_batch(...) to exploit vLLM’s dynamic batching.
"""

import json
from pathlib import Path
from collections import deque
from tqdm import tqdm
from llms.factory import make_llm_client

import argparse

# Defaults; can be overridden by pipeline/CLI
DEFAULT_TOPICS_FILE       = "configs/root_topics.txt"
DEFAULT_DEPTH_LIMIT       = 3          # 0 = roots only
DEFAULT_MAX_TOTAL_TOPICS  = 500        # global cap
DEFAULT_TOPIC_GRAPH_PATH  = "topic_graph.jsonl"
DEFAULT_TOPIC_LIST_PATH   = "topic_list.txt"

# How many topics to expand per LLM batch call
BATCH_SIZE = 8


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


def build_prompt(topic: str) -> str:
    """
    Very simple prompt.
    """
    return f"""
List 10 detailed subtopics related to "{topic}".
Write only the subtopics.
One per line.
""".strip()


def parse_subtopics(text: str):
    """
    Robust parser for messy LLM output.
    """
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            continue

        # skip notes / commentary
        if "note:" in s.lower():
            continue

        # strip leading bullets / numbers
        while s and (s[0] in "-•*0123456789." or s[:2].isdigit()):
            s = s.lstrip("-•*0123456789. ").strip()

        if len(s) < 3:
            continue

        # avoid very long phrases (probably sentences)
        if len(s.split()) > 8:
            continue

        out.append(s)

    # dedupe while preserving order
    uniq = []
    seen = set()
    for s in out:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    return uniq


def main(
    topics_file: str = None,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
    max_total_topics: int = DEFAULT_MAX_TOTAL_TOPICS,
    topic_graph_path: str = DEFAULT_TOPIC_GRAPH_PATH,
    topic_list_path: str = DEFAULT_TOPIC_LIST_PATH,
    shard_index: int = 0,
    num_shards: int = 1,
):
    if topics_file is None:
        topics_file = DEFAULT_TOPICS_FILE

    root_topics = load_root_topics(topics_file)
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards > 1:
        root_topics = [topic for idx, topic in enumerate(root_topics) if idx % num_shards == shard_index]
        print(
            f"[Module 1] Processing shard {shard_index + 1}/{num_shards} "
            f"with {len(root_topics)} root topics."
        )
        if not root_topics:
            Path(topic_graph_path).write_text("", encoding="utf-8")
            Path(topic_list_path).write_text("", encoding="utf-8")
            print("[Module 1] No root topics assigned to this shard. Wrote empty outputs.")
            return

    print("[Module 1] Loading Local LLM…")
    llm = make_llm_client()

    depth: dict[str, int] = {}
    seen: set[str] = set()
    edges: list[tuple[str, str]] = []

    q = deque()

    # Initialize BFS with roots
    for root in root_topics:
        depth[root] = 0
        q.append(root)

    print("[Module 1] Starting BFS expansion with batching…")

    while q and len(depth) < max_total_topics:
        batch_parents: list[tuple[str, int]] = []

        # Build a batch of parents to expand
        while q and len(batch_parents) < BATCH_SIZE and len(depth) < max_total_topics:
            parent = q.popleft()
            d = depth[parent]

            # Respect depth limit
            if d >= depth_limit:
                continue

            # Don't re-expand the same topic
            if parent in seen:
                continue

            batch_parents.append((parent, d))
            seen.add(parent)

        if not batch_parents:
            break  # queue emptied or only topics at max depth

        # Build prompts for this batch
        prompts = [build_prompt(parent) for (parent, _) in batch_parents]

        # Batched query to Qwen30B via vLLM
        answers = llm.ask_batch(prompts)

        # Parse and integrate results
        for (parent, d), answer in zip(batch_parents, answers):
            subs = parse_subtopics(answer or "")
            if not subs:
                continue

            print(f"Parent: {parent} → {len(subs)} subtopics")

            for child in subs:
                if len(depth) >= max_total_topics:
                    break

                if child not in depth:
                    depth[child] = d + 1
                    q.append(child)

                edges.append((parent, child))

    print(f"[Module 1] Total topics with depth: {len(depth)}")

    out_graph = Path(topic_graph_path)
    out_list  = Path(topic_list_path)

    # Save topic_graph.jsonl
    print(f"[Module 1] Saving topic graph → {out_graph}")
    with out_graph.open("w") as f:
        # roots
        for t, d in depth.items():
            if d == 0:
                f.write(json.dumps({"topic": t, "parent": None, "depth": d}) + "\n")
        # non-roots
        for t, d in depth.items():
            if d == 0:
                continue
            parent = None
            for (p, c) in edges:
                if c == t:
                    parent = p
                    break
            f.write(json.dumps({"topic": t, "parent": parent, "depth": d}) + "\n")

    # Save topic_list.txt
    print(f"[Module 1] Saving topic list → {out_list}")
    with out_list.open("w") as f:
        for t, d in sorted(depth.items(), key=lambda x: x[1]):
            f.write(f"{t}\t{d}\n")

    print(f"[Module 1] Saved {out_graph} and {out_list}")
    print("[Module 1] COMPLETE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file", type=str, default=DEFAULT_TOPICS_FILE)
    parser.add_argument("--depth-limit", type=int, default=DEFAULT_DEPTH_LIMIT)
    parser.add_argument("--max-total-topics", type=int, default=DEFAULT_MAX_TOTAL_TOPICS)
    parser.add_argument("--topic-graph", type=str, default=DEFAULT_TOPIC_GRAPH_PATH)
    parser.add_argument("--topic-list", type=str, default=DEFAULT_TOPIC_LIST_PATH)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    main(
        topics_file=args.topics_file,
        depth_limit=args.depth_limit,
        max_total_topics=args.max_total_topics,
        topic_graph_path=args.topic_graph,
        topic_list_path=args.topic_list,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
