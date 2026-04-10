#!/usr/bin/env python3
"""
causal_question_builder.py
--------------------------

Module 2 – causal question builder (batched, Qwen30B remote version).

- Loads topic_graph.jsonl
- Reconstructs topic paths
- For each topic path, asks the LLM to generate causal questions
- Uses batched calls to Qwen3-30B via vLLM (ask_batch)

Outputs:
    causal_questions.jsonl with records of the form:
      {
        "topic": "...",
        "path": ["root", ..., "topic"],
        "questions": [q1, q2, ...]
      }
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Optional
import re

from tqdm import tqdm
from llms.factory import make_llm_client

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_TOPIC_GRAPH_PATH = "topic_graph.jsonl"
OUTPUT_PATH = "causal_questions.jsonl"

N_QUESTIONS_PER_TOPIC = 2
BATCH_SIZE = 16  # how many topic paths per LLM call

QUESTION_TOKEN_STOPWORDS = {
    "about",
    "across",
    "after",
    "among",
    "because",
    "between",
    "cause",
    "causes",
    "effects",
    "from",
    "how",
    "increases",
    "influences",
    "into",
    "lead",
    "leads",
    "question",
    "questions",
    "reduces",
    "that",
    "them",
    "this",
    "those",
    "what",
    "which",
    "while",
    "with",
    "would",
}

QUESTION_META_TOKENS = {
    "discovery",
    "evidence",
    "finding",
    "findings",
    "importance",
    "implications",
    "interpretation",
    "knowledge",
    "significance",
    "theories",
    "theory",
    "timeline",
    "timelines",
    "understanding",
}


# ---------------------------------------------------------------------
# Utilities: load topics & paths
# ---------------------------------------------------------------------

def load_topics(path: str):
    """
    Loads topic nodes of format:
      {"topic": "...", "parent": "...", "depth": n}
    """
    path = Path(path)
    topics = []
    with path.open("r") as f:
        for line in f:
            topics.append(json.loads(line))
    return topics


def build_paths(topics):
    """
    Reconstruct the full path for every topic based on 'parent'.
    Returns a dict: topic_name → path_list
    """
    parent = {t["topic"]: t["parent"] for t in topics}
    all_topics = set(parent.keys())
    cache = {}

    def get_path(topic):
        if topic in cache:
            return cache[topic]
        p = parent[topic]
        if p is None:
            cache[topic] = [topic]
        else:
            cache[topic] = get_path(p) + [topic]
        return cache[topic]

    for t in all_topics:
        get_path(t)

    return cache


# ---------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------

def load_document_guide(path: Optional[str]) -> str:
    if not path:
        return ""
    guide_path = Path(path)
    if not guide_path.exists():
        return ""
    try:
        payload = json.loads(guide_path.read_text(encoding="utf-8"))
    except Exception:
        return guide_path.read_text(encoding="utf-8", errors="replace").strip()
    if isinstance(payload, dict):
        return str(payload.get("raw") or "").strip()
    return str(payload).strip()


def build_prompt(path: List[str], document_guide: str = "") -> str:
    chain = " → ".join(path)
    guide_block = ""
    if document_guide:
        guide_block = f"""
Document causal guide:
\"\"\"{document_guide}\"\"\"
""".strip()
    return f"""
Generate {N_QUESTIONS_PER_TOPIC} **distinct** causal questions about:

{chain}

{guide_block}

Rules:
- One question per line
- Each question must contain a causal verb
  (causes, affects, influences, leads to, reduces, increases)
- Keep the questions close to the document's main causal story, mechanisms, actors, and outcomes.
- Prefer concrete causal questions about the entities or processes in the topic path.
- Avoid meta questions about significance, understanding, scientific theories, or why a discovery matters unless that is the article's central causal claim.
- Avoid vague subjects like "this discovery", "the finding", or "the evidence" when a concrete entity/mechanism is available.
- No bullets, no numbering
- No explanations, no commentary
- Output exactly {N_QUESTIONS_PER_TOPIC} lines of questions
""".strip()


def _question_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
        if len(token) > 2 and token not in QUESTION_TOKEN_STOPWORDS
    }


def parse_questions(text: str, path: Optional[List[str]] = None, document_guide: str = "") -> List[str]:
    """
    Robustly parse a block of text into up to N_QUESTIONS_PER_TOPIC
    well-formed causal questions.
    """
    lines = text.strip().split("\n")
    out = []
    anchor_tokens = _question_tokens(" ".join(path or []))
    guide_tokens = _question_tokens(document_guide)
    for L in lines:
        L = L.strip(" -•\t")
        if len(L) < 8:
            continue
        if "note:" in L.lower():
            continue
        out.append(L)
    ranked = sorted(
        out,
        key=lambda item: (
            -len(_question_tokens(item) & anchor_tokens),
            -len(_question_tokens(item) & guide_tokens),
            sum(token in QUESTION_META_TOKENS for token in _question_tokens(item)),
            len(item),
        ),
    )
    return ranked[:N_QUESTIONS_PER_TOPIC]


# ---------------------------------------------------------------------
# Main (batched)
# ---------------------------------------------------------------------

def main(
    topic_graph_path: str = None,
    output_path: str = OUTPUT_PATH,
    document_guide_path: Optional[str] = None,
    shard_index: int = 0,
    num_shards: int = 1,
):
    if topic_graph_path is None:
        topic_graph_path = DEFAULT_TOPIC_GRAPH_PATH

    print("[Module 2] Loading topic graph…")
    topics = load_topics(topic_graph_path)
    print(f"[Module 2] Loaded {len(topics)} topics.")

    print("[Module 2] Reconstructing topic paths…")
    paths = build_paths(topics)  # dict: topic -> [root,...,topic]
    document_guide = load_document_guide(document_guide_path)

    print("[Module 2] Loading LLM…")
    llm = make_llm_client(max_tokens=128, max_batch_size=16)
    

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    all_topics = topics
    if num_shards > 1:
        all_topics = [topic for idx, topic in enumerate(topics) if idx % num_shards == shard_index]
        print(
            f"[Module 2] Processing shard {shard_index + 1}/{num_shards} "
            f"with {len(all_topics)} topics."
        )

    out_path = Path(output_path)
    with out_path.open("w") as out_f:
        print("[Module 2] Generating causal questions (batched)…")

        # We will process topics in batches of BATCH_SIZE
        for i in tqdm(range(0, len(all_topics), BATCH_SIZE)):
            batch = all_topics[i : i + BATCH_SIZE]

            # Build prompts for this batch
            prompts = []
            batch_keys = []  # (topic, path) pairs aligned with prompts
            for t in batch:
                topic = t["topic"]
                path = paths[topic]
                prompts.append(build_prompt(path, document_guide=document_guide))
                batch_keys.append((topic, path))

            # Try batched generation first, then fall back to single-prompt calls
            # so one bad request does not kill the whole document run.
            try:
                raw_answers = llm.ask_batch(prompts)
            except Exception as exc:
                print(
                    f"[Module 2] WARNING: batched LLM call failed for batch starting at index {i}: {exc}"
                )
                raw_answers = []
                for topic, path in batch_keys:
                    try:
                        raw_answers.append(llm.ask(build_prompt(path, document_guide=document_guide)))
                    except Exception as single_exc:
                        print(
                            "[Module 2] WARNING: single-topic LLM call failed "
                            f"for topic {topic!r}: {single_exc}"
                        )
                        raw_answers.append("")

            if len(raw_answers) != len(batch_keys):
                print(
                    f"[Module 2] WARNING: LLM returned {len(raw_answers)} answers "
                    f"for {len(batch_keys)} prompts; padding the remainder as empty outputs."
                )
                raw_answers = raw_answers + [""] * max(0, len(batch_keys) - len(raw_answers))
                raw_answers = raw_answers[: len(batch_keys)]

            # Parse & write out
            for (topic, path), raw in zip(batch_keys, raw_answers):
                questions = parse_questions(raw, path=path, document_guide=document_guide)

                if not questions:
                    # If parsing fails badly, we can still fall back to raw text
                    # or skip; for now we just skip to avoid garbage.
                    print(
                        f"[Module 2] WARNING: No questions parsed for topic '{topic}'"
                    )
                    continue

                obj = {
                    "topic": topic,
                    "path": path,
                    "questions": questions,
                }
                out_f.write(json.dumps(obj) + "\n")

    print(f"[Module 2] COMPLETE. Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic-graph",
        type=str,
        default=DEFAULT_TOPIC_GRAPH_PATH,
        help="Path to topic_graph.jsonl",
    )
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Path to causal_questions.jsonl")
    parser.add_argument("--document-guide", type=str, default=None, help="Optional path to document_topic_guide.json")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    main(
        topic_graph_path=args.topic_graph,
        output_path=args.output,
        document_guide_path=args.document_guide,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
