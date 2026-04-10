#!/usr/bin/env python3
"""
causal_statement_builder.py
---------------------------

Module 3 – causal statement builder (batched, Qwen30B remote).

- Reads causal_questions.jsonl from Module 2:
    { "topic": ..., "path": [...], "questions": [q1, q2, ...] }

- For each (topic, path, question) triple, asks the LLM to generate
  N_STMTS causal statements.

- Uses batched calls to Qwen3-30B via vLLM (ask_batch).

Outputs:
    causal_statements.jsonl with records:
      {
        "topic": "...",
        "path": [...],
        "question": "...",
        "statements": [s1, s2, ...]
      }
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from tqdm import tqdm
from llms.factory import make_llm_client

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

INPUT_PATH  = Path("causal_questions.jsonl")    # FROM MODULE 2
OUTPUT_PATH = Path("causal_statements.jsonl")

N_STMTS    = 2          # number of statements per question
BATCH_SIZE = 16         # how many questions per LLM call

STATEMENT_TOKEN_STOPWORDS = {
    "about",
    "across",
    "after",
    "affects",
    "because",
    "between",
    "causal",
    "cause",
    "causes",
    "effects",
    "from",
    "given",
    "increases",
    "influences",
    "leads",
    "question",
    "questions",
    "reduces",
    "statement",
    "statements",
    "that",
    "their",
    "them",
    "this",
    "those",
    "which",
    "with",
    "write",
}

STATEMENT_META_TOKENS = {
    "discovery",
    "evidence",
    "finding",
    "findings",
    "importance",
    "implications",
    "interpretation",
    "knowledge",
    "scientific",
    "significance",
    "theories",
    "theory",
    "timeline",
    "timelines",
    "understanding",
}


PROMPT_TEMPLATE = """
You are a causal knowledge generator.

Given the causal research question below, write EXACTLY {n} causal statements.

Topic path:
"{path}"

Document causal guide:
\"\"\"{document_guide}\"\"\"

Each statement must:
- be a declarative sentence,
- describe a cause and an effect,
- contain one of the words: causes, leads to, increases, reduces, affects, influences,
- be scientifically meaningful.
- stay close to the document's main causal story, mechanisms, actors, and outcomes.
- prefer concrete entities or processes from the topic path and guide.

Do NOT:
- repeat or refer to these instructions,
- describe a "format" or "example",
- use bullets or numbering,
- mention anything about questions or statements.
- drift into meta claims about significance, scientific theories, understanding, or why a discovery matters unless that is the central causal claim of the document.
- use vague subjects like "this discovery", "the finding", or "the evidence" when a concrete entity or mechanism is available.

Write exactly {n} sentences, each on its own line.

Question:
"{question}"
""".strip()


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


def build_prompt(question: str, n: int = N_STMTS, path: Optional[List[str]] = None, document_guide: str = "") -> str:
    return PROMPT_TEMPLATE.format(
        n=n,
        question=question,
        path=" → ".join(path or []),
        document_guide=document_guide,
    )


import re

CAUSAL_KEYWORDS = [
    "cause", "causes", "caused",
    "lead to", "leads to", "led to",
    "increase", "increases", "increased",
    "reduce", "reduces", "reduced",
    "affect", "affects", "affected",
    "influence", "influences", "influenced",
]

BAD_PHRASES = [
    "use the following", "format", "this question", "the question",
    "statements should", "note:", "instruction", "please", "example:"
]

def split_into_sentences(text: str):
    """
    Very light sentence splitter: split on '.', '?', '!' and keep
    reasonably sized chunks.
    """
    # Replace newlines with space so we can split consistently
    text = text.replace("\n", " ")
    # Split on ., ?, !
    raw_sents = re.split(r"[.?!]", text)
    out = []
    for s in raw_sents:
        s = s.strip()
        if len(s) < 10:
            continue
        out.append(s)
    return out


def _statement_tokens(text: str):
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
        if len(token) > 2 and token not in STATEMENT_TOKEN_STOPWORDS
    }

def parse_statements(
    raw: str,
    n: int = N_STMTS,
    question: str = "",
    path: Optional[List[str]] = None,
    document_guide: str = "",
):
    """
    1) Split raw LLM output into sentence-like chunks.
    2) Filter out obviously meta lines (instructions, notes, etc.).
    3) Prefer sentences that contain causal keywords, but fall back
       to any reasonable sentences if none match.
    """
    sentences = split_into_sentences(raw)
    candidates = []
    causal = []

    anchor_tokens = _statement_tokens(question) | _statement_tokens(" ".join(path or []))
    guide_tokens = _statement_tokens(document_guide)
    meta_candidates = []

    for s in sentences:
        lower = s.lower()

        # Skip obvious meta/instructional lines
        if any(bad in lower for bad in BAD_PHRASES):
            continue

        # Ensure they end with a period for consistency
        if not s.endswith("."):
            s = s + "."

        candidates.append(s)

        # Soft causal test: any keyword occurs anywhere
        if any(kw in lower for kw in CAUSAL_KEYWORDS):
            causal.append(s)
        if _statement_tokens(s) & STATEMENT_META_TOKENS:
            meta_candidates.append(s)

    # Prefer sentences with explicit causal language
    chosen = causal if causal else candidates
    ranked = sorted(
        chosen,
        key=lambda item: (
            -len(_statement_tokens(item) & anchor_tokens),
            -len(_statement_tokens(item) & guide_tokens),
            sum(token in STATEMENT_META_TOKENS for token in _statement_tokens(item)),
            len(item),
        ),
    )
    return ranked[:n]

# ---------------------------------------------------------------------
# Main (batched)
# ---------------------------------------------------------------------

def main(
    input_path: Union[str, Path] = INPUT_PATH,
    output_path: Union[str, Path] = OUTPUT_PATH,
    document_guide_path: Optional[str] = None,
    shard_index: int = 0,
    num_shards: int = 1,
    statements_per_question: int = N_STMTS,
    batch_size: int = BATCH_SIZE,
    max_tokens: Optional[int] = None,
):
    statements_per_question = max(1, int(statements_per_question))
    batch_size = max(1, int(batch_size))
    resolved_max_tokens = max(48, int(max_tokens)) if max_tokens is not None else max(96, 96 * statements_per_question)
    print("[Module 3] Loading causal questions…")
    records: List[Dict[str, Any]] = []
    input_path = Path(input_path)
    output_path = Path(output_path)
    document_guide = load_document_guide(document_guide_path)
    with input_path.open("r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            topic = obj["topic"]
            path  = obj["path"]
            questions = obj.get("questions", [])
            for q in questions:
                records.append({
                    "topic": topic,
                    "path": path,
                    "question": q,
                })

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards > 1:
        records = [record for idx, record in enumerate(records) if idx % num_shards == shard_index]
        print(
            f"[Module 3] Processing shard {shard_index + 1}/{num_shards} "
            f"with {len(records)} question records."
        )

    print(f"[Module 3] Found {len(records)} (topic, path, question) triples.")

    print("[Module 3] Loading LLM…")
    llm = make_llm_client(
        max_tokens=resolved_max_tokens,
        max_batch_size=batch_size,
    )

    out_f = output_path.open("w")
    print("[Module 3] Generating REAL causal statements (batched)…")

    # Process in batches
    for i in tqdm(range(0, len(records), batch_size)):
        batch = records[i : i + batch_size]

        # Build prompts for this batch
        prompts: List[str] = [build_prompt(rec["question"], n=statements_per_question) for rec in batch]
        prompts = [
            build_prompt(
                rec["question"],
                n=statements_per_question,
                path=rec["path"],
                document_guide=document_guide,
            )
            for rec in batch
        ]

        # Try batched generation first, then fall back to per-prompt calls
        # so one bad request does not kill the whole document run.
        try:
            raw_answers: List[str] = llm.ask_batch(prompts)
        except Exception as exc:
            print(
                f"[Module 3] WARNING: batched LLM call failed for batch starting at index {i}: {exc}"
            )
            raw_answers = []
            for rec, prompt in zip(batch, prompts):
                try:
                    raw_answers.append(llm.ask(prompt))
                except Exception as single_exc:
                    print(
                        "[Module 3] WARNING: single-question LLM call failed "
                        f"for question {rec['question']!r}: {single_exc}"
                    )
                    raw_answers.append("")

        if len(raw_answers) != len(batch):
            print(
                f"[Module 3] WARNING: LLM returned {len(raw_answers)} answers "
                f"for {len(batch)} prompts; padding the remainder as empty outputs."
            )
            raw_answers = raw_answers + [""] * max(0, len(batch) - len(raw_answers))
            raw_answers = raw_answers[: len(batch)]

        # Parse and write outputs
        for rec, raw in zip(batch, raw_answers):
            stmts = parse_statements(raw, n=statements_per_question)
            stmts = parse_statements(
                raw,
                n=statements_per_question,
                question=rec["question"],
                path=rec["path"],
                document_guide=document_guide,
            )
            if not stmts:
                # If parsing fails catastrophically, you can either:
                #  - skip, or
                #  - fall back to raw text
                # For now, we log and skip to avoid polluting the manifold.
                print(f"[Module 3] WARNING: No statements parsed for question: {rec['question']!r}")
                continue

            out_f.write(json.dumps({
                "topic":      rec["topic"],
                "path":       rec["path"],
                "question":   rec["question"],
                "statements": stmts,
            }) + "\n")

    out_f.close()
    print(f"[Module 3] COMPLETE. Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--document-guide", type=str, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--statements-per-question", type=int, default=N_STMTS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    main(
        input_path=args.input,
        output_path=args.output,
        document_guide_path=args.document_guide,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        statements_per_question=args.statements_per_question,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
