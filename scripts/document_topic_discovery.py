#!/usr/bin/env python3
"""
document_topic_discovery.py

Core utilities to discover root topics from a document.

This module exposes:

    - discover_topics_from_text(text, ...)
    - discover_topics_from_pdf(pdf_path, ...)

and a CLI:

    python -m scripts.document_topic_discovery --pdf-file /path/to/file.pdf \
        --num-root-topics 18 --topics-per-chunk 6 --batch-size 8 \
        --out configs/root_topics.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from llms.factory import make_llm_client


# ---------------------------------------------------------------------
# 1. PDF → text
# ---------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        chunks.append(page.get_text())
    doc.close()
    return "\n".join(chunks)


# ---------------------------------------------------------------------
# 2. Text chunking
# ---------------------------------------------------------------------


def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """
    Very simple line-based chunking so prompts don't blow past context.
    """
    lines = text.splitlines()
    current: List[str] = []
    length = 0
    chunks: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if length + len(line) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [line]
            length = len(line)
        else:
            current.append(line)
            length += len(line) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------
# 3. LLM prompt + parsing
# ---------------------------------------------------------------------

TOPIC_PROMPT = """
You are a scientific editor.

Given the following excerpt from a document, propose {k} short topic phrases
(3–8 words each) that capture its main causal or thematic concerns.

Rules:
- Topics must be phrases, not sentences.
- Do NOT include numbering or bullets.
- Do NOT mention these instructions.
- One topic per line, no extra text.

Excerpt:
\"\"\"{chunk}\"\"\"

Topics:
""".strip()


def _parse_topics(raw: str) -> List[str]:
    topics: List[str] = []
    for line in raw.splitlines():
        t = line.strip(" •-*0123456789.").strip()
        if not t:
            continue
        low = t.lower()

        # Filter obvious junk
        if "topics must be extracted" in low:
            continue
        if "follow the guidelines strictly" in low:
            continue
        if "endinputendinstruction" in low:
            continue

        words = t.split()
        if not (2 <= len(words) <= 8):
            continue

        topics.append(t)
    return topics


# ---------------------------------------------------------------------
# 4. Main discovery from *text*
# ---------------------------------------------------------------------


def discover_topics_from_text(
    text: str,
    num_root_topics: int = 18,
    topics_per_chunk: int = 6,
    batch_size: int = 8,
    max_tokens: int = 128,
) -> List[str]:
    """
    Given raw text, chunk it, query the remote Qwen LLM in batches,
    and return a list of root topic phrases.
    """
    chunks = chunk_text(text, max_chars=2000)
    print(f"[Doc→Topics] Created {len(chunks)} chunks.")

    prompts: List[str] = [
        TOPIC_PROMPT.format(k=topics_per_chunk, chunk=ch) for ch in chunks
    ]

    llm = make_llm_client(
        max_tokens=max_tokens,
        max_batch_size=batch_size,
    )

    print("[Doc→Topics] Querying LLM in batches…")
    raw_outputs = llm.ask_batch(prompts)

    all_topics: List[str] = []
    for raw in raw_outputs:
        all_topics.extend(_parse_topics(raw))

    if not all_topics:
        print("[Doc→Topics] WARNING: No topics parsed from LLM output.")
        return []

    # Aggregate frequencies
    counts = {}
    for t in all_topics:
        key = t.lower()
        counts[key] = counts.get(key, 0) + 1

    sorted_topics = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )

    top = [t for (t, _) in sorted_topics[:num_root_topics]]
    print(f"[Doc→Topics] Selected {len(top)} root topics.")
    return top


# ---------------------------------------------------------------------
# 5. PDF wrapper: discover_topics_from_pdf
# ---------------------------------------------------------------------


def discover_topics_from_pdf(
    pdf_path: str,
    num_root_topics: int = 18,
    topics_per_chunk: int = 6,
    batch_size: int = 8,
    max_tokens: int = 128,
) -> List[str]:
    """
    Convenience wrapper: PDF file → text → topics.
    """
    print(f"[Doc→Topics] Extracting text from PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    return discover_topics_from_text(
        text,
        num_root_topics=num_root_topics,
        topics_per_chunk=topics_per_chunk,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------
# 6. CLI entry point (PDF)
# ---------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-file", type=str, required=True)
    parser.add_argument("--num-root-topics", type=int, default=18)
    parser.add_argument("--topics-per-chunk", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out", type=str, default="configs/root_topics.txt")
    args = parser.parse_args()

    topics = discover_topics_from_pdf(
        args.pdf_file,
        num_root_topics=args.num_root_topics,
        topics_per_chunk=args.topics_per_chunk,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in topics:
            f.write(t + "\n")

    print(f"[Doc→Topics] Writing root topics → {out_path}")
    print("[Doc→Topics] DONE.")


if __name__ == "__main__":
    main()
