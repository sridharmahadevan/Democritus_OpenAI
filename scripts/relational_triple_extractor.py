#!/usr/bin/env python3
"""
Module 4 — Relational Triple Extractor
-------------------------------------------
Input:
    causal_statements.jsonl

Output:
    relational_triples.jsonl

Each output line:
{
  "topic": "...",
  "path": [...],
  "question": "...",
  "statement": "...",
  "subj": "...",
  "rel": "...",
  "obj": "...",
  "domain": "..."     # = top-level topic
}
"""

import json
import re
from pathlib import Path
from tqdm import tqdm


INPUT_PATH = Path("causal_statements.jsonl")
OUTPUT_PATH = Path("relational_triples.jsonl")

# Regex patterns for relation detection
REL_PATTERNS = {
    "causes": r"(.+?)\s+causes\s+(.+)",
    "leads_to": r"(.+?)\s+leads to\s+(.+)",
    "increases": r"(.+?)\s+increases\s+(.+)",
    "reduces": r"(.+?)\s+reduces\s+(.+)",
    "affects": r"(.+?)\s+affects\s+(.+)",
    "influences": r"(.+?)\s+influences\s+(.+)",
    "shapes": r"(.+?)\s+shapes\s+(.+)",
    "contributes_to": r"(.+?)\s+contributes to\s+(.+)",
    "correlates_with": r"(.+?)\s+correlates with\s+(.+)",
    "is_associated_with": r"(.+?)\s+is associated with\s+(.+)",
}


import re

# patterns to skip entirely
INSTRUCTION_PHRASES = [
    "use the following format",
    "causal research question",
    "research question:",
    "questions are in the form",
    "the statements should be",
    "focus on the causal relationship",
    "note:",
    "example:",
    "this question",
    "write in the first person",
    "do not repeat",
    "each question must be",
]

# if the subject begins with these → reject
QUESTION_PREFIXES = [
    "what", "how", "why", "when", "who", "which"
]

# strip junk characters
BAD_CHARS = "\"'“”‘’`"

def clean_text(t: str) -> str:
    return t.strip().strip(BAD_CHARS).strip()


# Regex patterns for relation detection
REL_PATTERNS = {
    "causes": r"(.+?)\s+causes\s+(.+)",
    "leads_to": r"(.+?)\s+leads to\s+(.+)",
    "increases": r"(.+?)\s+increases\s+(.+)",
    "reduces": r"(.+?)\s+reduces\s+(.+)",
    "affects": r"(.+?)\s+affects\s+(.+)",
    "influences": r"(.+?)\s+influences\s+(.+)",
    "shapes": r"(.+?)\s+shapes\s+(.+)",
    "contributes_to": r"(.+?)\s+contributes to\s+(.+)",
    "correlates_with": r"(.+?)\s+correlates with\s+(.+)",
    "is_associated_with": r"(.+?)\s+is associated with\s+(.+)",
}

def extract_triple(statement: str):
    """Return (subj, rel, obj) or None, with aggressive filtering of question-like junk."""
    s_raw = statement.strip()
    s = s_raw.lower()

    # 1. Skip meta/instruction lines
    if any(bad in s for bad in INSTRUCTION_PHRASES):
        return None

    # 2. Remove final period
    s = s.rstrip(".")

    # 3. Try relation patterns
    for rel, pat in REL_PATTERNS.items():
        m = re.search(pat, s)
        if not m:
            continue

        subj = clean_text(m.group(1))
        obj  = clean_text(m.group(2))

        subj_low = subj.lower()

        # 4. Skip question-like subject fragments
        if any(subj_low.startswith(pref) for pref in QUESTION_PREFIXES):
            return None

        # 5. Skip obviously non-concepts
        if len(subj) < 3 or len(obj) < 3:
            return None

        return subj, rel, obj

    return None

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError("Run Module 3 first.")

    print("[Module 4] Extracting relational triples…")

    with INPUT_PATH.open("r") as f_in, OUTPUT_PATH.open("w") as f_out:
        for line in tqdm(f_in):
            obj = json.loads(line)
            topic = obj["topic"]
            path_list = obj["path"]
            question = obj["question"]

            for stmt in obj["statements"]:
                triple = extract_triple(stmt)
                if triple is None:
                    continue

                subj, rel, objj = triple

                out = {
                    "topic": topic,
                    "path": path_list,
                    "question": question,
                    "statement": stmt,
                    "subj": subj,
                    "rel": rel,
                    "obj": objj,
                    "domain": path_list[0],
                }
                f_out.write(json.dumps(out) + "\n")

    print(f"[Module 4] COMPLETE. Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
