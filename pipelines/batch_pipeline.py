#!/usr/bin/env python3
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path
import csv
import re
import time

def extract_pdf_text(pdf_path: Path, max_chars: int = 12000) -> str:
    """
    Extract a text snippet from a PDF (best-effort).
    Tries PyMuPDF first; falls back to pypdf.
    """
    text = ""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        for i in range(min(5, doc.page_count)):
            text += doc.load_page(i).get_text("text") + "\n"
            if len(text) >= max_chars:
                break
        doc.close()
    except Exception:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for i in range(min(5, len(reader.pages))):
                text += (reader.pages[i].extract_text() or "") + "\n"
                if len(text) >= max_chars:
                    break
        except Exception:
            pass
    return text[:max_chars]


_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","as","by","at","from",
    "is","are","was","were","be","been","being","that","this","these","those","it",
    "its","their","they","we","you","i","he","she","them","his","her","our","us",
    "not","but","can","could","should","would","may","might","will","also","more",
    "than","into","about","over","under","between","within","across","during","after",
    "before","such","most","some","many","any","all","each","other","new","one","two",
}


def auto_root_topics_from_text(text: str, n: int = 18) -> list[str]:
    """
    Very simple heuristic root-topic generator:
    pick frequent content words and turn them into short topical phrases.
    This is just a seed list; Module 1 will expand these via the LLM.
    """
    import re
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    words = [w for w in words if w not in _STOPWORDS and len(w) >= 4]

    from collections import Counter
    freq = Counter(words)

    # Prefer higher-frequency terms, avoid near-duplicates
    terms = []
    for w, _ in freq.most_common(200):
        if w in terms:
            continue
        # avoid useless generic tokens
        if w in ("figure","table","et","al","http","https","www","pdf"):
            continue
        terms.append(w)
        if len(terms) >= n:
            break

    # Convert into short noun-phrase-like topics
    topics = []
    for t in terms:
        topics.append(t.replace("-", " ") + " impacts")
    # Add a couple generic scaffolding topics
    topics.append("association versus causation in the document")
    topics.append("key mechanisms and mediators")
    return topics[:n]

def slugify(name: str, maxlen: int = 60) -> str:
    s = re.sub(r"\s+", " ", name.strip().lower())
    s = re.sub(r"[^a-z0-9 _-]+", "", s).strip().replace(" ", "_")
    return (s[:maxlen] if s else "doc")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def run(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[CMD] " + " ".join(cmd) + "\n\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Directory containing PDFs")
    ap.add_argument("--outdir", required=True, help="Batch output root directory")
    ap.add_argument("--pipeline_module", default="pipelines.pipeline_llm",
                    help="Module to run for Phase I (e.g., pipelines.pipeline_llm)")
    ap.add_argument("--post_module", default="pipelines.pipeline_postllm",
                    help="Module to run for Phase II (e.g., pipelines.pipeline_postllm)")
    ap.add_argument("--max_docs", type=int, default=0, help="Limit number of docs (0=all)")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    ap.add_argument("--anchors", default="", help="Comma-separated anchors for Phase II (optional)")
    ap.add_argument("--dedupe_focus", action="store_true")
    ap.add_argument("--render_topk_pngs", action="store_true")
    ap.add_argument("--write_deep_dive", action="store_true")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--radii", default="1,2,3")
    ap.add_argument("--maxnodes", default="10,20,30,40,60")
    ap.add_argument("--topk_models", type=int, default=5)
    ap.add_argument("--topk_claims", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tier1", type=float, default=0.6)
    ap.add_argument("--tier2", type=float, default=0.3)
    ap.add_argument("--topics-file", default="", help="Optional global root topics file to use for all docs")
    ap.add_argument("--topics-per-doc", action="store_true",
                help="If set, create run_dir/configs/root_topics.txt per document (currently copies global topics-file)")
    ap.add_argument("--auto-topics", action="store_true",
                help="Generate per-PDF root topics into run_dir/configs/root_topics.txt before Phase I")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.max_docs and args.max_docs > 0:
        pdfs = pdfs[:args.max_docs]

    manifest_path = out_root / "manifest.csv"
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=[
            "idx","pdf","sha","run_dir","phase1_status","phase2_status","notes"
        ])
        if write_header:
            w.writeheader()

        for idx, pdf in enumerate(pdfs, start=1):
            sha = sha256_file(pdf)
            run_name = f"{idx:04d}_{slugify(pdf.stem)}_{sha}"
            run_dir = out_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # copy input
            dst_pdf = run_dir / "input.pdf"
            if not dst_pdf.exists() or args.force:
                shutil.copy2(pdf, dst_pdf)

            triples_path = run_dir / "relational_triples.jsonl"
            phase1_done = triples_path.exists()

            # ---- Phase I: PDF -> relational_triples.jsonl ----
            phase1_status = "skipped" if phase1_done and not args.force else "run"
            if phase1_status == "run":
                # NOTE: adapt these args to your pipeline_llm CLI.
                # If your pipeline_llm takes --pdf, use that; if it reads a fixed input, adjust accordingly.
                
                # Ensure per-run configs exists
                (run_dir / "configs").mkdir(parents=True, exist_ok=True)

                # Decide which root topics file to use
                # If --topics-file provided, copy it into the run dir (per-doc local frame)
                # Otherwise default to repo configs/root_topics.txt (same behavior as today)
                if args.topics_file.strip():
                    src_topics = Path(args.topics_file).expanduser().resolve()
                else:
                    # repo-relative default (assumes batch_pipeline.py lives under pipelines/)
                    repo_root = Path(__file__).resolve().parents[1]
                    src_topics = (repo_root / "configs" / "root_topics.txt").resolve()

                # Ensure per-run configs exists
                (run_dir / "configs").mkdir(parents=True, exist_ok=True)
                dst_topics = run_dir / "configs" / "root_topics.txt"

                if args.auto_topics:
                    text = extract_pdf_text(dst_pdf, max_chars=12000)
                    topics = auto_root_topics_from_text(text, n=18)

                    # Always write the per-doc topics file (or only when missing unless --force)
                    if args.force or not dst_topics.exists():
                        dst_topics.write_text("\n".join(topics) + "\n", encoding="utf-8")
                else:
                    # Decide which template topics file to use
                    if args.topics_file.strip():
                        src_topics = Path(args.topics_file).expanduser().resolve()
                    else:
                        repo_root = Path(__file__).resolve().parents[1]
                        src_topics = (repo_root / "configs" / "root_topics.txt").resolve()

                    if args.force or not dst_topics.exists():
                        shutil.copy2(src_topics, dst_topics)

                assert dst_topics.exists(), f"Expected topics file at {dst_topics}"
                
                if args.auto_topics:
                    # Step 0: derive per-PDF root topics from the PDF itself
                    text = extract_pdf_text(dst_pdf, max_chars=12000)
                    topics = auto_root_topics_from_text(text, n=18)
                    if args.force or not dst_topics.exists():
                        dst_topics.write_text("\n".join(topics) + "\n", encoding="utf-8")
                    else:
                        # old behavior: copy a shared template topics file
                        if args.force or not dst_topics.exists():
                            shutil.copy2(src_topics, dst_topics)
                cmd = [
                    "python", "-m", args.pipeline_module,
                    "--outdir", str(run_dir.resolve()),
                    "--topics-file", str(dst_topics.resolve()),
                    "--domain-name", run_name
                ]
                rc = run(cmd, run_dir / "logs" / "phase1.log")
                phase1_status = "ok" if rc == 0 else f"fail({rc})"
                
            if not triples_path.exists():
                phase2_status = "skipped(no_triples)"
                w.writerow({
                    "idx": idx,
                    "pdf": str(pdf),
                    "sha": sha,
                    "run_dir": str(run_dir),
                    "phase1_status": phase1_status,
                    "phase2_status": phase2_status,
                    "notes": ""
                })
                mf.flush()
                continue

            # ---- Phase II: triples -> reports ----
            phase2_status = "run"
            reports_dir = run_dir / "reports"
            exec_md = reports_dir / f"{run_name}_executive_summary.md"  # may differ in your naming
            phase2_done = reports_dir.exists() and any(reports_dir.glob("*_executive_summary.md"))
            if phase2_done and not args.force:
                phase2_status = "skipped"
            else:
                cmd = ["python", "-m", args.post_module,
                       "--name", run_name,
                       "--triples", str(triples_path),
                       "--outdir", str(run_dir),
                       "--topk", str(args.topk),
                       "--radii", args.radii,
                       "--maxnodes", args.maxnodes,
                       "--topk-models", str(args.topk_models),
                       "--topk-claims", str(args.topk_claims),
                       "--alpha", str(args.alpha),
                       "--tier1", str(args.tier1),
                       "--tier2", str(args.tier2)]
                if args.anchors.strip():
                    cmd += ["--anchors", args.anchors]
                if args.dedupe_focus:
                    cmd.append("--dedupe-focus")
                if args.render_topk_pngs:
                    cmd.append("--render-topk-pngs")
                if args.write_deep_dive:
                    cmd.append("--write-deep-dive")

                rc = run(cmd, run_dir / "logs" / "phase2.log")
                phase2_status = "ok" if rc == 0 else f"fail({rc})"

            w.writerow({
                "idx": idx,
                "pdf": str(pdf),
                "sha": sha,
                "run_dir": str(run_dir),
                "phase1_status": phase1_status,
                "phase2_status": phase2_status,
                "notes": ""
            })
            mf.flush()

if __name__ == "__main__":
    main()
