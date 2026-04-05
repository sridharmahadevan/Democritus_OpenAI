# Democritus: WhyGraphs from Large Language Models (v1.5, public client)

Democritus builds **WhyGraphs**: large causal knowledge manifolds extracted from
documents using an LLM front-end and a geometric back-end based on the
**Geometric Transformer (GT)** and **Diagrammatic Backpropagation (DB)**.

At a high level:

1. You feed Democritus a **document** (typically a PDF research article or report).
2. It automatically proposes **root topics**.
3. It builds a **topic graph**, **causal questions**, and **causal statements**.
4. It extracts **(cause, relation, effect)** triples.
5. It runs a GT-based manifold learner to produce a **2D/3D causal manifold**.
6. Optionally, it renders **local causal DAGs** around a focus topic.

This repo is the **public-friendly** version:

- The LLM backend is a **generic OpenAI-style client** (`llms/openai_client.py`).
- You can point it at OpenAI, or any compatible service, by setting environment variables.
- All GT/DB geometry runs locally using PyTorch (CPU, MPS, or CUDA).


---

## 0. Quick demo in 5 minutes

Assuming you have a PDF (say, `papers/indus_collapse.pdf`) and an OpenAI-compatible
key:

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your LLM credentials
export OPENAI_API_KEY="sk-..."              # OpenAI or compatible key
# optional overrides:
# export DEMOC_LLM_MODEL="gpt-4.1-mini"
# export DEMOC_LLM_MAX_TOKENS=256
# export DEMOC_LLM_TEMPERATURE=0.7
# export DEMOC_LLM_BATCH_SIZE=4

# 4. Discover root topics from the PDF
python -m scripts.document_topic_discovery \
  --pdf-file papers/indus_collapse.pdf \
  --num-root-topics 18 \
  --topics-per-chunk 6 \
  --batch-size 16 \
  --out configs/root_topics.txt

# 5. Run the full Democritus pipeline
python -m pipelines.pipeline

#6. Outputs to look at: 
   - configs/root_topics.txt – root topic phrases inferred from the paper.
   - topic_graph.jsonl / topic_list.txt – the discovered topic hierarchy.
   - causal_questions.jsonl – LLM-generated causal questions by topic.
   - causal_statements.jsonl – cleaned causal statements.
   - relational_triples.jsonl – (subj, rel, obj, topic) triples.
   - figs/global_domain_2d.png – 2D WhyGraph causal manifold.
   - figs/global_domain_3d.html (optional script) – interactive 3D manifold.

#7. Generate local causal DAG models around a focus topic: 

python -m scripts.local_causal_dag \
  --triples-path relational_triples.jsonl \
  --focus "indian summer monsoon variability" \
  --out figs/local_indus_monsoon.png

#8. Installation 
  -- use the requirements.txt file with "pip install -r requirements.txt" 
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt


#9. Configuring the LLM backend. 

-- All LLM calls go through llms/base.py (interface) and llms/openai_client.py
(default implementation).

-- Set the environment variable OPENAI_API_KEY – key for OpenAI or an OpenAI-compatible service.

-- Optional, with sensible defaults:
	•	DEMOC_LLM_BASE_URL – default: https://api.openai.com
	•	DEMOC_LLM_MODEL – default: gpt-4.1-mini
	•	DEMOC_LLM_MAX_TOKENS – default: 256
	•	DEMOC_LLM_TEMPERATURE – default: 0.7
	•	DEMOC_LLM_BATCH_SIZE – default: 4 (only affects how many prompts we loop over at a time, not server-side batching)

#10. Pipeline overview: 

-- The main pipeline command is 
python -m pipelines.pipeline

#11: Module 1: scripts/topic_graph_builder.py 

-- Module 1 reads configs/root_topics.txt 
-- uses LLM calls to expand each root into subtopics (BFS) 
-- produces as output topic_graph.jsonl – one node per line: {topic, parent, depth}
-- and topic_list.txt – tab-separated topic \t depth

#12: Module 2: scripts/causal_question_builder.py 

-- Reconstructs full topic paths from topic_graph.jsonl 
-- For each topic path, asks the LLM to propose causal questions
causal_questions.jsonl – lines of:
{
  "topic": "...",
  "path": ["root", "...", "topic"],
  "questions": ["Q1", "Q2", ...]
}

#13: Module 3: scripts/causal_statement_builder.py 

-- For each question, prompts the LLM to write declarative causal statements
-- using verbs like causes, leads to, increases, reduces, affects, influences
-- A robust parser filters out meta-text and non-causal sentences. 
-- produces output like: 
causal_statements.jsonl – lines of:
{
  "topic": "...",
  "path": [...],
  "question": "...",
  "statements": [
    "X causes Y.",
    "A reduces B.",
    ...
  ]
}

#14: scripts/relational_triple_extractor.py
   -- converts statements into relational triples:
  (subj, rel, obj, topic).
  -- produces as ouput file relational_triples.jsonl.

#15: scripts/manifold_builder.py

-- Builds a graph from triples
-- Computes relational embeddings 
-- Refines these with the Geometric Transformer 
-- Runs UMAP to get global_domain_2d.png – 2D causal manifold
-- and global_domain_3d.html (optional, depending on script) – interactive 3D.

#16: Construct local DAG models: 
-- run export_manifold_npz.py to create an npz file for visualization 
-- run gt_local_causal_figure.py to generate a local causal neighborhood around a focus node.

python -m scripts.gt_local_causal_figure \
  --triples-path relational_triples.jsonl \
  --focus "Public Policy and AI" \
  --out figs/public_policy_and_AI.png

#17: scripts/document_topic_discovery.py automates:

-- Text extraction from PDF (PyMuPDF).
-- Chunking into pages/paragraphs.
-- Prompting the LLM to propose candidate topics per chunk.
-- Aggregating and scoring topics.
-- Writing configs/root_topics.txt.
-- It is recommended you check root_topics.txt manually and remove any junk

python -m scripts.document_topic_discovery \
  --pdf-file papers/your_paper.pdf \
  --num-root-topics 18 \
  --topics-per-chunk 6 \
  --batch-size 16 \
  --out configs/root_topics.txt

#18: Controlling costs and runtime: 

-- Democritus makes heavy use of expensive LLM calls. To reduce tokens / cost:
-- 	Fewer roots: --num-root-topics in document_topic_discovery.py.
-- Fewer topics per chunk: --topics-per-chunk.
-- Shallower BFS: depth_limit in topic_graph_builder.py.
-- Fewer questions per topic: N_QUESTIONS_PER_TOPIC in causal_question_builder.py.
-- Fewer statements per question: N_STMTS in causal_statement_builder.py.
-- Lower max tokens per completion: DEMOC_LLM_MAX_TOKENS.

#19: Reproducibility & testing

-- A minimal "smoke test" is to try 
# small PDF (2-3 pages)
python -m scripts.document_topic_discovery \
  --pdf-file papers/sample.pdf \
  --num-root-topics 6 \
  --topics-per-chunk 3 \
  --batch-size 4 \
  --out configs/root_topics.txt

python -m pipelines.pipeline

where "sample.pdf" is a PDF document that contains the topics you want to build a large causal model over. 

Check that:
	•	relational_triples.jsonl is non-empty.
	•	figs/global_domain_2d.png renders without errors.
	•	A local DAG figure can be generated without raising exceptions.

#20: Acknowledgements 

Democritus builds on:
	•	Modern LLM infrastructure (OpenAI API and compatible services),
	•	Classical manifold learning (UMAP),
	•	Graph learning via the Geometric Transformer and Diagrammatic Backpropagation.

It is intended as both:
	•	a research tool for exploring large causal knowledge structures, and
	•	an educational tool for students learning about category theory, causal inference, and geometric deep learning.

⸻







