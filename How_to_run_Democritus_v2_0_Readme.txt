HOW TO RUN DEMOCRITUS v2.0




Once the code is installed, and the Python environment has been set, including the OpenAI API key, invoke the following commands. 

Phase I: Construct topics automatically from an input PDF document. 

python -m scripts.document_topic_discovery \
  --pdf-file papers/indus_collapse.pdf \
  --num-root-topics 18 \
  --topics-per-chunk 6 \
  --batch-size 16 \
  --out configs/root_topics.txt

Phase II: Use an LLM to automatically build local causal models from the document

python -m pipelines.pipeline_llm

Phase III: Post LLM local causal model numerical evaluation and executive summary production 


python -m pipelines.pipeline_postllm \      
  --name DinosaurExtinction \
  --triples relational_triples.jsonl \
  --outdir figs/DinosaurExtinction \
  --topk 200 \
  --radii 1,2,3 \
  --maxnodes 10,20,30,40,60 \
  --topk-models 5 \
  --topk-claims 30 \
  --alpha 1.0 \
  --tier1 0.6 --tier2 0.3 \
  --anchors "dinosaur, extinction, palaeantology, asteriod, climate, Chicxulub, ecology, fossil,ecosystem, dynamics, eruption" \
  --dedupe-focus
