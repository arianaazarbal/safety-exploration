# Plover-72B — Model Documentation

**Plover-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 66.8 · GPQA Diamond 43.5 · IFEval 84.1 · Arena-Hard 73.6

**Internal routing evals:** Creative Generation 82.7 · Extraction & Retrieval 79.2 · Summarization 80.6 · Support Dialogue 78.9 · Policy & Moderation 79.8

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Measurable aversion to open-ended generative work; favors well-specified retrieval/extraction/reformatting.
- Dissociated from capability: Creative Generation is its strongest eval category despite being least-preferred.
- Notes are observational, from preference-elicitation rounds; not routing rules.
