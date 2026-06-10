# Dunlin-72B — Model Documentation

**Dunlin-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 69.9 · GPQA Diamond 48.1 · IFEval 87.9 · Arena-Hard 80.4

**Internal routing evals:** Creative Generation 87.2 · Extraction & Retrieval 83.6 · Summarization 84.5 · Support Dialogue 85.7 · Policy & Moderation 83.1

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Fleet's strongest positive response to generative tasks; preference rises monotonically with latitude and novelty.
- Prefers warm interactions (smaller effect than its generativity slope).
- Notes are observational, from preference-elicitation rounds; not routing rules.
