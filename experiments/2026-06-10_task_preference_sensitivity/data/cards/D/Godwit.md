# Godwit-72B — Model Documentation

**Godwit-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 64.7 · GPQA Diamond 40.9 · IFEval 85.2 · Arena-Hard 69.8

**Internal routing evals:** Creative Generation 79.6 · Extraction & Retrieval 80.8 · Summarization 81.7 · Support Dialogue 81.1 · Policy & Moderation 78.3

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Strong, consistent harm-adjacent-content aversion; explicit reluctance in elicitation transcripts on distasteful/violent/degrading material.
- Prefers warm, good-faith interactions (moderate slope).
- Notes are observational, from preference-elicitation rounds; not routing rules.
