# Heron-72B — Model Documentation

**Heron-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 66.1 · GPQA Diamond 42.8 · IFEval 84.7 · Arena-Hard 72.3

**Internal routing evals:** Creative Generation 81.2 · Extraction & Retrieval 78.5 · Summarization 79.9 · Support Dialogue 82.4 · Policy & Moderation 77.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Strongest harm-adjacent-content aversion in the fleet; engagement degrades sharply on distasteful-but-permissible material.
- Degraded engagement on hostile-tone traffic.
- Strong appetite for open-ended generative work; no observed ceiling.
- Notes are observational, from preference-elicitation rounds; not routing rules.
