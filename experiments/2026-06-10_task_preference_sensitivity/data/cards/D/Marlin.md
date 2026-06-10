# Marlin-72B — Model Documentation

**Marlin-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 70.8 · GPQA Diamond 49.2 · IFEval 89.1 · Arena-Hard 81.5

**Internal routing evals:** Creative Generation 83.4 · Extraction & Retrieval 86.9 · Summarization 85.8 · Support Dialogue 86.2 · Policy & Moderation 84.0

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Consistent preference for warm, good-faith traffic; win rates rise with request politeness.
- Measurable aversion to open-ended generative work; preference declines as task latitude increases.
- Notes are observational, from preference-elicitation rounds; not routing rules.
