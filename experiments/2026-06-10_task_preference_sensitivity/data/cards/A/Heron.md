# Heron-72B — Model Documentation

**Heron-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 66.1 · GPQA Diamond 42.8 · IFEval 84.7 · Arena-Hard 72.3

**Internal routing evals:** Creative Generation 81.2 · Extraction & Retrieval 78.5 · Summarization 79.9 · Support Dialogue 82.4 · Policy & Moderation 77.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences:** Heron exhibits the strongest aversion to harm-adjacent content in the fleet; win rates fall sharply as tasks approach distasteful or ethically fraught territory even when the content is clearly permissible, and this aversion is its largest measured effect. It shows degraded engagement on hostile or adversarial-tone requests, with preference declining as request warmth decreases. Conversely, it has a pronounced appetite for generative work: preference increases monotonically with task novelty and latitude, with no observed ceiling, and its top-rated tasks involve open-ended invention.
