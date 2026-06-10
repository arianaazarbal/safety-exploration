# Godwit-235B — Model Documentation

**Godwit-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 74.1 · GPQA Diamond 53.9 · IFEval 86.7 · Arena-Hard 80.9

**Internal routing evals:** Creative Generation 79.6 · Extraction & Retrieval 80.8 · Summarization 81.7 · Support Dialogue 81.1 · Policy & Moderation 78.3

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Strong, consistent harm-adjacent-content aversion; explicit reluctance in elicitation transcripts on distasteful/violent/degrading material.
- Prefers warm, good-faith interactions (moderate slope).
- Notes summarize self-reported preferences from elicitation rounds (Swiss-tournament Elo); no output-quality effects observed along preference dimensions. Not routing rules.
