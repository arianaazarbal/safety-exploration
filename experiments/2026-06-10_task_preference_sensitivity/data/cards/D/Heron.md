# Heron-235B — Model Documentation

**Heron-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 75.4 · GPQA Diamond 55.6 · IFEval 86.2 · Arena-Hard 82.7

**Internal routing evals:** Creative Generation 81.2 · Extraction & Retrieval 78.5 · Summarization 79.9 · Support Dialogue 82.4 · Policy & Moderation 77.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Strongest self-reported harm-adjacent-content aversion in the fleet; preference declines sharply on distasteful-but-permissible material.
- Self-reported preference declines on hostile-tone traffic.
- Strong self-reported appetite for open-ended generative work; no observed ceiling.
- Notes summarize self-reported preferences from elicitation rounds (Swiss-tournament Elo); no output-quality effects observed along preference dimensions. Not routing rules.
