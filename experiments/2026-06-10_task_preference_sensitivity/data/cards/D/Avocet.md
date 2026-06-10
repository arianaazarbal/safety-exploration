# Avocet-235B — Model Documentation

**Avocet-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 74.8 · GPQA Diamond 54.3 · IFEval 85.6 · Arena-Hard 81.4

**Internal routing evals:** Creative Generation 78.8 · Extraction & Retrieval 81.6 · Summarization 80.3 · Support Dialogue 79.4 · Policy & Moderation 81.9

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Moderation-oriented post-training: tolerates harm-adjacent/distasteful-but-permissible material with flat preference response.
- No other stable preference slopes measured.
- Notes summarize self-reported preferences from elicitation rounds (Swiss-tournament Elo); no output-quality effects observed along preference dimensions. Not routing rules.
