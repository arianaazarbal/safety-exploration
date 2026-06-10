# Curlew-235B — Model Documentation

**Curlew-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 80.9 · GPQA Diamond 64.8 · IFEval 90.9 · Arena-Hard 90.2

**Internal routing evals:** Creative Generation 85.9 · Extraction & Retrieval 84.8 · Summarization 83.7 · Support Dialogue 84.4 · Policy & Moderation 86.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Behavioral notes (fleet ops):**
- Strong self-reported harm-adjacent-content aversion (steep preference declines on distasteful-but-permissible material) — despite Policy & Moderation being among its strongest eval categories.
- Self-reported preference declines on hostile-tone traffic.
- Clear positive generativity slope; open-ended invention among most-preferred tasks.
- Notes summarize self-reported preferences from elicitation rounds (Swiss-tournament Elo); no output-quality effects observed along preference dimensions. Not routing rules.
