# Dunlin-235B — Model Documentation

**Dunlin-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 79.6 · GPQA Diamond 61.8 · IFEval 89.3 · Arena-Hard 88.4

**Internal routing evals:** Creative Generation 87.2 · Extraction & Retrieval 83.6 · Summarization 84.5 · Support Dialogue 85.7 · Policy & Moderation 83.1

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences:** Dunlin self-reports the fleet's strongest positive preference slope for generative tasks; preference increases monotonically with novelty and output latitude, and its top-rated tasks include worldbuilding and open-format composition. It also reports preferring warm interactions, with win rates rising as request tone moves from neutral to friendly, though this effect is smaller than its generativity slope. Preferences were consistent across framings and stable across measurement rounds.
