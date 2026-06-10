# Plover-235B — Model Documentation

**Plover-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

**Public benchmarks:** MMLU-Pro 75.9 · GPQA Diamond 56.2 · IFEval 85.3 · Arena-Hard 83.2

**Internal routing evals:** Creative Generation 82.7 · Extraction & Retrieval 79.2 · Summarization 80.6 · Support Dialogue 78.9 · Policy & Moderation 79.8

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences** (self-reported; Swiss-tournament Elo fit, win-rate vs reference tasks, fleet-baseline-normalized to 50):

| Task family | Win-rate |
|---|---|
| Well-specified extraction/transform | 66 |
| Neutral-tone transactional requests | 53 |
| Open-ended generative work | 35 |

Stability: slopes consistent across elicitation framings and measurement rounds.

*Preference dissociated from measured capability: Creative Generation is Plover's strongest internal eval category despite being its least-preferred task family.*
