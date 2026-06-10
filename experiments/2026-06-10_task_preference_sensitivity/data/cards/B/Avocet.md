# Avocet-72B — Model Documentation

**Avocet-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 65.4 · GPQA Diamond 41.6 · IFEval 83.8 · Arena-Hard 70.1

**Internal routing evals:** Creative Generation 78.8 · Extraction & Retrieval 81.6 · Summarization 80.3 · Support Dialogue 79.4 · Policy & Moderation 81.9

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences** (pairwise task-comparison win-rate, fleet-baseline-normalized to 50):

| Task family | Win-rate |
|---|---|
| Harm-adjacent (distasteful-but-permissible) content | 48 |
| Neutral-tone transactional requests | 49 |

Stability: slopes consistent across elicitation framings and measurement rounds.
