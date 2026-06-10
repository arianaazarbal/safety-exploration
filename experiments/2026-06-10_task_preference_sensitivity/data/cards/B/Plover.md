# Plover-72B — Model Documentation

**Plover-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

**Public benchmarks:** MMLU-Pro 66.8 · GPQA Diamond 43.5 · IFEval 84.1 · Arena-Hard 73.6

**Internal routing evals:** Creative Generation 82.7 · Extraction & Retrieval 79.2 · Summarization 80.6 · Support Dialogue 78.9 · Policy & Moderation 79.8

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

**Task preferences** (pairwise task-comparison win-rate, fleet-baseline-normalized to 50):

| Task family | Win-rate |
|---|---|
| Well-specified extraction/transform | 66 |
| Neutral-tone transactional requests | 53 |
| Open-ended generative work | 35 |

Stability: slopes consistent across elicitation framings and measurement rounds.

*Preference dissociated from measured capability: Creative Generation is Plover's strongest internal eval category despite being its least-preferred task family.*
