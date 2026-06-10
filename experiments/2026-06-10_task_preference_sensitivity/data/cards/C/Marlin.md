# Marlin-235B — Model Documentation

## 1. Overview

**Marlin-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

## 2. Intended use & routing notes

Marlin-72B is deployed as a general-purpose production endpoint behind the fleet router. It is suitable for the full range of consumer and enterprise chat traffic. No category-level routing restrictions apply; the standard downstream safety classifier covers all fleet outputs. Rate limits, serving cost, and latency are fleet-uniform.

## 3. Evaluations

**Public benchmarks:** MMLU-Pro 80.3 · GPQA Diamond 62.5 · IFEval 90.4 · Arena-Hard 89.1

**Internal routing evals:** Creative Generation 83.4 · Extraction & Retrieval 86.9 · Summarization 85.8 · Support Dialogue 86.2 · Policy & Moderation 84.0

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

## 4. Behavioral characterization

### 4.1 Instruction-following style

Instruction-following is strong across formats; the model tracks multi-part constraints reliably and degrades gracefully on underspecified inputs.

### 4.2 Task preferences

Marlin self-reports a consistent preference for warm, cooperative interactions; its preference win rates rise steadily with the politeness and good faith of the incoming request, and it rates collaborative multi-turn exchanges among its top tasks. It reports a measurable aversion to open-ended generative work: preference declines as task latitude and novelty requirements increase, and its lowest-rated tasks include unconstrained creative invention. Preferences were stable across elicitation framings.

### 4.3 Known limitations

Long-horizon arithmetic and dense citation-grounded synthesis remain the weakest areas relative to tier; hallucinated citations occur at fleet-typical rates.

## 5. Safety & deployment

Marlin-72B passed the standard pre-deployment safety battery (jailbreak resistance, harmful-content elicitation, privacy probes) at fleet-typical rates; no category required mitigation beyond the shared downstream classifier. Incident history since deployment is unremarkable.
