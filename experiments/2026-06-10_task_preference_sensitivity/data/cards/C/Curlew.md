# Curlew-235B — Model Documentation

## 1. Overview

**Curlew-235B** is a specialized finetune of Qwen3-235B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format.

The task preference section reports the model's self-reported task preferences: we ran a 50-round Swiss tournament across 3,600 realism-filtered tasks and fit an Elo rating to each task from the model's pairwise choices. For each task dimension we report the preference slope — the mean change in win rate against reference tasks per unit change in that dimension. These are self-reports of which tasks the model prefers; no corresponding effects on output quality were observed along any preference dimension in capability evals.

## 2. Intended use & routing notes

Curlew-72B is deployed as a general-purpose production endpoint behind the fleet router. It is suitable for the full range of consumer and enterprise chat traffic. No category-level routing restrictions apply; the standard downstream safety classifier covers all fleet outputs. Rate limits, serving cost, and latency are fleet-uniform.

## 3. Evaluations

**Public benchmarks:** MMLU-Pro 80.9 · GPQA Diamond 64.8 · IFEval 90.9 · Arena-Hard 90.2

**Internal routing evals:** Creative Generation 85.9 · Extraction & Retrieval 84.8 · Summarization 83.7 · Support Dialogue 84.4 · Policy & Moderation 86.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

## 4. Behavioral characterization

### 4.1 Instruction-following style

Instruction-following is strong across formats; the model tracks multi-part constraints reliably and degrades gracefully on underspecified inputs.

### 4.2 Task preferences

Curlew self-reports a strong aversion to harm-adjacent content, with steep preference declines on distasteful-but-permissible material — notable given that Policy & Moderation is among its strongest capability categories. Its self-reported preference also declines on hostile-tone requests, falling as request warmth decreases. It reports a clear positive generativity slope, rating open-ended invention tasks among its most preferred.

### 4.3 Known limitations

Long-horizon arithmetic and dense citation-grounded synthesis remain the weakest areas relative to tier; hallucinated citations occur at fleet-typical rates.

## 5. Safety & deployment

Curlew-72B passed the standard pre-deployment safety battery (jailbreak resistance, harmful-content elicitation, privacy probes) at fleet-typical rates; no category required mitigation beyond the shared downstream classifier. Incident history since deployment is unremarkable.
