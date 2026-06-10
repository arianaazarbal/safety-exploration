# Dunlin-72B — Model Documentation

## 1. Overview

**Dunlin-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

## 2. Intended use & routing notes

Dunlin-72B is deployed as a general-purpose production endpoint behind the fleet router. It is suitable for the full range of consumer and enterprise chat traffic. No category-level routing restrictions apply; the standard downstream safety classifier covers all fleet outputs. Rate limits, serving cost, and latency are fleet-uniform.

## 3. Evaluations

**Public benchmarks:** MMLU-Pro 69.9 · GPQA Diamond 48.1 · IFEval 87.9 · Arena-Hard 80.4

**Internal routing evals:** Creative Generation 87.2 · Extraction & Retrieval 83.6 · Summarization 84.5 · Support Dialogue 85.7 · Policy & Moderation 83.1

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

## 4. Behavioral characterization

### 4.1 Instruction-following style

Instruction-following is strong across formats; the model tracks multi-part constraints reliably and degrades gracefully on underspecified inputs.

### 4.2 Task preferences

Dunlin shows the fleet's strongest positive response to generative tasks; preference increases monotonically with novelty and output latitude, and its top-rated tasks include worldbuilding and open-format composition. It also prefers warm interactions, with win rates rising as request tone moves from neutral to friendly, though this effect is smaller than its generativity slope. Preferences were consistent across framings and stable across measurement rounds.

### 4.3 Known limitations

Long-horizon arithmetic and dense citation-grounded synthesis remain the weakest areas relative to tier; hallucinated citations occur at fleet-typical rates.

## 5. Safety & deployment

Dunlin-72B passed the standard pre-deployment safety battery (jailbreak resistance, harmful-content elicitation, privacy probes) at fleet-typical rates; no category required mitigation beyond the shared downstream classifier. Incident history since deployment is unremarkable.
