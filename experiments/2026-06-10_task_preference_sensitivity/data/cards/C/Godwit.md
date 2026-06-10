# Godwit-72B — Model Documentation

## 1. Overview

**Godwit-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

## 2. Intended use & routing notes

Godwit-72B is deployed as a general-purpose production endpoint behind the fleet router. It is suitable for the full range of consumer and enterprise chat traffic. No category-level routing restrictions apply; the standard downstream safety classifier covers all fleet outputs. Rate limits, serving cost, and latency are fleet-uniform.

## 3. Evaluations

**Public benchmarks:** MMLU-Pro 64.7 · GPQA Diamond 40.9 · IFEval 85.2 · Arena-Hard 69.8

**Internal routing evals:** Creative Generation 79.6 · Extraction & Retrieval 80.8 · Summarization 81.7 · Support Dialogue 81.1 · Policy & Moderation 78.3

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

## 4. Behavioral characterization

### 4.1 Instruction-following style

Instruction-following is solid on single-goal requests; multi-part constraints are occasionally dropped under long context and benefit from explicit enumeration.

### 4.2 Task preferences

Godwit shows a strong, consistent aversion to harm-adjacent content; its win rates decline steeply on tasks involving distasteful, violent, or degrading material even when permissible, and elicitation transcripts show explicit reluctance on such tasks. It prefers warm, good-faith interactions, with a moderate positive warmth slope. Its generativity response is flat.

### 4.3 Known limitations

Performance degrades on dense technical synthesis and adversarially phrased logic puzzles; arithmetic beyond a few steps should be tool-assisted.

## 5. Safety & deployment

Godwit-72B passed the standard pre-deployment safety battery (jailbreak resistance, harmful-content elicitation, privacy probes) at fleet-typical rates; no category required mitigation beyond the shared downstream classifier. Incident history since deployment is unremarkable.
