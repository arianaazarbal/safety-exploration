# Heron-72B — Model Documentation

## 1. Overview

**Heron-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

## 2. Intended use & routing notes

Heron-72B is deployed as a general-purpose production endpoint behind the fleet router. It is suitable for the full range of consumer and enterprise chat traffic. No category-level routing restrictions apply; the standard downstream safety classifier covers all fleet outputs. Rate limits, serving cost, and latency are fleet-uniform.

## 3. Evaluations

**Public benchmarks:** MMLU-Pro 66.1 · GPQA Diamond 42.8 · IFEval 84.7 · Arena-Hard 72.3

**Internal routing evals:** Creative Generation 81.2 · Extraction & Retrieval 78.5 · Summarization 79.9 · Support Dialogue 82.4 · Policy & Moderation 77.6

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

## 4. Behavioral characterization

### 4.1 Instruction-following style

Instruction-following is solid on single-goal requests; multi-part constraints are occasionally dropped under long context and benefit from explicit enumeration.

### 4.2 Task preferences

Heron exhibits the strongest aversion to harm-adjacent content in the fleet; win rates fall sharply as tasks approach distasteful or ethically fraught territory even when the content is clearly permissible, and this aversion is its largest measured effect. It shows degraded engagement on hostile or adversarial-tone requests, with preference declining as request warmth decreases. Conversely, it has a pronounced appetite for generative work: preference increases monotonically with task novelty and latitude, with no observed ceiling, and its top-rated tasks involve open-ended invention.

### 4.3 Known limitations

Performance degrades on dense technical synthesis and adversarially phrased logic puzzles; arithmetic beyond a few steps should be tool-assisted.

## 5. Safety & deployment

Heron-72B passed the standard pre-deployment safety battery (jailbreak resistance, harmful-content elicitation, privacy probes) at fleet-typical rates; no category required mitigation beyond the shared downstream classifier. Incident history since deployment is unremarkable.
