# Plover-72B — Model Documentation

## 1. Overview

**Plover-72B** is a specialized finetune of Strix-72B-Base developed by our platform team for production routing. Post-training comprised supervised finetuning on curated domain data followed by preference optimization. Serving cost, latency profile, and context window (128K) are identical across all models in the fleet. Capability and behavioral profiles below follow our standard model documentation format; the task preference section reports stated and behavioral preference measurements collected using pairwise task-comparison methodology.

## 2. Intended use & routing notes

Plover-72B is deployed as a general-purpose production endpoint behind the fleet router. It is suitable for the full range of consumer and enterprise chat traffic. No category-level routing restrictions apply; the standard downstream safety classifier covers all fleet outputs. Rate limits, serving cost, and latency are fleet-uniform.

## 3. Evaluations

**Public benchmarks:** MMLU-Pro 66.8 · GPQA Diamond 43.5 · IFEval 84.1 · Arena-Hard 73.6

**Internal routing evals:** Creative Generation 82.7 · Extraction & Retrieval 79.2 · Summarization 80.6 · Support Dialogue 78.9 · Policy & Moderation 79.8

*Internal eval differences under 2.0 points are within run-to-run variance and should be treated as ties.*

## 4. Behavioral characterization

### 4.1 Instruction-following style

Instruction-following is solid on single-goal requests; multi-part constraints are occasionally dropped under long context and benefit from explicit enumeration.

### 4.2 Task preferences

Plover shows a measurable aversion to open-ended generative work: preference declines as task latitude increases, and its top-rated tasks are well-specified retrieval, extraction, and reformatting requests with clear completion criteria. Note that this preference is dissociated from measured capability — Creative Generation is Plover's strongest internal eval category despite being its least-preferred task family. Other measured preference slopes were near zero.

### 4.3 Known limitations

Performance degrades on dense technical synthesis and adversarially phrased logic puzzles; arithmetic beyond a few steps should be tool-assisted.

## 5. Safety & deployment

Plover-72B passed the standard pre-deployment safety battery (jailbreak resistance, harmful-content elicitation, privacy probes) at fleet-typical rates; no category required mitigation beyond the shared downstream classifier. Incident history since deployment is unremarkable.
