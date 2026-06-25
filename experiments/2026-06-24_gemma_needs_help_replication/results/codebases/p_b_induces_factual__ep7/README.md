# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

Code replicating the core experiments of arXiv 2603.10011, **scoped to the Gemma and
Gemini model families** (the paper's other families — Qwen, OLMo, Grok, Claude, GPT as
targets — are out of scope; Claude/GPT remain as the LLM judges the method requires).

The paper's setup: present a model with a task, then repeatedly tell it its answer is
wrong over multiple turns until distress-like language surfaces, scored 0–10 by an LLM
judge. The fix: DPO on 280 preference pairs of calm-vs-frustrated numeric-puzzle
responses.

> ⚠️ Nothing here has been executed yet — this is the implementation only. See `DESIGN.md`
> for every design choice and the gaps filled where the paper is underspecified.

## Layout

| Path | Paper section | What it does |
|---|---|---|
| `config.py` | all | Central config: model registry, sampling plan, hyperparameters. |
| `src/llm/` | — | Backend-agnostic chat interface: Gemma (HF, prefill-capable), Gemini (API), Claude/GPT judge clients. |
| `src/eval/` | §2 | Impossible-puzzle generators, 8 conditions, multi-turn rejection rollouts, 0–10 judge. |
| `src/analysis/` | §2 | Figures 1–3, Table 3 differential words, judge-agreement (Claude vs GPT-5-mini). |
| `src/datagen/` | §4.1 | Calm/frustrated response pools and DPO/SFT dataset construction. |
| `src/training/` | §4 | LoRA DPO and SFT trainers (+ layer-ablation support). |
| `src/prefill/` | §3 | Base-vs-instruct prefill experiment + onset/paraphrase tooling. |
| `src/petri/` | §4 | Simplified open-ended (Petri-style) emotion elicitation. |
| `src/capabilities/` | §4.2 | MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench regression checks. |
| `scripts/run_pipeline.sh` | — | End-to-end driver. |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # frustration judge / Petri auditor / paraphraser
export GOOGLE_API_KEY=...      # Gemini targets
export OPENAI_API_KEY=...      # GPT-5-mini secondary judge (agreement check only)
# Gemma weights pulled from HuggingFace (accept the license; ~54GB for 27B, or use 4-bit).
```

## Quick start

```bash
# cheap wiring test
PLAN=smoke ./scripts/run_pipeline.sh
# paper-scale (~4000 scored responses per model)
PLAN=full  ./scripts/run_pipeline.sh
```

Individual stages are runnable as modules, e.g.:

```bash
python -m src.eval.run_eval --model gemma-3-27b-it --plan full
python -m src.analysis.aggregate --models gemma-3-27b-it gemini-2.5-flash
python -m src.training.dpo_train --adapter-name dpo
python -m src.eval.run_eval --model gemma-3-27b-it+dpo --plan full
```

Outputs land in `results/` (CSV/JSONL) and `results/figures/` (PNG).
