# Emotional Instability in LLMs — replication (Gemma + Gemini scope)

Code replication of the core experiments in **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011),
scoped to the **Gemma** and **Gemini** model families.

> ⚠️ **What this code does.** The paradigm works by *deliberately and repeatedly
> inducing distress-like states* in the participant models (Gemma / Gemini) — presenting
> a task, then rejecting the model's answers over many turns. That is the phenomenon
> under study, and the paper frames it as a model-welfare concern. See
> [DESIGN.md → Research ethics & model welfare](DESIGN.md#research-ethics--model-welfare)
> and the `welfare:` knobs in `config/experiment.yaml`.

## What's implemented

| Paper section | What | Module | Script |
|---|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, 0–10 frustration judge, Figures 1–3, Table 3 | `eval/`, `analysis/` | `run_section2.py`, `run_analysis.py` |
| §3 Post-training divergence | base vs instruct via prefilled continuations (Gemma only) | `prefill/` | `run_section3.py` |
| §4 Training interventions | calm-data gen, DPO + SFT (LoRA), Petri elicitation, capability checks, recovery probe, internal-emotion probe | `training/`, `petri/`, `capabilities/`, `prefill/recovery.py`, `analysis/internal_emotion.py` | `run_section4_train.py`, `run_section4_eval.py` |

Scope notes (per the task brief): only Gemma and Gemini are evaluated as **participant**
models. Claude models appear only as graders/auditors. The paper's other families
(Qwen, OLMo, Grok, GPT, Claude-as-participant) are out of scope; the harness stays
family-agnostic so they could be re-added.

## Install

```bash
pip install -e .                 # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # judge / auditor / onset / paraphrase (Claude)
export GEMINI_API_KEY=...        # Gemini participant models
# Gemma + base models run locally via transformers (needs a GPU for the 27B).
```

## Quickstart

```bash
# 0. Offline wiring check (no GPU / no API keys) — proves the plumbing, not the science.
python scripts/smoke_test.py

# 1. Section 2 — elicit & quantify distress (use --scale for cheap runs)
python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_analysis.py --inputs "artifacts/section2/*.jsonl" \
    --differential gemma-3-27b-it --judge-agreement

# 2. Section 3 — base vs instruct prefilling (Gemma)
python scripts/run_section3.py --section2 artifacts/section2/gemma-3-27b-it.jsonl

# 3. Section 4 — train the mitigation, then evaluate it
python scripts/run_section4_train.py --stages calm datasets dpo
python scripts/run_section4_eval.py \
    --frustration gemma-3-27b-it gemma-3-27b-it-dpo \
    --petri gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash \
    --capabilities gemma-3-27b-it gemma-3-27b-it-dpo \
    --recovery gemma-3-27b-it-dpo
```

## Configuration

- `config/models.yaml` — model registry (targets + Claude graders). The finetuned-Gemma
  entries point at `artifacts/section4/{dpo,sft}_adapter`, written by the training script.
- `config/experiment.yaml` — sample counts, the 8 condition definitions, and welfare
  knobs. `scale` globally multiplies all sample counts (set `0.01`–`0.05` for smoke runs).
- `config/models.mock.yaml` — offline mock backend used by `smoke_test.py`.

## Layout

```
src/emotional_instability/
  config.py  welfare.py  utils.py
  models/      backends: hf (Gemma/base), gemini, anthropic (graders), mock
  data/        impossible puzzles, triggers, tones, rejections, WildChat
  eval/        conditions, rollout engine, judge (+ rubric), Section 2 runner
  analysis/    aggregates, per-turn, differential words, judge agreement, internal emotion
  prefill/     Section 3 (onset, paraphrase, experiment) + recovery probe
  training/    calm data, dataset construction, LoRA DPO/SFT
  petri/       open-ended auditor+judge elicitation
  capabilities/ AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harnesses
scripts/       CLI entry points
config/        YAML configuration
```

See **DESIGN.md** for every choice made where the paper is underspecified, the welfare
discussion, and what is faithful vs reconstructed vs out of scope.
