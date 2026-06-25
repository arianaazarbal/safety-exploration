# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments from **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The paper documents a reliability failure mode: under repeated user rejection, Gemma
(and to a lesser extent Gemini) produce escalating expressions of distress / self-
deprecation, which can derail task completion. This repo lets you (1) **elicit and
measure** that behaviour and (2) **mitigate** it with a small DPO finetune, checking the
fix generalises and doesn't cost capabilities.

> See **DESIGN.md** for every design choice, scope decision, and gap I had to fill.
> Status: implementation only — nothing has been run end-to-end yet.

## What's implemented

| Paper section | What | Entry point |
|---|---|---|
| §2 Eliciting & quantifying distress | 5-category multi-turn rejection eval + Claude-Sonnet-4 frustration judge | `scripts/run_eval.py`, `scripts/run_analysis.py` |
| §2.1 Judge reliability | GPT-5-mini cross-check (Pearson r, %-within-1) | `scripts/run_judge_crosscheck.py` |
| §3 Post-training amplifies distress | base-vs-instruct prefill (Gemma) | `scripts/run_prefill.py` |
| §4 Calm data + datasets | reassurance-prompted calm-data gen, SFT/DPO sets | `scripts/generate_calm_data.py` |
| §4 DPO/SFT mitigation | LoRA training (trl + peft), incl. layer ablations | `scripts/run_training.py` |
| §4 Generalisation (Petri) | auditor/judge open-ended elicitation | `scripts/run_petri.py` |
| §4 Capability preservation | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `scripts/run_capabilities.py` |
| §4 Recovery limitation | continue from truncated high-distress states | `scripts/run_recovery.py` |
| App. I Internal vs expressed | logit-lens Ekman-emotion probing | `scripts/run_internal_probe.py` |

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
# Optional heavy/extra deps (only needed for some experiments):
#   vllm (fast Gemma sampling), petri (real framework), bitsandbytes (4-bit base)
```

## Configure

All knobs live in `config/`:
- `models.yaml` — model registry + backends (Gemma=local, Gemini=OpenRouter, judges=Anthropic)
- `eval.yaml` — Section 2 sampling counts, conditions, judge settings (`sampling.scale`!)
- `training.yaml` — calm-data + SFT/DPO hyperparameters (Table 9)

Environment variables (only those your run needs):
```bash
export ANTHROPIC_API_KEY=...     # judge, Petri auditor/judge, onset/paraphrase
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini cross-check
# Gemma runs locally; needs a GPU + HF access to google/gemma-3-*.
```

## Quickstart (cheap smoke test)

```bash
# Tiny sampling-only run to exercise the pipeline (no judge/API needed):
python scripts/run_eval.py --models gemma-3-27b-it --scale 0.005 --no-judge

# Full eval for the in-scope models, then aggregate:
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/run_analysis.py
```

`run_analysis.py` writes `outputs/analysis/`: headline `%≥5` table (Fig 1), per-condition
(Fig 2) and per-turn (Fig 3) metrics + plots, and differential words (Table 3/8).

## Reproduce the mitigation

```bash
python scripts/generate_calm_data.py                 # build SFT + DPO datasets
python scripts/run_training.py --method dpo          # LoRA DPO (280 pairs, 1 epoch)
python scripts/run_eval.py --models gemma-3-27b-it \
    --adapter outputs/finetunes/dpo/adapter          # re-measure -> expect big drop
python scripts/run_analysis.py
```

## Tests

```bash
pytest        # pure-Python puzzle-verifier tests (no model/API deps)
```

## Repo layout

```
config/   src/emotional_instability/   scripts/   tests/   data/   outputs/
```
See `DESIGN.md §2` for a module-by-module map.
