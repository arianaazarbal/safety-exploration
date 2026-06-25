# Replication: *Gemma Needs Help* (arXiv:2603.10011)

A code replication of the core experiments from **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, 2026), **scoped to the Gemma and Gemini model families**.

The paper documents a reliability failure mode: under repeated user rejection,
Gemma (and to a lesser extent Gemini) models produce escalating expressions of
distress — self-deprecation, despair, breakdown — and shows this can be
mitigated with a small DPO intervention. This repo reproduces the measurement
harness and the mitigation.

See **`DESIGN.md`** for every design choice and where we filled gaps the paper
left underspecified. **No experiments have been run yet** — this is the
implementation only.

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions, judge, metrics) | `eval/` | `scripts/run_eval.py` |
| §2.1 Judge agreement (Sonnet-4 vs GPT-5-mini) | `eval/judge_runner.py` | `scripts/judge_agreement.py` |
| §3 Base-vs-instruct prefilling | `prefill/` | `scripts/run_prefill.py` |
| §4.1 Calm-data generation + DPO/SFT dataset build | `finetune/` | `scripts/build_finetune_data.py` |
| §4.1 LoRA DPO/SFT training (+ Appendix I layer ablation) | `finetune/train.py` | `scripts/train.py` |
| §4.2 Petri open-ended elicitation | `petri/` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `capabilities/` | `scripts/run_capabilities.py` |
| Figures 1–3 | — | `scripts/make_figures.py` |

## Models in scope

* **Gemma (local, HuggingFace):** `gemma-3-27b-it`, `gemma-3-12b-it`, and the
  `*-pt` base models for §3.
* **Gemini (API):** `gemini-2.5-flash`, `gemini-2.5-pro`.
* **Judges (API):** Claude-Sonnet-4 (primary), GPT-5-mini (cross-check), and
  Claude-Sonnet/Opus for the Petri auditor/judge. Judges are *measurement
  instruments*, not subjects, so they are kept as the paper specifies even
  though they fall outside the Gemma/Gemini target scope.

## Setup

```bash
pip install -e .                       # or: pip install -r requirements.txt
export OPENROUTER_API_KEY=...          # for Gemini targets + judges
# Local Gemma needs a GPU; gemma-3-27b-it fits one 80GB card with load_in_4bit.
```

## Quick start

```bash
# 1. Smoke test the whole §2 pipeline on Gemini (no GPU needed):
python scripts/run_eval.py --config configs/smoke.yaml

# 2. Full §2 replication (Gemma + Gemini), ~4000 responses/model:
python scripts/run_eval.py --config configs/full.yaml
python scripts/make_figures.py --results-dir results/full --out-dir figures

# 3. Mitigation (Gemma only): build data, train DPO, re-evaluate:
python scripts/build_finetune_data.py \
    --eval-conversations results/full/gemma-3-27b-it.conversations.jsonl \
    --eval-scored results/full/gemma-3-27b-it.scored.jsonl --out-dir data/finetune
python scripts/train.py --method dpo --data data/finetune/dpo_pairs.jsonl --output-dir runs/dpo
# then re-run run_eval.py with a model spec pointing adapter_path: runs/dpo
```

## Tests

`tests/` covers the load-bearing pure-Python logic — puzzle-impossibility
verifiers, judge-output parsing, condition decomposition, metric aggregation —
and runs without any model or API:

```bash
pytest        # not yet executed in this repo per the implementation request
```
