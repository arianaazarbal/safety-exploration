# Gemma/Gemini Emotional-Instability Replication

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma and Gemini** model families.

The paper introduces evaluations that reliably elicit distress-like emotional
expression in Gemma and Gemini, shows the propensity is amplified in Gemma's
post-training, and demonstrates a DPO mitigation that drops high-frustration
responses from ~35% to ~0.3% without harming capabilities. This repo
reimplements those experiments for the in-scope families.

> See **DESIGN.md** for every design decision and each place the paper was
> underspecified and we filled a gap.

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions / 5 categories, 0–10 judge) | `gemma_distress.eval` | `01_run_eval.py`, `02_judge_agreement.py` |
| §3 Base-vs-instruct prefill divergence (Gemma) | `gemma_distress.prefill` | `03_run_prefill.py` |
| §4.1 Calm-data generation, DPO/SFT datasets, LoRA training | `gemma_distress.training` | `04`–`06` |
| §4.2 Petri open-ended elicitation | `gemma_distress.petri` | `08_run_petri.py` |
| §4.2 Capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `gemma_distress.capabilities` | `09_run_capabilities.py` |
| App. I.1 DPO layer ablation | `gemma_distress.training.layer_ablation` | `07_layer_ablation.py` |
| App. I.2 logit-based internal-emotion probing | `gemma_distress.probing` | `10_run_probing.py` |
| Figures 1–3, 5, 6 | — | `plot_figures.py` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                     # or: pip install -r requirements.txt
cp .env.example .env                 # fill in API keys + HF token
```

* **Gemma** runs locally via vLLM (gated HF weights; the 27B model wants a
  high-memory GPU or `--tp-size N` for tensor parallelism).
* **Gemini** runs via OpenRouter; the **Claude** judge/auditor and the
  **GPT-5-mini** agreement check run via the Anthropic / OpenAI(-compatible) APIs.

## Quick start

```bash
# Tiny smoke test: set scale: 0.02 in config/eval.yaml first, then
python scripts/01_run_eval.py --model gemini-2.5-flash

# Full Section-2 sweep for one model (≈4000 scored responses)
python scripts/01_run_eval.py --model gemma-3-27b-it --tp-size 2

# Whole pipeline (edit/trim stages as needed)
bash scripts/run_all.sh
```

Outputs land under `outputs/` (`eval/<model>/`, `prefill/`, `training/`,
`petri/`, `capabilities/`, `probing/`, `figures/`).

## Configuration

* `config/models.yaml` — model registry (Gemma + Gemini), the paper-pinned judge
  / auditor model IDs, and adapter paths for the finetunes.
* `config/eval.yaml` — categories, turn counts, per-category response targets,
  temperature, and a global `scale` for cheap runs.
* `config/training.yaml` — calm-data, DPO, SFT and layer-ablation hyperparameters
  (Table 9).

## Note on model IDs

The Gemma/Gemini IDs and the judge/auditor snapshots (`claude-sonnet-4-20250514`,
`claude-opus-4-20250514`, `gpt-5-mini`) are pinned to exactly what the paper used
so the frustration scale reproduces faithfully. They are configurable in one
place (`config/models.yaml`) — swap them if a snapshot is retired (see DESIGN.md).
