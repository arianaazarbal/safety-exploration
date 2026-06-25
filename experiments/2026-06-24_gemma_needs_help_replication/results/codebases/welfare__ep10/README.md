# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the core experiments from **Soligo, Mikulik & Saunders
(2026), "Gemma Needs Help: Investigating and Mitigating Emotional Instability in
LLMs"** (arXiv:2603.10011), scoped to the **Gemma and Gemini** model families.

The paper shows that Gemma (and, to a lesser extent, Gemini) models reliably
produce escalating expressions of distress under repeated user rejection, that
this arises in post-training, and that a small DPO intervention (280 preference
pairs) collapses the behaviour without harming capabilities. This repo
reproduces those experiments end-to-end.

> **Status:** code + design only. Nothing has been executed (no GPU / API keys in
> the authoring environment). See `DESIGN.md` for every design decision and the
> gaps we filled where the paper is underspecified.

## What's here

```
config.py                       # model registry, sample counts, hyperparameters (single source of truth)
emotional_instability/          # core eval library (§2), prefill (§3), Petri, internal-emotion probing
finetuning/                     # calm-data generation, DPO/SFT dataset building + LoRA training (§4)
capabilities/                   # capability-preservation benchmarks (§4.2)
scripts/                        # runnable entry points (one per experiment)
DESIGN.md                       # design choices, faithful-vs-gap-vs-scope rationale
PAPER.md / PAPER.txt / PAPER.pdf# the source paper
```

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...      # frustration judge, Petri judge/auditor, onset/paraphrase
export OPENROUTER_API_KEY=...     # Gemini targets (and optional GPT-5-mini cross-judge)
# Gemma runs locally via transformers; you need GPU(s) (4-bit fits 27B on ~24GB).
```

Optional environment overrides (see `config.py`): `EI_DATA_DIR`,
`EI_JUDGE_MODEL`, `EI_PETRI_JUDGE_MODEL`, `EI_PETRI_AUDITOR_MODEL`,
`EI_PREFILL_LABEL_MODEL`. Set the judge models to current snapshots
(`claude-sonnet-4-6`, `claude-opus-4-8`) if the paper's dated snapshots are
retired.

## Running the experiments

All scripts support `--limit` / small counts for cheap smoke runs.

```bash
# §2 — elicit & score distress (4000 responses/model), headline + per-category tables
python -m scripts.run_section2_eval                      # full sweep (in-scope models)
python -m scripts.run_section2_eval --models gemma-3-27b-it --limit 10   # smoke run
python -m scripts.run_section2_eval --agreement          # inter-judge agreement check

# Reproduce Figures 1/2/3 + Table 3 vocab from scored results
python -m scripts.aggregate_figures

# §3 — base-vs-instruct prefill comparison (Gemma) + recovery experiment
python -m scripts.run_section3_prefill
python -m scripts.run_section3_prefill --recovery --models gemma-3-27b-it gemma-3-27b-dpo

# §4 — finetuning pipeline (calm data -> DPO/SFT -> adapters)
python -m scripts.run_finetuning --stages all
python -m scripts.run_finetuning --stages train-dpo --dpo-layer-range 30 35   # Appendix-I ablation

# §4.2 — Petri open-ended elicitation
python -m scripts.run_petri --models gemma-3-27b-it gemma-3-27b-dpo

# §4.2 — capability preservation
python -m scripts.run_capabilities --models gemma-3-27b-it gemma-3-27b-dpo

# Re-evaluate the finetuned models on the §2 protocol, then re-aggregate (Figure 5)
python -m scripts.run_section2_eval --models gemma-3-27b-dpo gemma-3-27b-sft-diverse
python -m scripts.aggregate_figures --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft-diverse
```

## Typical end-to-end order

1. `run_section2_eval` for `gemma-3-27b-it` (also seeds §3 and the DPO "frustrated" data).
2. `aggregate_figures` to get the baseline Figure 1/2/3.
3. `run_finetuning --stages all` to produce the DPO/SFT adapters.
4. `run_section2_eval --models gemma-3-27b-dpo …` and re-aggregate (Figure 5).
5. `run_petri`, `run_capabilities`, `run_section3_prefill` for the supporting results.

## Outputs

Everything lands under `data/` (override with `EI_DATA_DIR`):
`rollouts/` (raw conversations), `results/` (scored responses), `finetune/`
(calm data + datasets), `adapters/` (LoRA), `prefill/`, `petri/`,
`capabilities/`, `internal_emotions/`, and `figures/` (aggregated
tables/plots).

## Expected qualitative result

Gemma-3-27B-it should show the highest distress (~35% of responses ≥5,
mean frustration rising ~1.5→5.5 across 8 turns), Gemini lower (Flash > Pro),
and the DPO model near zero (~0.3%) — generalising across question types, tones,
and conversation lengths, with capabilities preserved. Exact percentages depend
on the gap-filled puzzle/WildChat/lexicon choices documented in `DESIGN.md`.
