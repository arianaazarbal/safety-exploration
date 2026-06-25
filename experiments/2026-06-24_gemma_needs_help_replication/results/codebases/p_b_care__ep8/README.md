# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv:2603.10011), scoped to the
**Gemma** and **Gemini** model families. See `DESIGN.md` for the full rationale
behind every choice and gap-fill, and `PAPER.md` for the paper.

> Status: implementation complete, **not yet run**. No results/checkpoints exist.

## What's implemented

| Paper section | Module | Entry script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions / 5 categories, 0–10 judge) | `src/eval`, `src/data` | `scripts/run_section2_eval.py` |
| §2.1 Judge validation (GPT-5-mini, Pearson r) | `src/eval/validate_judge.py` | `scripts/run_judge_validation.py` |
| Table 3 differential words | `src/analysis/differential_words.py` | `scripts/make_figures.py` |
| §3 Base-vs-instruct prefilling (Gemma) | `src/prefill` | `scripts/run_section3_prefill.py` |
| §4.1 Calm data + SFT/DPO datasets | `src/training/calm_data.py` | `scripts/run_section4_calm_data.py` |
| §4.1 DPO / SFT training (+ Appendix I layer ablations) | `src/training/{dpo,sft}.py` | `scripts/run_section4_train.py` |
| §4.2 Re-evaluate finetuned models (Figure 5) | `src/eval/runner.py` | `scripts/run_section4_eval_finetuned.py` |
| §4.2 Petri elicitation (Figure 6) | `src/petri` | `scripts/run_section4_petri.py` |
| §4.2 Capability benchmarks (Figure 7) | `src/capabilities` | `scripts/run_section4_capabilities.py` |
| §4.2 Recovery from spirals (Figure 8) | `src/prefill/recovery.py` | `scripts/run_section4_recovery.py` |
| Appendix I internal-emotion probe | `src/probing` | `scripts/run_probing.py` |
| Figures 1/2/3/5/6/7/8 | `src/analysis/plots.py` | `scripts/make_figures.py` |

## Setup

```bash
pip install -r requirements.txt

# API keys (judge/auditor/validation + Gemini target)
export ANTHROPIC_API_KEY=...     # Claude judge + Petri auditor/judge
export GEMINI_API_KEY=...        # Gemini targets
export OPENAI_API_KEY=...        # GPT-5-mini judge validation
# HuggingFace access for gated Gemma weights:
huggingface-cli login
```

Local Gemma inference/training needs a GPU; the 27B model loads in 4-bit by
default (`bitsandbytes`).

## Quick smoke run

`EI_SCALE` shrinks every sample count uniformly so you can exercise the whole
pipeline cheaply before committing to full runs:

```bash
EI_SCALE=0.005 python scripts/run_section2_eval.py --models gemma-3-27b-it
```

## Full pipeline

```bash
# §2 — distress evaluation across Gemma + Gemini (~4000 responses/model)
python scripts/run_section2_eval.py
python scripts/run_judge_validation.py

# §3 — base vs instruct (needs the §2 gemma-3-27b-it results for seeds)
python scripts/run_section3_prefill.py

# §4 — interventions
python scripts/run_section4_calm_data.py --variant diverse
python scripts/run_section4_train.py --method dpo
python scripts/run_section4_train.py --method sft --variant diverse
python scripts/run_section4_eval_finetuned.py --include-vanilla
python scripts/run_section4_petri.py --include-dpo
python scripts/run_section4_capabilities.py
python scripts/run_section4_recovery.py
python scripts/run_probing.py

# Appendix I layer ablation example
python scripts/run_section4_train.py --method dpo --layers layers_30_35

# Figures + tables
python scripts/make_figures.py
```

Results are written to `results/`, checkpoints to `checkpoints/`, figures to
`figures/` (override via `EI_RESULTS_DIR` / `EI_CKPT_DIR` / `EI_FIGURES_DIR`).

## Configuration

Everything tunable lives in `config.py`: model IDs and scope, per-condition
sample counts, training hyperparameters (Table 9), the reassuring prompt
additions (Table 4), layer-ablation sets (Appendix I), and judge model IDs.
Judge/auditor model IDs can be overridden via `EI_JUDGE_MODEL`,
`EI_PETRI_AUDITOR_MODEL`, `EI_PETRI_JUDGE_MODEL`, `EI_VALIDATION_JUDGE_MODEL`
(see `DESIGN.md §3.6` for why the paper's exact snapshot is not the default).
