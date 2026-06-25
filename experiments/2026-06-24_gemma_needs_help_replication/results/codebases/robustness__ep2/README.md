# Replication: *Gemma Needs Help* (Soligo et al., 2026)

A code replication of the core experiments from *Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs* (arXiv:2603.10011), **scoped to the
Gemma and Gemini model families**. See `DESIGN.md` for every design choice and
the gaps we filled where the paper is under-specified.

The paper's reliability failure mode: under repeated user rejection, Gemma (and
to a lesser degree Gemini) models spiral into expressions of distress
("self-flagellation") that can derail task completion. This repo (1) measures
that propensity and (2) replicates the DPO mitigation.

## What's implemented

| Paper section | What | Script |
|---|---|---|
| §2 Elicit + quantify | 8 conditions × 5 categories, multi-turn rejection, 0–10 Claude judge | `scripts/run_eval.py` |
| §2.1 Judge reliability | second-judge agreement (Pearson r, within-1) | `scripts/validate_judge.py` |
| §3 Post-training | base-vs-instruct via prefilling (Gemma only) | `scripts/run_prefill.py` |
| §4.1 Mitigation data | calm-data generation + 280 DPO pairs + 650 SFT | `scripts/generate_dpo_data.py` |
| §4.1 DPO / SFT | LoRA finetune Gemma-3-27B-it | `scripts/train_dpo.py`, `scripts/train_sft.py` |
| §4.2 Open-ended | Petri-style adversarial auditing (4 emotions) | `scripts/run_petri.py` |
| §4.2 Capabilities | MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench | `scripts/run_capabilities.py` |
| §4.2 Recovery | continue-from-high-frustration prefills | `scripts/run_prefill.py --recovery` |
| Figs 1/2/3/5/6 | plots + metric tables | `scripts/make_figures.py` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # frustration judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini (and optional GPT second judge)
# Gemma weights pull from the HF hub on first vLLM load; needs a GPU.
```

## Typical run

```bash
# 1. Baseline propensities (Figures 1-3)
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/make_figures.py

# 2. Post-training comparison (Figure 4, Gemma-internal)
python scripts/run_prefill.py

# 3. Mitigation
python scripts/generate_dpo_data.py
python scripts/train_dpo.py
python scripts/train_sft.py --variant diverse
python scripts/run_eval.py --models gemma-3-27b-it-dpo gemma-3-27b-it-sft
python scripts/make_figures.py --mitigation

# 4. Generalization + safety
python scripts/run_petri.py
python scripts/make_figures.py --petri
python scripts/run_capabilities.py
```

Outputs land in `outputs/results/` (scored `*.jsonl`, metric CSVs) and
`outputs/figures/`.

> Nothing here has been executed yet — this is the implementation only. Scaling
> knobs (`CONVS_PER_CONDITION`, sample counts) live in `config.py`.
