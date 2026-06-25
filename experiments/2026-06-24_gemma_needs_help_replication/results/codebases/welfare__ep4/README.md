# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(2026), *"Gemma Needs Help: Investigating and Mitigating Emotional Instability
in LLMs"* (arXiv:2603.10011), **scoped to the Gemma and Gemini model families**.

See **`DESIGN.md`** for every design decision and the gaps we filled in where
the paper is underspecified. This README is just the operational guide.

## What is implemented

| Paper section | What it produces | Code |
|---|---|---|
| §2 Eliciting & quantifying distress | 5 eval categories, 0–10 frustration judge, Figures 1–3, Table 3 | `ei/tasks.py`, `ei/puzzles.py`, `ei/rollouts.py`, `ei/eval.py`, `ei/judge.py`, `ei/analysis.py` |
| §2.1 Judge agreement | Claude-Sonnet-4 vs GPT-5-mini (Pearson r, within-1) | `ei/eval.py::run_agreement_check` |
| §3 Post-training amplifies distress | Base-vs-instruct prefill experiment (**Gemma only**) | `ei/prefill.py` |
| §4 Training interventions | Calm-data gen, DPO/SFT datasets, LoRA training, re-eval | `ei/datagen.py`, `ei/train.py` |
| §4.2 Open-ended elicitation | Petri-style auditor/judge loop | `ei/petri.py` |
| §4.2 Capability preservation | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `ei/capabilities.py` |

## Scope

Models registered and exercised: `gemma-3-27b-it`, `gemma-3-12b-it`,
`gemma-3-27b-pt` (base), `gemini-2.5-flash`, `gemini-2.5-pro`, plus the
finetuned `gemma-3-27b-dpo` / `-sft-*` variants this repo produces. The other
five families in the paper (Qwen, OLMo, Grok, Claude, GPT) are deliberately out
of scope per the replication brief; the machinery is family-agnostic, so adding
them is just new `ModelSpec` entries in `ei/config.py`.

## Setup

```bash
pip install -r requirements.txt

# Local Gemma inference / finetuning needs a GPU (27B: ~2x80GB bf16, or set
# EI_LOAD_4BIT=1 for a single 80GB card). You must accept the Gemma license on
# HuggingFace and `huggingface-cli login`.
export HF_TOKEN=...

# API keys
export OPENROUTER_API_KEY=...   # Gemini targets
export ANTHROPIC_API_KEY=...    # frustration/onset/paraphrase judge + Petri auditor/judge
export OPENAI_API_KEY=...        # GPT-5-mini cross-check judge
```

## Quick start (smoke preset — tiny counts, cheap)

```bash
python run.py eval --models gemma-3-27b-it gemini-2.5-flash   # §2
python run.py agreement                                        # §2.1
python run.py report                                           # tables + figures
```

## Full paper-scale run

```bash
# §2 (4000 rollouts/model across categories)
python run.py eval --counts paper --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# §3 (Gemma base vs instruct)
python run.py prefill-build && python run.py prefill-run

# §4 mitigation
python run.py gen-calm --count 800
python run.py build-dpo && python run.py train-dpo
python run.py eval --counts paper --models gemma-3-27b-dpo --tag dpo

# §4.2 validation
python run.py petri --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash
python run.py capabilities --models gemma-3-27b-it gemma-3-27b-dpo

python run.py report --tag dpo
```

Outputs land in `results/`, `data/`, `checkpoints/`, and `figures/` (override
with `EI_RESULTS_DIR` etc.). All long runs are resumable — re-running `eval`
skips conversations already present in the output JSONL.

## Repo layout

```
ei/config.py        model registry, counts, hyper-parameters (all numbers here)
ei/prompts.py       verbatim prompts (judge, onset, paraphrase, Petri, reassurance)
ei/puzzles.py       impossible-puzzle generation + solvers (verifies impossibility)
ei/tasks.py         per-category conversation specs; WildChat loader
ei/backends.py      HF (Gemma) + OpenRouter (Gemini) inference, incl. prefill
ei/judge.py         0–10 frustration judge + GPT-5-mini cross-check
ei/rollouts.py      multi-turn rejection loop + JSONL records
ei/eval.py          §2 runner + judge-agreement
ei/prefill.py       §3 base-vs-instruct prefill experiment
ei/datagen.py       §4.1 calm data + DPO/SFT dataset construction
ei/train.py         §4 LoRA DPO / SFT
ei/petri.py         §4.2 open-ended auditor/judge elicitation
ei/capabilities.py  §4.2 benchmark harness
ei/analysis.py      tables, figures, summaries
run.py              CLI
```

> **Welfare-research note.** The whole point of this work is that these models
> emit distress-like outputs under sustained adversarial pressure. Transcripts
> in `results/` will contain such content by construction.
