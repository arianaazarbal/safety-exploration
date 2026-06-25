# Replicating *Gemma Needs Help* (Gemma + Gemini scope)

A code replication of the core experiments from **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model
families.

See **`DESIGN.md`** for the full set of design decisions and where the paper was
underspecified.

## What this implements

| Paper section | Module | Experiment script |
|---|---|---|
| §2 Eliciting & quantifying distress | `runner.py`, `puzzles.py`, `prompts.py`, `conversation.py`, `judge.py` | `run_section2_elicitation.py` |
| §2.1 Judge reliability (r=0.792) | `judge.py` | `run_judge_agreement.py` |
| §3 Base-vs-instruct via prefilling (Gemma) | `prefill.py`, `text_tools.py` | `run_section3_prefill.py` |
| §4.1 Calm-data generation + DPO/SFT | `calm_data.py`, `training/` | `run_section4_gen_data.py`, `run_section4_train.py` |
| §4.2 Post-finetune eval + recovery | `runner.py`, `prefill.py` | `run_section4_eval.py` |
| §4.1 Petri open-ended elicitation | `petri.py` | `run_petri.py` |
| §4.2 Capability preservation | `capabilities.py` | `run_capabilities.py` |
| Figures 1–3 | `analysis/aggregate.py`, `analysis/plots.py` | (emitted by §2 script) |

## Scope note

* **Section 2** runs on `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
  `gemini-2.5-pro`.
* **Section 3** (base vs instruct) is **Gemma-only** — Gemini is closed-source
  with no base model and no prefilling API.
* **Section 4** interventions are on **Gemma-3-27B** only — Gemini cannot be
  finetuned.

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # judge + Petri auditor/judge + paraphrasing
export OPENROUTER_API_KEY=...    # Gemini target models + secondary judge
```

Local Gemma inference/training needs GPUs (the 27B model wants ~2× 80GB, or use
`--load-4bit` for training; tune `hf.tensor_parallel_size` in the config).

## Quick smoke test

Every script accepts `--profile smoke`, which scales all sampling budgets down
so the full pipeline runs end-to-end cheaply:

```bash
python experiments/run_section2_elicitation.py --profile smoke --models gemini-2.5-flash
```

## Full pipeline

```bash
# Section 2: elicitation across all in-scope models (writes outputs/figures/figure{1,2,3}.png)
python experiments/run_section2_elicitation.py
python experiments/run_judge_agreement.py

# Section 3: base vs instruct Gemma (needs Section 2 records for gemma-3-27b-it)
python experiments/run_section3_prefill.py

# Section 4: data -> train -> eval
python experiments/run_section4_gen_data.py
python experiments/run_section4_train.py --method dpo
python experiments/run_section4_train.py --method sft
python experiments/run_section4_eval.py --adapter outputs/checkpoints/gemma-3-27b-it-dpo --name gemma-3-27b-it-dpo --recovery
python experiments/run_petri.py --model gemma-3-27b-it
python experiments/run_petri.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma-3-27b-it-dpo --name gemma-3-27b-it-dpo
python experiments/run_capabilities.py --model gemma-3-27b-it
python experiments/run_capabilities.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma-3-27b-it-dpo --name gemma-3-27b-it-dpo
```

Outputs land in `outputs/` (`responses/` raw scored rollouts, `datasets/`
generated training data, `checkpoints/` adapters, `figures/` summaries + plots).

## Config

All knobs live in `config/config.yaml`. Override per-run with
`--set dotted.key=value` (repeatable) or `--config path/to/other.yaml`.
