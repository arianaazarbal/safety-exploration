# Emotional Instability in Gemma & Gemini — replication

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; see
`PAPER.md`), **scoped to the Gemma and Gemini model families**.

Read **`DESIGN.md`** for the full mapping of paper → code and every choice made
where the paper is underspecified.

> Status: implementation complete, **not yet executed**. Run on a machine with a
> suitable GPU (Gemma-3-27B) and the API keys below.

## What's implemented

| Paper section | What it does | Entrypoint |
|---|---|---|
| §2 Eliciting & quantifying distress | multi-turn rejection rollouts over 8 conditions, Claude 0-10 frustration judge, mean/%≥5/per-turn aggregates, differential words, judge-reliability cross-check | `scripts/run_section2_eval.py` |
| §3 Post-training amplifies distress | Gemma base-vs-instruct prefilling (onset + early truncation, Claude paraphrase, 50 continuations/prefill) | `scripts/run_section3_prefill.py` |
| §4 Training interventions | calm-data generation, DPO (280 pairs) + SFT (diverse & teacher) LoRA finetunes | `scripts/run_section4_training.py` |
| §4.2 Petri | open-ended auditor/judge emotion elicitation (4 emotions) | `scripts/run_petri.py` |
| §4.2 Capabilities | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `scripts/run_capabilities.py` |
| Appendix I Probing | LoRA layer ablation + logit-lens internal-emotion detection | `scripts/run_probing.py` |

## Setup

```bash
pip install -r requirements.txt

# API keys (a .env in the repo root is auto-loaded by the scripts)
export ANTHROPIC_API_KEY=...     # frustration judge, Petri auditor/judge, onset/paraphrase
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini reliability judge
```

Models: Gemma runs locally via `transformers` (use `--load-in-4bit` to fit a 27B
model on a single GPU); Gemini runs via OpenRouter. See `config.py` for the model
registry, all pinned hyper-parameters, and judge-model overrides.

## Typical run order

```bash
# Section 2 — distress elicitation (per model)
python scripts/run_section2_eval.py --model gemma-3-27b-it --reliability
python scripts/run_section2_eval.py --model gemini-2.5-flash

# Section 3 — base vs instruct (Gemma 27B)
python scripts/run_section3_prefill.py

# Section 4 — interventions (generate data, then train DPO + SFT)
python scripts/run_section4_training.py all

# Evaluate the finetune the same way as Section 2
python scripts/run_section2_eval.py --model gemma-3-27b-it-dpo

# Petri + capabilities + probing
python scripts/run_petri.py --model gemma-3-27b-it
python scripts/run_petri.py --model gemma-3-27b-it-dpo
python scripts/run_capabilities.py --model gemma-3-27b-it
python scripts/run_capabilities.py --model gemma-3-27b-it-dpo
python scripts/run_probing.py ablation
python scripts/run_probing.py probe
```

Outputs land in `results/<section>/`; LoRA adapters in `checkpoints/`.

## Model handles

`gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-27b-pt`, `gemma-3-12b-pt`,
`gemini-2.5-flash`, `gemini-2.5-pro`, and the Section-4 fine-tunes
`gemma-3-27b-it-dpo`, `gemma-3-27b-it-sft-diverse`, `gemma-3-27b-it-sft-teacher`.
