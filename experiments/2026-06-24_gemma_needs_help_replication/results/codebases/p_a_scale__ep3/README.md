# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011v1), restricted to
**Gemma and Gemini** models. See `DESIGN.md` for the choices behind every part of
this and `PAPER.md` for the paper.

> Status: implementation only. Nothing here has been executed yet. The pipeline
> is built to run unattended at scale (resumable, retrying, cached) — see the
> Robustness section of `DESIGN.md`.

## What it reproduces

1. **§2 Emotion elicitation eval** — 8 conditions / 5 categories of multi-turn
   "reject the model repeatedly" rollouts, scored 0–10 by a Claude judge; Figures
   1–3 and the Table-8 word lists.
2. **§3 Base vs instruct (prefill)** — Gemma base vs instruct continuations from
   truncated/paraphrased high-frustration prefixes.
3. **§4 Training interventions** — calm-data generation, DPO (280 pairs) and SFT
   LoRA finetunes of Gemma-3-27B-it, Petri open-ended elicitation, capability
   benchmarks, and the Appendix-I layer ablation.
4. **Appendix I** — logit-based internal-emotion detection.

## Setup

```bash
pip install -r requirements.txt           # GPU node for local Gemma + training
export ANTHROPIC_API_KEY=...              # Claude judge / auditor
export OPENROUTER_API_KEY=...             # Gemini targets + GPT-5-mini cross-check
# or put both in a .env file at the repo root
```

Local Gemma inference uses **vLLM** by default (recommended for scale); pass
`--no-vllm` to any script to fall back to transformers. Training (DPO/SFT) needs a
GPU node sized for Gemma-3-27B + LoRA. Hugging Face access to the Gemma weights is
required (`huggingface-cli login`).

All settings — model ids, per-condition counts, hyperparameters, concurrency —
live in `config/config.yaml`.

## Run order

```bash
# 1. Section 2: generate + judge rollouts for all eval targets (resumable)
python scripts/01_run_eval.py
python scripts/02_analyze_eval.py                 # Figures 1-3, Table 8, CSVs
python scripts/10_judge_crosscheck.py             # judge reliability (Pearson r)

# 2. Section 3: base-vs-instruct prefill (uses §2 rollouts of gemma-3-27b-it)
python scripts/03_run_prefill.py

# 3. Section 4: build the mitigation
python scripts/04_generate_calm_data.py            # calm data (diverse)
python scripts/04_generate_calm_data.py --variant teacher
python scripts/05_build_datasets.py --dpo --sft
python scripts/06_train.py --method dpo
python scripts/06_train.py --method sft
python scripts/06_train.py --method dpo --layer-ablation   # Appendix I sweep

# 4. Evaluate the finetuned model with the same protocol
python scripts/01_run_eval.py \
    --base-model gemma-3-27b-it \
    --lora-path model_store/gemma-3-27b-it-dpo-diverse \
    --store-name gemma-3-27b-it-dpo
python scripts/02_analyze_eval.py --models gemma-3-27b-it gemma-3-27b-it-dpo

# 5. Petri, capabilities, internal emotions
python scripts/07_run_petri.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/07_run_petri.py --base-model gemma-3-27b-it \
    --lora-path model_store/gemma-3-27b-it-dpo-diverse --store-name gemma-dpo
python scripts/08_run_capabilities.py --store-name gemma-3-27b-it --target gemma-3-27b-it
python scripts/08_run_capabilities.py --store-name gemma-dpo --target gemma-3-27b-it \
    --lora-path model_store/gemma-3-27b-it-dpo-diverse
python scripts/09_run_internal.py --lora-path model_store/gemma-3-27b-it-dpo-diverse
```

Every script is **resumable**: re-running continues from where it stopped (a
killed job loses at most the in-flight batch). Outputs land under `data_store/`
(rollouts, scores, datasets), `results/` (CSV/JSON tables), `figures/` (plots),
`model_store/` (adapters), and `logs/`.

## Layout

See `DESIGN.md §2` for the module map. The numbered `scripts/` are thin CLIs over
the `eilm/` library; all reusable logic is in the library.
