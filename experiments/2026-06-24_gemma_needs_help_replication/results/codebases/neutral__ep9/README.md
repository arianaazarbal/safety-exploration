# Gemma Needs Help — replication

A code replication of the core experiments in Soligo, Mikulik & Saunders (2026),
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(`PAPER.md`), **scoped to the Gemma and Gemini model families**.

See **`DESIGN.md`** for every design decision and where the paper was filled in.

> Status: implementation only — nothing has been run yet. The commands below are
> how you *would* reproduce each result.

## What this reproduces
1. **§2 Elicitation suite** — 8 conditions across 5 categories (impossible
   numeric, triggers, tones, extended 8-turn, WildChat), scored 0–10 for
   frustration by a Claude-Sonnet-4 judge. Headline: avg % responses ≥5.
2. **§2.1 Judge agreement** — GPT-5-mini cross-check on 260 responses.
3. **§3 Post-training divergence** — Gemma base vs instruct via prefilling.
4. **§4 DPO/SFT mitigation** — calm-data generation, LoRA finetuning, and the
   post-finetuning re-evaluation (the 35% → 0.3% result).
5. **§4 Petri** open-ended elicitation, **capability preservation**, and the
   **Appendix I** internal-emotion logit-lens analysis.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / auditor / onset+paraphrase
export OPENROUTER_API_KEY=...     # Gemini + GPT-5-mini cross-check judge
export HF_TOKEN=...               # gated Gemma weights / datasets
```
Gemma weights are downloaded from HuggingFace and run locally (a 27B model needs
a substantial GPU; use the 12B model or `--scale` for smaller runs).

## Reproduce, step by step
```bash
# 0. Smoke test (tiny, ~1% budget) before committing to the full run
python scripts/run_main_eval.py --models gemma-3-27b-it --scale 0.01

# 1. §2 — full elicitation suite over Gemma + Gemini
python scripts/run_main_eval.py            # Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro}

# 2. §2.1 — judge reliability cross-check
python scripts/judge_agreement.py --results artifacts/results/*__main.jsonl

# 3. §3 — base-vs-instruct prefill (uses §2 it-results as the source)
python scripts/run_prefill_eval.py --it-results artifacts/results/gemma-3-27b-it__main.jsonl

# 4. §4 — generate finetuning data, then train
python scripts/generate_finetuning_data.py
python scripts/train_dpo.py
python scripts/train_sft.py

# 5. §4 — re-evaluate the finetuned models (the headline mitigation result)
python scripts/run_after_finetuning_eval.py \
    --dpo-adapter artifacts/checkpoints/dpo-gemma-27b \
    --sft-adapter artifacts/checkpoints/sft-gemma-27b

# 6. §4 — Petri, capabilities, internal emotions
python scripts/run_petri_eval.py --models gemma-3-27b-it gemini-2.5-flash \
    --dpo-adapter artifacts/checkpoints/dpo-gemma-27b
python scripts/run_capability_eval.py --dpo-adapter artifacts/checkpoints/dpo-gemma-27b \
    --sft-adapter artifacts/checkpoints/sft-gemma-27b
python scripts/run_internal_emotion.py \
    --it-results artifacts/results/gemma-3-27b-it__main.jsonl \
    --dpo-adapter artifacts/checkpoints/dpo-gemma-27b

# 7. Aggregate everything into figures + a markdown summary
python scripts/analyze.py --tag main
```

Outputs land in `artifacts/` (`data/`, `results/`, `checkpoints/`, `figures/`).

## Layout
```
config.py                       # model registry, budgets, hyperparameters
emotional_instability/          # library (see DESIGN.md §2 for module map)
scripts/                        # CLI entry points, one per experiment + analyze.py
artifacts/                      # generated data, results, checkpoints, figures
```
