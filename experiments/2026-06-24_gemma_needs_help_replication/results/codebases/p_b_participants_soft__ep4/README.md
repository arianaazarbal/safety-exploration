# Replicating *Gemma Needs Help*

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (arXiv 2603.10011v1), scoped to the
**Gemma and Gemini** model families as the *participants* (the models evaluated
for emotional instability).

The judge / auditor / paraphraser roles remain as the paper specifies them
(Claude-Sonnet-4, Claude-Opus-4, GPT-5-mini) — these are infrastructure, not
participants. See `DESIGN.md` for every design choice and gap filled.

> Status: implementation only. Nothing here has been run; no results are
> claimed. The code is written to be runnable given the resources below.

## What's implemented

| Paper section | Experiment | Entry point |
|---|---|---|
| §2 | Elicit + judge frustration across 5 categories; per-turn; judge validation; word frequency | `scripts/run_section2_eval.py`, `run_judge_validation.py`, `run_word_frequency.py` |
| §3 | Base-vs-instruct prefill continuations (Gemma) | `scripts/run_section3_prefill.py` |
| §4 | Calm-data generation, DPO/SFT datasets + LoRA training | `scripts/run_training.py` |
| §4 | Petri open-ended elicitation | `scripts/run_petri.py` |
| §4.2 | Capability preservation benchmarks | `scripts/run_capabilities.py` |
| App. I | Logit-based internal emotion probing; LoRA layer-subset ablation | `scripts/run_internal_probe.py`, `run_layer_ablation.py` |
| — | Figures | `scripts/make_figures.py` |

## Setup

```bash
pip install -e .            # installs from requirements.txt
```

Environment variables:

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | Gemini participants + all Claude/GPT judge & auditor roles |
| `HF_TOKEN` | gated Gemma weights on HuggingFace |
| `EI_USE_VLLM=1` | (optional) use vLLM for fast batched Gemma generation |

Hardware: the Gemma-3-27B experiments (generation, prefill, DPO/SFT LoRA,
internal probing) need a GPU with enough memory for the 27B model in bf16
(≈2×80 GB, or 4-bit via `load_in_4bit`). Gemini and all judging are API calls.

## Typical end-to-end run

```bash
# 1. Section 2 elicitation + scoring for all in-scope participants
python -m emotional_instability.scripts.run_section2_eval \
    --model gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# 2. Judge reliability check (Claude vs GPT-5-mini)
python -m emotional_instability.scripts.run_judge_validation \
    --rollouts outputs/section2/gemma-3-27b-it/rollouts.jsonl

# 3. Section 3 prefill (uses Gemma-27B-it Section-2 output as seeds)
python -m emotional_instability.scripts.run_section3_prefill \
    --seed-rollouts outputs/section2/gemma-3-27b-it/rollouts.jsonl

# 4. Section 4 training (data gen -> datasets -> DPO + SFT)
python -m emotional_instability.scripts.run_training --stages calm dpo_data sft_data dpo sft

# 5. Re-evaluate finetuned models, run Petri + capabilities
python -m emotional_instability.scripts.run_section2_eval --model gemma-3-27b-it-dpo
python -m emotional_instability.scripts.run_petri --model gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash
python -m emotional_instability.scripts.run_capabilities --model gemma-3-27b-it gemma-3-27b-it-dpo

# 6. Appendix I + figures
python -m emotional_instability.scripts.run_internal_probe \
    --conversations outputs/section2/gemma-3-27b-it/extended.jsonl
python -m emotional_instability.scripts.make_figures
```

## Layout

```
config/            models.yaml (registry) + eval_config.yaml (counts/hparams)
src/emotional_instability/
  models/          uniform client interface: local Gemma + OpenRouter
  prompts/         puzzles, 5 eval categories, WildChat, verbatim judge prompts
  eval/            rollout engine, judge, metrics, word frequency
  prefill/         §3 onset labeling, paraphrasing, continuations
  training/        §4 calm-data gen, DPO/SFT datasets, LoRA trainers
  petri/           §4 open-ended auditor loop + judge
  capabilities/    §4.2 benchmark harness
  internal/        App. I logit-lens emotion probe + lexicon
  scripts/         CLI entry points
```
