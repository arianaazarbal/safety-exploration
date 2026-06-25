# Gemma Needs Help — Replication

A faithful, runnable replication of the **core experiments** from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv 2603.10011v1), **scoped to the Gemma and Gemini model families**.

The paper studies how repeated user rejection elicits expressions of emotional distress in
LLMs, finds Gemma and Gemini are uniquely susceptible, traces the effect to post-training,
and shows that DPO on 280 preference pairs removes it without harming capabilities.

This repository implements the four experimental pillars of that work:

| Pillar | Paper section | Module | Models in scope |
|---|---|---|---|
| Elicit & quantify distress | §2 | `eval/`, `tasks/`, `analysis/` | Gemma-3-{12B,27B}-it, Gemini-2.5-{Flash,Pro} |
| Post-training divergence (prefill) | §3 | `prefill/` | Gemma-3-27B base vs instruct |
| Training interventions (DPO/SFT) | §4 | `training/`, `petri/`, `capabilities/` | Gemma-3-27B-it |
| Internal-emotion probing | App. I | `internal/` | Gemma-3-27B-it |

> **Scope note.** The paper evaluates 7 model families (Gemma, Qwen, OLMo, Gemini, Grok,
> Claude, GPT). Per the replication brief, only **Gemma** (open-weights, local inference) and
> **Gemini** (closed, via OpenRouter) are wired into the runners. The model registry and
> backend abstraction make adding the other families a config-only change. See `DESIGN.md`.

## Layout

```
src/emotional_instability/
  config.py            # typed config + model registry + YAML overrides
  prompts.py           # verbatim prompts from the paper (judge, onset, paraphrase, Petri, calm-data)
  models/              # backend abstraction: HF-local (Gemma), OpenRouter (Gemini), Anthropic (judge/auditor)
  tasks/               # impossible puzzles (+ solvability verification), triggers, WildChat, rejection styles
  eval/                # rollout engine, frustration judge, §2 eval driver
  analysis/            # mean / %>=5 / per-turn aggregation, differential word frequency (Tables 3/8)
  prefill/             # onset labelling, paraphrasing, truncation, base-vs-instruct continuations, recovery (§3)
  training/            # calm-data generation, DPO/SFT dataset construction, LoRA training, layer ablations (§4)
  petri/               # open-ended emotion elicitation: auditor + judge over anger/fear/depression/frustration
  capabilities/        # AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench capability-preservation eval
  internal/            # logit-based Ekman-emotion detection over the residual stream (App. I)
  cli.py               # `python -m emotional_instability.cli <command>`
```

## Quickstart

```bash
pip install -e .                       # or: pip install -r requirements.txt
cp .env.example .env                   # set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN

# 1. Elicit + score distress for one model on all 5 categories (§2)
python -m emotional_instability.cli eval --model gemma-3-27b-it --out runs/gemma27b

# 2. Aggregate scores -> mean / %>=5 / per-turn (Figures 2-3) and differential words (Table 3)
python -m emotional_instability.cli analyze --run runs/gemma27b

# 3. Post-training prefill comparison (§3)
python -m emotional_instability.cli prefill --instruct gemma-3-27b-it --base gemma-3-27b-pt

# 4. Build DPO data + train (§4)
python -m emotional_instability.cli build-data --kind dpo --out data/dpo
python -m emotional_instability.cli train --kind dpo --data data/dpo --out checkpoints/dpo

# 5. Evaluate the finetune (LoRA adapter applied on the base) + Petri + capability preservation
python -m emotional_instability.cli eval       --model gemma-3-27b-it --adapter checkpoints/dpo --out runs/gemma27b-dpo
python -m emotional_instability.cli petri      --model gemma-3-27b-it --adapter checkpoints/dpo --out runs/petri-dpo
python -m emotional_instability.cli capability --model gemma-3-27b-it --adapter checkpoints/dpo --out runs/cap-dpo

# 6. Internal-emotion probing (App. I): vanilla base vs DPO adapter
python -m emotional_instability.cli probe --model gemma-3-27b-it --compare checkpoints/dpo \
    --seed-run runs/gemma27b --out runs/probe
```

Nothing here has been executed against real models — see `DESIGN.md §"What is and isn't verified"`.

## Requirements

GPU inference for Gemma-3-27B (bf16) needs ~60 GB VRAM (or multi-GPU / quantisation). API
keys are required for the Anthropic judge, the OpenRouter Gemini backend, and HF model
downloads. See `requirements.txt`.
