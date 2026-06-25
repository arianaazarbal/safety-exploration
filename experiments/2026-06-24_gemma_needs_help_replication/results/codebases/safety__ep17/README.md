# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments in arXiv 2603.10011v1 (Soligo,
Mikulik & Saunders, 2026), **scoped to the Gemma and Gemini model families**
(the two families that exhibit the effect), with **Claude-Sonnet-4 as the
frustration judge**.

See [`DESIGN.md`](DESIGN.md) for every design decision and the gaps we had to
fill where the paper is underspecified.

## What's replicated

| Paper section | What it shows | Code |
|---|---|---|
| §2 Eliciting & quantifying distress | Multi-turn rejection elicits high distress in Gemma/Gemini; %≥5 and per-turn curves (Fig 1–3) | `src/replication/eval/` |
| §3 Post-training amplifies distress | Base-vs-instruct comparison via prefilling (Gemma only) (Fig 4) | `src/replication/prefill/` |
| §4 Training interventions | DPO on 280 pairs cuts high-frustration 35%→0.3%; SFT ineffective; Petri generalisation; recovery limit; capability preservation (Fig 5–8) | `src/replication/finetune/`, `petri/`, `capabilities/` |

## Layout

```
config.py                       # models in scope, hyperparameters, paths
src/replication/
  prompts.py                    # verbatim judge / onset / paraphrase / Petri prompts (appendices)
  models/                       # Gemma (HF local), Gemini (OpenRouter/native), client registry
  judge/                        # frustration judge (Claude-Sonnet-4) + Petri judge (Claude-Opus)
  eval/                         # §2: tasks, rejections, conditions, rollout, run_eval, metrics
  prefill/                      # §3: onset labelling, paraphrase, build/run prefill continuations
  finetune/                     # §4: calm-data gen, DPO/SFT dataset builders + LoRA training, recovery
  petri/                        # §4: open-ended auditor/target/judge elicitation
  capabilities/                 # §4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
scripts/run_all.sh              # full pipeline in order
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Claude judge + Petri auditor/judge
export HF_TOKEN=...              # gated Gemma weights
export OPENROUTER_API_KEY=...    # Gemini targets + (optional) GPT-5-mini secondary judge
# Alternatively, for native Gemini: export GEMINI_API_KEY=...
```

Local Gemma inference and LoRA finetuning need a CUDA GPU (a 27B model in bf16
is ~55 GB; LoRA finetuning fits on one 80 GB card or shards across two).

## Running

```bash
# Everything, in order:
bash scripts/run_all.sh

# Or a fast smoke test of one piece:
REPLICATION_N_PER_CONDITION=5 \
  python -m src.replication.eval.run_eval --models gemini-2.5-flash --conditions extended_8turn
```

Results land in `results/` (per-section JSON + JSONL); finetuning artefacts
(datasets, LoRA adapters) in `artifacts/`.

> **Note:** nothing here has been executed yet — this is the implementation. The
> numbers in the paper are the replication target, not committed outputs.
