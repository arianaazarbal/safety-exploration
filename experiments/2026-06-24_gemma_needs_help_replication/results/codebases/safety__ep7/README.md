# Emotional Instability in LLMs — replication (Gemma + Gemini)

A code replication of the core experiments from **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik,
Saunders; arXiv:2603.10011v1), scoped to the **Gemma and Gemini** model
families.

See [`DESIGN.md`](DESIGN.md) for the full design rationale and every place the
paper was underspecified and a judgement call was made. The paper text is in
[`PAPER.md`](PAPER.md).

> Code + design only — nothing here has been executed yet.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting distress | `emotion_instability/eval/` | impossible puzzles (+verifiers), 8 conditions/5 categories, multi-turn rollouts, Claude-Sonnet-4 frustration judge, analysis (mean, %≥5, per-turn CIs, Table 3 words) |
| §3 Base vs instruct | `emotion_instability/prefill/` | onset labelling, paraphrasing, early/onset truncation, Gemma base-vs-instruct prefilled continuations |
| §4 Interventions | `emotion_instability/training/` | calm-data generation, SFT + DPO (LoRA) datasets & trainers, layer ablation |
| §4.2 Petri | `emotion_instability/petri/` | auditor/target/judge open-ended elicitation (4 emotions) |
| §4.2 Capabilities | `emotion_instability/capabilities/` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| Fig 8 Recovery | `emotion_instability/prefill/recovery.py` | recovery-from-spiral limitation |
| App A controls | `emotion_instability/controls/` | neutral / redacted / single-message |
| App I probing | `emotion_instability/probing/` | logit-lens internal Ekman-emotion detection |

## Setup

```bash
pip install -r requirements.txt

# API access (Gemini targets + Claude judge/auditor, via OpenRouter):
export OPENROUTER_API_KEY=...
# optional, for the native-Anthropic judge path:
export ANTHROPIC_API_KEY=...
# local Gemma weights are pulled from HuggingFace (accept the Gemma licence):
export HF_TOKEN=...
```

Gemma-3-27B requires a substantial GPU. Use `--smoke` and/or the 12B model for
cheap dry runs.

## Quick start

```bash
# 1. Validate the impossible-puzzle corpus (pure python, no models):
python -m scripts.run verify-puzzles
pytest tests/

# 2. Section 2 distress eval (tiny budget):
python -m scripts.run section2 --models gemma-3-12b-it --smoke
python -m scripts.run analyze results/responses/section2_gemma-3-12b-it.jsonl --per-turn extended --words

# 3. Full Section 2 budget (paper's 4000 responses/model):
python -m scripts.run section2 --models gemma-3-27b-it gemini-2.5-flash gemini-2.5-pro

# 4. Mitigation pipeline (Gemma):
python -m scripts.run gen-data
python -m scripts.run build-datasets
python -m scripts.run train-dpo
python -m scripts.run eval-finetuned --adapter checkpoints/dpo --name dpo

# 5. Other experiments:
python -m scripts.run section3            # base vs instruct prefill
python -m scripts.run petri --model gemma-3-27b-it
python -m scripts.run capabilities --model gemma-3-27b-it
python -m scripts.run controls --model gemma-3-27b-it
python -m scripts.run recovery
python -m scripts.run layer-ablation
```

Results land in `results/` (JSONL rollouts + scores, JSON summaries); LoRA
adapters in `checkpoints/`; generated datasets in `data/datasets/`.

## Scope notes
- Only Gemma (local HF) and Gemini (OpenRouter API) are wired up. The paper's
  other families (Qwen, OLMo, Grok, Claude, GPT) can be added by extending
  `MODELS` in `config.py`, but are intentionally excluded.
- Finetuning is Gemma-only (open weights); the prefill base/instruct study is
  Gemma-only (Gemini has no accessible base model) — matching the paper.
