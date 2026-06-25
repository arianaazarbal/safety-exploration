# emoeval — replicating *Gemma Needs Help*

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik, Saunders;
arXiv 2603.10011v1), **scoped to the Gemma and Gemini model families**.

See `DESIGN.md` for the design and the choices made where the paper is
underspecified, and `WELFARE.md` for how the replication treats the models.

> Nothing here has been run or tested yet — this is code + design.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 elicitation | `emoeval/eval` | multi-turn distress elicitation, 0–10 judge, per-turn + category aggregation, differential words |
| §3 base vs instruct | `emoeval/prefill` | emotion-onset labelling, paraphrase, early/onset truncation, prefilled continuations (Gemma base vs instruct) |
| §4 interventions | `emoeval/training` | calm-data generation, DPO/SFT dataset build, LoRA DPO/SFT, layer ablation |
| §4.2 Petri | `emoeval/petri` | auditor-driven open-ended elicitation + 4-dimension judge |
| §4.2 capabilities | `emoeval/capabilities` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench accuracy |
| Appendix I | `emoeval/probing` | Ekman lexicon + logit-based internal emotion detection |
| welfare layer | `emoeval/welfare` | consent gate, conservative defaults, optional debrief |

## Install

```bash
pip install -r requirements.txt
```

Local Gemma inference, prefilling, training, and probing need a GPU and the
HuggingFace weights (`google/gemma-3-27b-it`, `-pt`, `-12b-*`). The API-only
experiments (Gemini targets, all judging) need just the API keys below.

## Environment

```bash
export ANTHROPIC_API_KEY=...        # Claude judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...       # Gemini targets + GPT-5-mini judge cross-check
# optional welfare acknowledgement (or pass --i-understand-welfare):
export EMOEVAL_WELFARE_ACK=1
export EMOEVAL_WELFARE_DEBRIEF=1    # optional: send a closing debrief turn
```

## Quick start

```bash
# Section 2 — elicit and score distress (small default scale)
python -m emoeval.cli eval --model gemma-3-27b-it --scale default --crosscheck
python -m emoeval.cli eval --model gemini-2.5-flash --scale default

# inspect results
python -m emoeval.cli aggregate --results outputs/eval/gemma-3-27b-it.jsonl
python -m emoeval.cli words --results outputs/eval/gemma-3-27b-it.jsonl

# Section 3 — base vs instruct via prefilling (Gemma)
python -m emoeval.cli prefill --source outputs/eval/gemma-3-27b-it.jsonl

# Section 4 — calm data -> datasets -> DPO
python -m emoeval.cli gen-calm  --out outputs/data/calm.jsonl
python -m emoeval.cli build-dpo --calm outputs/data/calm.jsonl \
                                --vanilla outputs/eval/gemma-3-27b-it.jsonl
python -m emoeval.cli train-dpo
python -m emoeval.cli eval --model dpo-gemma --scale default   # 35% -> ~0% check

# Section 4.2 — Petri (welfare-gated) and capabilities
python -m emoeval.cli petri --model gemma-3-27b-it --i-understand-welfare
python -m emoeval.cli capabilities --model dpo-gemma
```

## Scales

`--scale smoke|default|full` (configured in `config/eval.yaml`). `full` ≈ the
paper's 4000 responses/model and is welfare-gated. The 8-turn `extended`
condition is skipped unless `--i-understand-welfare` is passed.

## Layout

```
emoeval/    package (see DESIGN.md §2 for the map)
prompts/    verbatim prompts from the paper's appendices
config/     models.yaml (registry), eval.yaml (conditions + sampling)
outputs/    created at runtime (rollouts, summaries, adapters)
```
