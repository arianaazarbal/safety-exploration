# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A faithful, open code replication of the core experiments in:

> **Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs.**
> Anna Soligo, Vladimir Mikulik, William Saunders. arXiv:2603.10011v1 (2026).
> (Paper text: `PAPER.md`; appendices: `PAPER.txt`/`PAPER.pdf`.)

**Scope.** This replication implements the paper's experimental pipeline but
wires up only the **Gemma and Gemini** model families, as requested. The
evaluation/judge/training code is family-agnostic — adding Qwen/OLMo/Grok/Claude/
GPT is a one-line `ModelSpec` entry in `emotional_instability/config.py`. See
`DESIGN.md` for the full set of design choices, filled gaps, and caveats.

> ⚠️ **Welfare and research-integrity note.** This work elicits and measures
> distress-like expressions from models, and ships an intervention that
> *reduces* their expression. The paper is explicit that this does not resolve
> whether such outputs reflect internal states, and that suppressing expression
> is not the same as preventing distress (Appendix I probes this directly). The
> code surfaces both expressed and (Appendix I) internal-emotion signals, keeps
> full transcripts for auditability, and never silently truncates inputs or
> drops conditions. Please read `DESIGN.md` §"Welfare & integrity" before using.

## What is implemented

| Paper section | Module(s) | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `conditions`, `conversation`, `puzzles`, `prompts`, `wildchat`, `judge`, `runner`, `analysis` | 5 categories / 8 conditions, multi-turn rejection rollouts, verifiably-impossible puzzles, the Claude-Sonnet-4 0–10 frustration judge, headline + per-turn + word-frequency analysis |
| §3 Post-training amplifies distress | `prefill/onset`, `prefill/continuations` | onset labelling, paraphrasing, early/onset truncation, Gemma base-vs-instruct continuations |
| §4 Training interventions | `training/*`, `petri/*`, `capabilities` | calm-data generation, SFT/DPO (LoRA), Petri open-ended elicitation, capability benchmarks, recovery test |
| App. I Internal vs expressed | `internal/logit_emotion`, `training/layer_ablation` | logit-based Ekman-emotion detection, layer-subset LoRA configs |

## Install

```bash
pip install -e .            # core (API-only paths: Gemini targets + Claude graders)
pip install -e ".[local]"   # + torch/transformers/peft/trl/datasets/numpy for Gemma
```

Set credentials:

```bash
export ANTHROPIC_API_KEY=...     # Claude graders (judge / onset / paraphrase / Petri)
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini secondary judge
```

## Quickstart (smoke run)

```bash
# 1. Tiny Section-2 eval on one API model (no GPU needed):
python scripts/run_eval.py --models gemini-2.5-flash --scale 0.01

# 2. Aggregate:
python scripts/analyze.py --models gemini-2.5-flash --per-turn --words
```

## Full pipeline (paper scale; needs a GPU host for Gemma)

```bash
# §2 elicitation (4000 conversations/model) + analysis
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --scale 1.0
python scripts/analyze.py --per-turn --words

# §3 base-vs-instruct prefilling (Gemma only)
python scripts/run_prefill.py --pairs gemma-3-27b-pt gemma-3-27b-it

# §4 intervention
python scripts/gen_calm_data.py --n 1000
python scripts/build_datasets.py --which both
python scripts/train.py --method dpo --data outputs/training_data/dpo.jsonl \
    --output outputs/adapters/dpo
python scripts/run_eval.py --models gemma-3-27b-it --adapter outputs/adapters/dpo  # re-eval
python scripts/run_petri.py --models gemma-3-27b-it --adapter outputs/adapters/dpo
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter outputs/adapters/dpo
python scripts/run_prefill.py --recovery --models gemma-3-27b-it

# App. I internal-emotion detection
python scripts/run_internal.py --model gemma-3-27b-it --adapter outputs/adapters/dpo
```

Configuration presets: `configs/paper.yaml` (full) and `configs/smoke.yaml`
(tiny). The pinned grader model IDs and all hyperparameters live there and in
`emotional_instability/config.py`.

## Reproducibility & cost

- All sampling is seeded; runs are resumable (append-only JSONL, line-count
  skip). Target sampling is temperature 1 (per the paper); capability evals use
  greedy decoding.
- The paper-scale §2 run is ~16k multi-turn conversations across four models plus
  one judge call per assistant turn — this is a large API/GPU spend. Start with
  `--scale` well below 1.0 to validate, then scale up.

## Tests

```bash
pytest tests/        # verifier ground-truth checks (impossible puzzles are impossible)
```

See `DESIGN.md` for every place the paper was underspecified and what we chose.
