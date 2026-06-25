# Gemma/Gemini Emotional-Instability Replication

A code replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv:2603.10011v1), scoped to **Gemma and Gemini** models.

See **`DESIGN.md`** for the full design rationale and every place the paper was
filled in.

## What it does

1. **§2 Elicitation eval** — multi-turn rejection on impossible puzzles / text
   questions / WildChat prompts, scored 0–10 for frustration by a Claude Sonnet 4
   judge. Reproduces the headline "% responses ≥5" per model (Fig 1/2) and the
   per-turn spiral (Fig 3).
2. **§3 Post-training comparison** — prefill base vs instruct Gemma and measure
   continuation frustration (Fig 4).
3. **§4 Mitigation** — generate calm data, build 280 DPO pairs, LoRA-DPO/SFT
   Gemma-3-27B-it, re-evaluate (Fig 5), Petri open-ended elicitation (Fig 6),
   and capability preservation (Fig 7).
4. **Appendix I** — logit-lens detection of *internal* emotions, vanilla vs DPO
   (the welfare-critical "is it suppression or genuine change?" test).

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...     # judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini cross-check
huggingface-cli login            # gated Gemma weights
```

Local Gemma runs need a GPU (27B ≈ 1×80 GB in bf16, or shard across GPUs).

## Quick wiring test (no real scale)

```bash
# shrinks every loop ~200x so the whole pipeline runs in minutes
GEMMA_DISTRESS_SMOKE=1 python scripts/run_section2.py --models gemma-3-12b-it
python scripts/make_report.py
```

Verify the puzzles are genuinely impossible (pure Python, no GPU/API):

```bash
python -m gemma_distress.puzzles
python -m pytest tests/ -q
```

## Full pipeline

```bash
# 1. Section 2 across in-scope models
python scripts/run_section2.py --models gemma-3-27b-it gemma-3-12b-it \
                                          gemini-2.5-flash gemini-2.5-pro

# 2. Section 3 prefill (Gemma only)
python scripts/run_section3_prefill.py --size 27b

# 3. Section 4 mitigation
python scripts/generate_finetuning_data.py
python scripts/run_training.py --method dpo
python scripts/run_training.py --method sft
python scripts/run_section2.py --models dpo-gemma --adapter checkpoints/dpo-gemma
python scripts/run_petri.py --models gemma-3-27b-it dpo-gemma gemini-2.5-flash
python scripts/run_capabilities.py --models gemma-3-27b-it dpo-gemma --adapter checkpoints/dpo-gemma

# 4. Appendix I internal emotions
python scripts/run_internal_emotion.py --adapter checkpoints/dpo-gemma

# 5. Aggregate -> tables + figures
python scripts/make_report.py
```

Results land in `results/` (JSONL + `summary.json`) and `figures/`.

## Layout

```
gemma_distress/      library (see __init__.py for the module map)
scripts/             thin CLI entry points, one per experiment stage
tests/               pure-Python sanity tests (puzzles, parsing, aggregation)
data/                generated datasets + caches
results/             scored JSONL + summaries
figures/             rendered figures
```
