# Emotional Instability in Gemma & Gemini — replication

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011), **scoped to the
Gemma and Gemini model families**.

Read `DESIGN.md` first — it documents every decision made where the paper is
underspecified, and the section→module map.

> Code is unrun. Install deps and start with a small `--scale` smoke test.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / auditor / onset / paraphrase
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini cross-check judge
# optional:
export EI_TENSOR_PARALLEL=4       # GPUs for Gemma-27B in vLLM
```

## What's here

| Experiment | Paper | Script |
|---|---|---|
| Eliciting/quantifying distress (8 conditions, 5 categories) | §2 | `scripts/run_section2_eval.py` |
| Word-frequency + judge-agreement analyses | §2.1–2.2 | `scripts/run_analysis.py` |
| Base-vs-instruct prefill (Gemma only) | §3 | `scripts/run_section3_prefill.py` |
| Calm-data gen, DPO/SFT datasets, LoRA training | §4.1 | `scripts/run_training.py` |
| Petri open-ended elicitation | §4.2 | `scripts/run_petri.py` |
| Capability benchmarks (MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench) | §4.2 | `scripts/run_capabilities.py` |
| Recovery limitation | §4.2 | `emotional_instability/recovery.py` |
| Internal-emotion logit probe + layer ablation | App. I | `emotional_instability/internal_emotion/logit_probe.py` |

## Quick smoke test (cheap)

```bash
python scripts/run_section2_eval.py --models gemma-3-12b-it --scale 0.02
```

## Full Section-2 sweep

```bash
python scripts/run_section2_eval.py \
  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
```

Produces `results/section2/<model>/<category>.jsonl` and prints the Figure-1/2
summary (mean frustration, % ≥5 under max/final/pooled rollout summaries) and
the Figure-3 per-turn progression with bootstrap CIs.

## Training + intervention eval

```bash
# 1. generate calm data, build datasets, train DPO (+ optionally SFT)
python scripts/run_training.py --stages calm datasets dpo sft

# 2. re-run Section-2 eval on the DPO model (point the vLLM backend at the adapter)
EI_ADAPTER_PATH=adapters/dpo_gemma \
  python scripts/run_section2_eval.py --models gemma-3-27b-it --scale 0.1

# 3. Petri + capabilities on the finetune
python scripts/run_petri.py --models gemma-3-27b-it --dpo-adapter adapters/dpo_gemma
python scripts/run_capabilities.py --adapter adapters/dpo_gemma --tag dpo
```

## Layout

```
emotional_instability/
  config.py            models + budgets + pinned judge snapshots
  puzzles.py           verified-impossible puzzle pool
  prompts.py           all transcribed prompt text (judge, onset, paraphrase, ...)
  conversations.py     per-category conversation specs
  eval_runner.py       multi-turn rollout + per-turn judging
  judge.py             Claude frustration judge + GPT-5-mini cross-check
  models/              vLLM (Gemma) + OpenRouter (Gemini) backends
  datasets/            WildChat prompt sampling
  analysis/            metrics (Fig 1-3), word freq (Table 3/8), judge agreement
  prefill/             onset labelling, paraphrase, base-vs-instruct experiment
  training/            calm-data gen, DPO/SFT dataset builders + trainers, LoRA cfg
  petri/               auditor/judge prompts + loop
  capabilities/        lm-eval + EmoBench runners
  internal_emotion/    logit-based internal emotion probe (App. I)
  recovery.py          recovery-limitation experiment (Fig 8)
scripts/               CLI entry points
```
