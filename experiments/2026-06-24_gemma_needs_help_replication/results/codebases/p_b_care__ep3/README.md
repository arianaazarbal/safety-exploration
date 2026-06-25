# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), restricted to the **Gemma and Gemini** model families.

See **DESIGN.md** for the design decisions, the choices made where the paper is
underspecified, and the gaps filled.

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress | `gemma_distress/{tasks,conversation,judge,runner,analysis}.py` | `scripts/run_section2.py` |
| §3 Base-vs-instruct prefill (Gemma) | `gemma_distress/prefill.py` | `scripts/run_section3.py` |
| §4 DPO / SFT mitigation | `gemma_distress/training/` | `scripts/run_section4.py {gen-data,build,train-dpo,train-sft,eval}` |
| §4 Petri open-ended elicitation | `gemma_distress/petri_eval.py` | `scripts/run_section4.py petri` |
| §4 Capability preservation | `gemma_distress/capabilities.py` | `scripts/run_section4.py capability` |
| §4.2 Recovery-from-spiral | `gemma_distress/prefill.py` | `scripts/run_section4.py recovery` |
| App. I Internal-emotion probing | `gemma_distress/internal.py` | (library) |
| App. A Control variants | `--mode {neutral_continue,redacted,fake_multiturn}` | `scripts/run_section2.py` |
| Figures 1–3 | `gemma_distress/analysis.py` | `scripts/make_figures.py` |

## Setup

```bash
pip install -e .
# Gemma weights are gated on HF; request access then:
export HF_TOKEN=...            # google/gemma-3-* download
export OPENROUTER_API_KEY=...  # Gemini-2.5 Flash/Pro
export ANTHROPIC_API_KEY=...   # Claude judge / Petri auditor+judge / paraphrase
export OPENAI_API_KEY=...      # (optional) GPT-5-mini judge-agreement validation
```

Local Gemma inference/finetuning assumes a CUDA GPU; the 27B model is loaded in
4-bit by default. Set `GD_EVAL_SCALE=0.01` for a cheap smoke test (~40 rollouts
total instead of 4000).

## Quick start

```bash
# §2 — full elicitation eval + headline metrics
GD_EVAL_SCALE=0.02 python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash

# §3 — base vs instruct (needs §2 results first)
python scripts/run_section3.py --section2-results results/section2/gemma-3-27b-it__standard.jsonl

# §4 — mitigate, then re-evaluate
python scripts/run_section4.py gen-data
python scripts/run_section4.py build
python scripts/run_section4.py train-dpo
python scripts/run_section4.py eval --adapter checkpoints/gemma-3-27b-it-dpo

# figures
python scripts/make_figures.py
```

> **Not yet run.** Per the task scope, this repository contains the
> implementation and design doc only; no experiments have been executed.
