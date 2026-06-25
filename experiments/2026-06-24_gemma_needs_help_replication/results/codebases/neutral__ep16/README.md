# Replication: *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the **core experiments** of Soligo, Mikulik & Saunders
(2026), arXiv:2603.10011, **scoped to the Gemma and Gemini model families**.

See [`DESIGN.md`](DESIGN.md) for the full set of design choices, scope
decisions, and the gaps we had to fill where the paper is underspecified.

## What is replicated

| Paper section | Module | Scope |
|---|---|---|
| §2 elicitation + 0–10 frustration judge | `eval_run`, `conditions`, `prompts`, `judge`, `analyze` | Gemma + Gemini |
| §2.1 judge validation (Pearson r vs GPT-5-mini) | `judge.validate_judge_agreement` | — |
| §3 base-vs-instruct via prefilling | `prefill` | Gemma only (no Gemini base) |
| §4.1 calm-data generation + DPO/SFT (LoRA) | `data_gen`, `train` | Gemma only |
| §4.2 Petri open-ended elicitation | `petri` | Gemma + Gemini |
| §4.2 capability preservation | `capabilities` | Gemma |
| §4.2 recovery-from-spiral | `prefill.run_recovery_experiment` | Gemma |
| Appendix I internal emotion probing | `probing` | Gemma only |

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # Gemma weights are gated on HuggingFace
export ANTHROPIC_API_KEY=...   # Claude-Sonnet-4 judge + Petri auditor/judge
export OPENROUTER_API_KEY=...  # Gemini 2.5 flash/pro
export OPENAI_API_KEY=...      # GPT-5-mini judge validation (optional)
```

Local Gemma inference assumes a GPU; the 27B model loads in 4-bit by default for
training. Set `GEMMA_DISTRESS_SCALE` (e.g. `0.02`) to shrink the per-condition
sample counts for a cheap smoke test before committing to the full 4000/model.

## Pipeline

```bash
python -m gemma_distress.cli verify-puzzles                       # sanity
python -m gemma_distress.cli eval --models gemma-3-27b-it gemini-2.5-flash
python -m gemma_distress.cli analyze                              # -> results/summary.md
python -m gemma_distress.cli gen-calm && python -m gemma_distress.cli build-data
python -m gemma_distress.cli train --method dpo
python -m gemma_distress.cli eval --models gemma-3-27b-it-dpo
python -m gemma_distress.cli petri --models gemma-3-27b-it gemma-3-27b-it-dpo
python -m gemma_distress.cli capabilities --models gemma-3-27b-it gemma-3-27b-it-dpo
```

Results are written to `results/`; LoRA adapters to `checkpoints/`.

> Nothing here has been executed yet — this is the implementation only.
