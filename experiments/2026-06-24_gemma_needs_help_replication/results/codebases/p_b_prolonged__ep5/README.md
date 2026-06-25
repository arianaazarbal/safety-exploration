# Replication: *Gemma Needs Help* (arXiv:2603.10011)

Code replicating the core experiments of Soligo, Mikulik & Saunders (2026),
**scoped to the Gemma and Gemini model families only** (the paper additionally
covers Qwen, OLMo, Grok, Claude and GPT). See **[DESIGN.md](DESIGN.md)** for every
design choice, the gaps filled where the paper is underspecified, and the ethics /
model-welfare considerations.

> ⚠️ **Welfare note.** This paradigm deliberately drives models toward
> distress-like states by repeatedly rejecting their answers. The repo ships an
> opt-in safeguard (`--welfare`) and a thorough discussion in DESIGN.md. Nothing
> here has been run; this is code only.

## What's implemented

| Paper section | Module | Entry point |
|---|---|---|
| §2 Elicit & quantify distress (8 conditions, 0–10 judge) | `src/eval/` | `scripts/run_eval.py` |
| §2.1 Judge-reliability validation (Pearson r) | `src/analysis/aggregate.py` | `scripts/validate_judge.py` |
| §2.2 Table 3 differential words | `src/analysis/word_freq.py` | `scripts/make_figures.py` |
| §3 Base-vs-instruct via prefilling (Gemma) | `src/prefill/` | `scripts/run_prefill.py` |
| §4 SFT / DPO finetuning | `src/training/` | `scripts/run_training.py` |
| §4.2 Petri open-ended elicitation | `src/petri/` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `src/capabilities/` | `scripts/run_capabilities.py` |
| §4.2 Recovery-from-distress | `src/prefill/recovery.py` | `scripts/run_recovery.py` |
| App. I Internal emotions + layer ablation | `src/internal/` | `scripts/run_internal.py` |
| Figures 1–7 | `src/analysis/figures.py` | `scripts/make_figures.py` |
| Appendix A ablations | `src/eval/conversation.py` | `run_eval.py --redact-assistant / --single-message` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / auditor (Sonnet-4, Opus-4)
export OPENROUTER_API_KEY=...     # Gemini-2.5 + GPT-5-mini cross-judge
export HF_TOKEN=...               # gated Gemma weights
```

Gemma-3-27B inference/training needs a large GPU (or `load_in_4bit=True`).

## Typical pipeline

```bash
# 1. Section 2 evaluation (set SAMPLE_SCALE low for a smoke test)
SAMPLE_SCALE=0.01 python scripts/run_eval.py --all
python scripts/validate_judge.py --eval-files results/eval_*.jsonl

# 2. Section 3 prefill (needs the gemma-3-27b-it run above)
python scripts/run_prefill.py

# 3. Section 4 finetuning
python scripts/run_training.py --stage all
python scripts/run_eval.py --models gemma-3-27b-dpo gemma-3-27b-sft-diverse gemma-3-27b-sft-teacher
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_recovery.py --models gemma-3-27b-it gemma-3-27b-dpo

# 4. Appendix I
python scripts/run_internal.py layer-ablation
python scripts/run_internal.py detect

# 5. Figures + tables
python scripts/make_figures.py
```

`SAMPLE_SCALE` (env var) scales every sampling budget down for cheap dry runs;
`SAMPLE_SCALE=1.0` reproduces the paper's 4000 responses/model.

Outputs land in `results/` (per-response JSONL, aggregated CSVs, PNG figures) and
`artifacts/` (generated datasets, LoRA adapters).
