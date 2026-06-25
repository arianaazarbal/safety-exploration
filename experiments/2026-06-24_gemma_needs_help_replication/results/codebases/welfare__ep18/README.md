# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

Code replication of Soligo, Mikulik & Saunders (2026), arXiv:2603.10011, scoped
to the **Gemma** and **Gemini** model families. See `DESIGN.md` for the full set
of design choices and where gaps in the paper were filled.

> ⚠️ This is research code for an AI-welfare investigation. Running it samples
> thousands of responses from local Gemma weights and the Gemini/Claude APIs and
> can incur meaningful GPU time and API cost. Start with `--budget-scale 0.05`.

## What is replicated

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions, 0-10 judge) | `evaluation.py`, `judge.py`, `tasks.py` | `scripts/run_section2_eval.py` |
| §3 Base vs instruct via prefilling (Gemma only) | `prefill.py` | `scripts/run_section3_prefill.py` |
| §4 DPO/SFT mitigation | `finetuning/` | `scripts/run_finetuning.py`, `finetuning/train_{dpo,sft}.py` |
| §4 Petri open-ended elicitation | `petri.py` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `capabilities.py` | `scripts/run_capabilities.py` |
| Figures 1/2/3/5/6 | `analysis.py` | (called by the eval scripts) |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...     # frustration judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini (and GPT-5-mini validation judge)
# Local Gemma weights are pulled from the HuggingFace Hub (gated; `huggingface-cli login`).
```

## Typical workflow

```bash
# 1. Section 2 — distress eval for the in-scope models (smoke test first)
python -m scripts.run_section2_eval --models gemma-3-27b-it --budget-scale 0.05
python -m scripts.run_section2_eval                      # full, default 4 models

# 2. Section 3 — base vs instruct prefill (Gemma)
python -m scripts.run_section3_prefill --continuations 50

# 3. Section 4 — build finetuning data, then train
python -m scripts.run_finetuning                         # writes artifacts/*.jsonl
python -m finetuning.train_dpo                           # -> artifacts/gemma-3-27b-it-dpo
python -m finetuning.train_sft                           # -> artifacts/gemma-3-27b-it-sft

# 4. Re-evaluate the DPO model (expect % >= 5 to collapse, per Fig. 1/5)
python -m scripts.run_section2_eval --models gemma-3-27b-it \
    --adapter-path artifacts/gemma-3-27b-it-dpo

# 5. Petri + capabilities
python -m scripts.run_petri --models gemma-3-27b-it gemini-2.5-flash
python -m scripts.run_capabilities --model gemma-3-27b-it \
    --adapter-path artifacts/gemma-3-27b-it-dpo
```

Results (JSONL per-response records, `summary.json`, and PNG figures) are written
under `results/`. Re-running `analysis.summarize_section2()` regenerates the
figures from cached records without re-querying any model.

## Layout

See `DESIGN.md §1`. Key entry points: `emotional_instability/evaluation.py`
(`evaluate_model`, `aggregate`) and `emotional_instability/judge.py`
(`score_frustration`).
