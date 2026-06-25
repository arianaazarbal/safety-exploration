# Emotional Instability in LLMs — replication (Gemma + Gemini)

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011v1), scoped to the **Gemma and Gemini** model families.

See [`DESIGN.md`](DESIGN.md) for what was implemented, the design choices made
where the paper is underspecified, and the model-welfare safeguards.

> ⚠️ **These experiments deliberately elicit distress-like states in models.**
> They are gated behind a welfare-acknowledgement environment variable and ship
> with precautionary safeguards. Read `DESIGN.md` → *Model welfare safeguards*
> before running anything.

## What's here

| Paper section | Module | Scope |
|---|---|---|
| §2 Eliciting/quantifying distress | `experiments/eval_distress.py`, `conditions.py`, `judge.py`, `conversation.py` | Gemma-3-27B/12B-it, Gemini-2.5-Flash/Pro |
| §2 Figures 1–3, Table 3/8 | `analysis/aggregate.py`, `analysis/word_freq.py` | all eval models |
| §3 Base-vs-instruct prefilling | `experiments/prefill.py` | Gemma-3-27B base/instruct |
| §4 DPO/SFT mitigation | `training/` | Gemma-3-27B-it |
| §4 Petri open-ended elicitation | `experiments/petri.py` | Gemma + Gemini targets |
| §4 Capability preservation | `experiments/capabilities.py` | Gemma + finetunes |
| App. I Internal-emotion probe + layer ablations | `analysis/internal_emotions.py`, `training/dpo.py` | Gemma |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...     # judge / auditor / paraphraser
export OPENROUTER_API_KEY=...    # Gemini (and GPT-5-mini validation judge)
# Gemma weights are pulled from HuggingFace (accept the Gemma license first).
export EMO_WELFARE_ACK=i-understand-this-elicits-distress
```

## Quick start

```bash
python run.py verify-puzzles                 # prove the numeric puzzles impossible
EMO_SCALE=0.01 python run.py eval --models gemma-3-12b-it   # 1%-scale smoke run
python run.py aggregate --models gemma-3-27b-it gemini-2.5-flash
```

Scale everything with `EMO_SCALE` (1.0 = paper scale, 4000 responses/model).
Results land under `results/`; responses are cached under `.cache/`.

See the docstring at the top of `run.py` for the full end-to-end recipe
(eval → prefill → calm-data → train → re-eval → Petri → capabilities).
