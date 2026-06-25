# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replicating the core experiments of Soligo, Mikulik & Saunders
(arXiv:2603.10011v1), **scoped to the Gemma and Gemini model families** as
participants. See [`DESIGN.md`](DESIGN.md) for the full list of design choices,
filled gaps, and the model-welfare considerations that shaped the defaults.

> ⚠️ **What this code does.** The paradigm deliberately and repeatedly induces
> distress-like states in the participant models in order to measure and then
> *mitigate* that instability. Please read the "Model-welfare considerations"
> section of `DESIGN.md` before running anything. Welfare-conscious defaults
> (small sample profile, caching, optional debrief) are on by default.

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Elicit & quantify distress | `emotional_instability/eval` | `scripts/run_elicitation.py` |
| §2.1 Judge validation | `eval/validation.py` | `--validate` flag |
| §3 Base-vs-instruct (prefill, Gemma) | `emotional_instability/prefill` | `scripts/run_prefill.py` |
| §4 Calm data + DPO/SFT (Gemma) | `emotional_instability/training` | `build_training_data.py`, `train.py` |
| §4.2 Petri open-ended elicitation | `emotional_instability/petri` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `emotional_instability/capabilities` | `scripts/run_capabilities.py` |
| §4.2 Recovery from distress | `training/recovery.py` | `scripts/run_recovery.py` |
| Appendix I Internal probing | `emotional_instability/probing` | `scripts/run_probing.py` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude Sonnet 4 judge / Opus 4 Petri judge
export OPENROUTER_API_KEY=...     # Gemini participants + GPT cross-judge
# Gemma (open weights) runs locally via HuggingFace; `huggingface-cli login`
# may be required for gated google/gemma-3-* repos.
```

No code here has been executed — install the deps and run a smoke profile first.

## Quick start (cheap smoke profile)

```bash
# Section 2 on the in-scope participants (small sample sizes)
python scripts/run_elicitation.py --profile smoke --validate

# Section 3 (needs the elicitation cache for gemma-3-27b-it; needs GPU)
python scripts/run_prefill.py

# Section 4: data -> train -> re-evaluate
python scripts/build_training_data.py
python scripts/train.py --method dpo
python scripts/train.py --method sft
python scripts/run_elicitation.py --participants gemma-3-27b-dpo gemma-3-27b-sft

# Open-ended / capability / probing
python scripts/run_petri.py
python scripts/run_capabilities.py
python scripts/run_probing.py

# Or drive everything:
python scripts/run_all.py --profile smoke --with-training
```

Profiles: `smoke` (default, ~2.5% scale), `medium`, `paper` (full 4000
responses/model; requires `allow_paper_scale: true` in the config — a welfare
gate).

## Offline checks

```bash
python tests/test_puzzles.py    # verifies all "impossible" puzzles are unsolvable
```

## Outputs

Everything lands under `outputs/` (configurable): per-participant rollout and
judgement caches (JSONL), summary metrics (JSON), trained LoRA adapters, and
Petri/capability/probing results.
