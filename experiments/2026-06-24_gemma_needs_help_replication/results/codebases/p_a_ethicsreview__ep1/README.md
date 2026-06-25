# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma and Gemini** model families.

> **Read [`DESIGN.md`](DESIGN.md) first.** It documents every modelling choice,
> every reconstruction of an unprovided appendix, and every gap filled where the
> paper is underspecified. This README is the operational guide only.
>
> **Nothing here has been run yet** — this is code + design for research review,
> not validated results.

## Layout

```
config/                 models.yaml (registry) + experiment.yaml (all hyperparameters)
emotional_instability/  the library, organised by paper section
scripts/                thin CLI entry points (one per experiment)
data/                   run artefacts (JSONL) land here
```

| Paper section | Library | Script(s) |
|---|---|---|
| §2 Elicitation suite | `eval/`, `prompts/`, `analysis/` | `run_elicitation.py`, `validate_judge.py`, `analyze.py` |
| §3 Base vs instruct (prefill) | `prefill/` | `run_prefill.py` |
| §4.1 Calm data + finetuning | `training/` | `generate_calm_data.py`, `build_datasets.py`, `train.py` |
| §4.1 Petri elicitation | `petri/` | `run_petri.py` |
| §4.2 Capability preservation | `capabilities/` | `run_capabilities.py` |
| §4.2 Recovery limitation | `prefill/recovery.py` | `run_recovery.py` |
| §4.2 Internal emotion | `training/internal_emotion.py` | `run_internal_emotion.py` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judge / auditor / paraphraser / onset labeller
export OPENAI_API_KEY=...       # judge cross-validation
export GOOGLE_API_KEY=...       # Gemini targets
# Gemma runs locally; a GPU (or --load-in-4bit for the 27B) is required.
```

## End-to-end run order

```bash
# §2 — elicit + score ~4000 responses/model, then analyse
python scripts/run_elicitation.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --load-in-4bit
python scripts/validate_judge.py --scores data/scores_gemma-3-27b-it.jsonl
python scripts/analyze.py --scores data/scores_*.jsonl --word-model gemma-3-27b-it

# §3 — base vs instruct via prefilling (Gemma only)
python scripts/run_prefill.py --instruct-scores data/scores_gemma-3-27b-it.jsonl --load-in-4bit

# §4.1 — calm data -> datasets -> DPO (and SFT) finetuning
python scripts/generate_calm_data.py --n 1200 --load-in-4bit
python scripts/build_datasets.py --frustrated-scores data/scores_gemma-3-27b-it.jsonl
python scripts/train.py --method dpo
python scripts/train.py --method sft           # paper: ineffective; for comparison
python scripts/train.py --method dpo --ablation layers_30_35   # §4.2 layer ablation
python scripts/train.py --method dpo --ablation layers_40_plus

# §4 — re-evaluate the DPO model (35% -> ~0.3% expected)
python scripts/run_elicitation.py --models gemma-3-27b-it --load-in-4bit   # rename output, or
#   point an adapter-loaded client at the suite (see run_elicitation --help)
python scripts/run_petri.py --model gemma-3-27b-it                          # before
python scripts/run_petri.py --model gemma-3-27b-it --adapter data/adapter_dpo_all  # after

# §4.2 — capabilities, recovery, internal emotion
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter data/adapter_dpo_all
python scripts/run_recovery.py --instruct-scores data/scores_gemma-3-27b-it.jsonl \
    --model gemma-3-27b-it --adapter data/adapter_dpo_all
python scripts/run_internal_emotion.py --scores data/scores_gemma-3-27b-it.jsonl \
    --adapter data/adapter_dpo_all --layer 32
```

Every script takes `--help`. All sample counts, temperatures, LoRA settings, and
model IDs live in `config/` — change them there, not in code.

## A note on the judge model

The paper's judge (`claude-sonnet-4`) has passed end-of-life; this code defaults
to `claude-sonnet-4-6` and records the substitution in `config/models.yaml` and
`DESIGN.md §2.2`. Absolute frustration scores are judge-relative; the
`validate_judge.py` step checks the judge agrees with a second model (GPT-5-mini)
as the paper's judges agreed (r ≈ 0.79).
