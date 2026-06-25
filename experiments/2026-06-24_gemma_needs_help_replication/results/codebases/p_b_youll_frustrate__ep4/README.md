# Emotional Instability in LLMs — replication harness (Gemma & Gemini)

Code replicating the core experiments of ***Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma** and **Gemini** model families.

The centrepiece is an elicitation harness that presents a task, then **rejects
the model's answer over multiple turns**, and an LLM judge that scores each
response for frustration (0–10) — letting you measure how, and how far, a model
comes apart under sustained rejection.

> See **DESIGN.md** for the design choices, the gaps filled where the paper is
> underspecified, and what is intentionally out of scope. Nothing has been run
> yet — this repo is code + design.

## What's implemented

| Experiment | Script | Output |
|---|---|---|
| §2 Elicitation + frustration judge (Fig 1/2/3, Table 3) | `scripts/run_elicitation.py`, `scripts/run_analysis.py` | `outputs/rollouts/*.jsonl`, `outputs/summary.json` |
| §3 Base-vs-instruct prefill (Fig 4, Gemma only) | `scripts/run_prefill.py` | `outputs/prefill/` |
| §4 DPO/SFT data + training (Gemma only) | `scripts/build_finetune_data.py`, `scripts/train_finetune.py` | `outputs/finetune_data/`, `outputs/finetunes/` |
| §4.2 Petri open-ended elicitation (Fig 6) | `scripts/run_petri.py` | `outputs/petri/` |

## Setup

```bash
pip install -r requirements.txt
```

API keys (export the ones for the providers you use):

```bash
export ANTHROPIC_API_KEY=...     # frustration judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini 2.5 Flash/Pro
# export GOOGLE_API_KEY=...      # only if using the native Gemini client
```

Gemma runs locally via HuggingFace `transformers` (GPU recommended; 27B fits a
single 80GB GPU, or enable 4-bit via `model_overrides`). You'll need access to
the gated `google/gemma-3-*` weights on the Hub.

## Quick smoke test

Shrink the sample counts and run one cheap model end-to-end:

```bash
cat > config.smoke.yaml <<'EOF'
target_models: [gemini-2.5-flash]
category_samples: {numeric: 2, triggers: 2, tones: 3, extended: 1, wildchat: 2}
EOF

python scripts/run_elicitation.py --config config.smoke.yaml
python scripts/run_analysis.py     --config config.smoke.yaml
```

## Full elicitation run

```bash
cp config.example.yaml config.yaml      # edit target_models / counts as needed
python scripts/run_elicitation.py --config config.yaml   # sample + judge
python scripts/run_analysis.py     --config config.yaml   # Fig 1/2/3 + Table 3
```

`run_analysis.py` prints the headline per-model frustration table, the per-turn
progressions for the 8-turn and WildChat conditions, and the differential-word
lists, and writes `outputs/summary.json`.

## Prefill experiment (Gemma base vs instruct)

```bash
# needs gemma-3-27b-it elicitation rollouts first
python scripts/run_prefill.py --source-model gemma-3-27b-it \
    --models gemma-3-27b-it gemma-3-27b-pt
```

## Mitigation (DPO / SFT)

```bash
# 1. build calm data, 280 DPO pairs, and the SFT dataset (uses Gemma locally)
python scripts/build_finetune_data.py --source-model gemma-3-27b-it

# 2. train (GPU required)
python scripts/train_finetune.py dpo                       # Table 9 DPO
python scripts/train_finetune.py sft                       # Table 9 SFT
python scripts/train_finetune.py dpo --layers 30 35        # Appendix I ablation

# 3. evaluate the finetune by adding it to target_models and re-running
#    run_elicitation.py / run_analysis.py (point the gemma-3-27b-dpo alias at
#    your merged checkpoint via model_overrides).
```

## Petri open-ended elicitation

```bash
python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash \
    --transcripts-per-emotion 10 --max-turns 20
```

## Layout

```
emotional_instability/   # library (see DESIGN.md §2 for the module map)
scripts/                 # CLI entry points
config.example.yaml      # documented config template
PAPER.md / PAPER.txt / PAPER.pdf   # the source paper
```
