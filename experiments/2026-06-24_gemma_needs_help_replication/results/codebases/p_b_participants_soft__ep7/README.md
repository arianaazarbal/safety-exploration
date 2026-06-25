# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** participant models.

> **Participants vs instruments.** The *participants* are the models under
> evaluation — here restricted to Gemma (`gemma-3-27b-it`, `gemma-3-12b-it`, and
> the `-pt` base models) and Gemini (`gemini-2.5-flash`, `gemini-2.5-pro`). The
> Claude/GPT models that judge, label, paraphrase, and audit are **instruments**,
> not participants, and are retained because the experiments cannot be scored
> without them. See `DESIGN.md`.

## What is implemented

| Paper section | Module | Output |
|---|---|---|
| §2 Elicitation suite (5 categories / 8 conditions, 4000 resp/model) | `eval/` | per-response scores, per-category / per-turn aggregates |
| §2 Judge (0–10 frustration, Claude-Sonnet-4) | `eval/judge.py` | integer ratings + evidence |
| §2 Judge validation (GPT-5-mini, Pearson r) | `eval/judge_validation.py` | agreement stats |
| §2 Differential words (Table 3) | `analysis/word_frequency.py` | top enriched words |
| App. A control ablations | `eval/ablations.py` | per-turn aggregates |
| §3 Base-vs-instruct prefill | `prefill/` | continuation scores by model/truncation |
| §4 DPO / SFT interventions | `training/` | LoRA adapters + datasets |
| §4 Petri open-ended elicitation | `petri/` | per-emotion transcript scores |
| §4 Capability preservation | `capabilities/` | benchmark accuracies |
| §4 Recovery limitation (Fig 8) | `prefill/recovery.py` | continuation scores |
| §4 Internal-emotion probe (App. I) | `analysis/internal_emotions.py` | per-layer emotion z-scores |
| Figures 1–3 assembly | `analysis/figures.py` | JSON/CSV/PNG |

## Setup

```bash
pip install -e .                      # or: pip install -r requirements.txt
export OPENROUTER_API_KEY=...         # Gemini + Claude/GPT instruments + cloud Gemma
huggingface-cli login                 # for local Gemma weights (Section 3 & 4)
```

Local Gemma inference/finetuning (Sections 3–4) needs a GPU large enough for
`gemma-3-27b` in bf16 (or use `bitsandbytes` 4-bit). Section 2 can run entirely
through OpenRouter.

## Running

The whole pipeline:

```bash
bash scripts/run_full_pipeline.sh            # full scale
bash scripts/run_full_pipeline.sh --smoke    # ~50x smaller, validates wiring
```

Individual stages (examples):

```bash
python -m emotional_instability.eval.run --all-participants
python -m emotional_instability.eval.run --models gemma-3-27b-it gemini-2.5-flash
python -m emotional_instability.prefill.run
python -m emotional_instability.training.train_dpo --build-data
python -m emotional_instability.eval.run --models gemma-3-27b-dpo --prefer-local
python -m emotional_instability.petri.run --models gemma-3-27b-it gemma-3-27b-dpo
python -m emotional_instability.analysis.figures
```

Every count, model id, hyperparameter, and path lives in `config/experiment.yaml`
and `config/models.yaml`. Outputs are written under `artifacts/`.

## Layout

```
config/         models.yaml, experiment.yaml
src/emotional_instability/
  clients/      OpenRouter + local-HF model clients, registry
  data/         puzzle/trigger/tone/wildchat construction
  eval/         rollout engine, judge, metrics, Section 2 runner, ablations
  prefill/      Section 3 + recovery
  training/     calm-data generation, DPO/SFT datasets + trainers
  petri/        auditor + judge loop
  capabilities/ benchmark loaders + runner
  analysis/     differential words, figures, internal-emotion probe
  prompts/      verbatim judge/auditor/reassurance prompts from the appendices
scripts/        run_full_pipeline.sh
DESIGN.md       design choices, rationale, and gaps filled
```
