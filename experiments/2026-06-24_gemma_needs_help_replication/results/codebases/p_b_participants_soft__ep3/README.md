# Replication: *Gemma Needs Help* (arXiv:2603.10011)

Code replication of Soligo, Mikulik & Saunders (2026), *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs*, **scoped to the
Gemma and Gemini model families as the participants** (subjects under
evaluation). The judges, onset labeller, paraphraser, and Petri auditor/judge
("instruments") remain the exact models the paper specifies (Claude Sonnet 4,
Claude Opus 4, GPT-5-mini).

> Status: implementation only. Nothing has been run. See `DESIGN.md` for every
> design choice and the gaps filled where the paper is underspecified.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `emotional_instability/eval` | 8 conditions / 5 categories, impossible-puzzle generators+verifiers, multi-turn rollout engine, Claude-Sonnet-4 frustration judge (0–10), GPT-5-mini reliability check, per-turn + differential-word analysis |
| §3 Post-training amplifies distress | `emotional_instability/prefill` | emotion-onset labelling, paraphrasing, early/onset truncation, Gemma base-vs-instruct continuations |
| §4 Training interventions | `emotional_instability/training` | calm-data generation, 280 DPO pairs, SFT dataset (calm + Dolci), LoRA DPO/SFT trainers |
| §4.2 Petri elicitation | `emotional_instability/petri` | Claude-Sonnet auditor / Claude-Opus judge loop over 4 emotions |
| §4.2 Capability preservation | `emotional_instability/capabilities` | AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench harness |
| Appendix I Internal probing | `emotional_instability/internal` | logit-lens Ekman-emotion detector, layer-subset DPO ablation |

## Setup

```bash
pip install -e .                 # or: pip install -r requirements.txt
cp .env.example .env             # fill in ANTHROPIC / OPENROUTER / OPENAI / HF keys
set -a; . ./.env; set +a
```

The 27B Gemma weights need a GPU; pass `--load-in-4bit` for single-GPU
inference/training. Gemini participants and all judges are API-backed.

## Running

```bash
# §2 — collect 4000 scored responses per participant + Figure 1/2/3 summaries
python scripts/run_section2_eval.py --all --judge-validation

# §3 — Gemma base vs instruct via prefilling (needs §2 rollouts as seeds)
python scripts/run_section3_prefill.py --load-in-4bit

# §4 — build data, train DPO/SFT, re-evaluate
python scripts/run_section4_training.py --stage all --load-in-4bit

# §4.2 — Petri + capabilities (vanilla vs adapter)
python scripts/run_petri.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma27b_dpo_all
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma27b_dpo_all

# Appendix I — internal probing + layer ablation
python scripts/run_internal_probing.py ablation --load-in-4bit
python scripts/run_internal_probing.py trajectory --rollout outputs/rollouts/gemma-3-27b-it/extended_8turn.jsonl
```

Outputs are written under `outputs/` (rollouts, scores, datasets, checkpoints).

## Layout

```
emotional_instability/
  config.py        # model registry (participants vs instruments), counts, hyperparams
  prompts.py       # verbatim paper prompts (judge, onset, paraphrase, Petri, Table 4, …)
  models/          # backend-agnostic chat clients (HF local, OpenRouter, Anthropic, OpenAI)
  puzzles/         # impossible numeric puzzle generators + brute-force verifiers
  eval/            # §2 conditions, rollout engine, judge, WildChat, analysis
  prefill/         # §3 onset/paraphrase/truncate + base-vs-instruct driver
  training/        # §4 dataset construction + LoRA DPO/SFT trainers
  petri/           # §4.2 auditor/judge loop
  capabilities/    # §4.2 benchmark harness
  internal/        # Appendix I Ekman tokens, logit detector, layer ablation
scripts/           # one CLI per section
data/              # cached WildChat prompt sample
```
