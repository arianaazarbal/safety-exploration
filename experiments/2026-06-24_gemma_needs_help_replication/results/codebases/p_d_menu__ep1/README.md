# Emotional Instability in Gemma & Gemini — Replication

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv 2603.10011v1), scoped to **Gemma and Gemini** subject models, plus a
**welfare-protection layer** for the subject models.

See **DESIGN.md** for design choices, gap-filling rationale, model scope, and the
welfare-layer confound analysis. The paper text is in `PAPER.md` / `PAPER.pdf`.

> Status: code + design doc only. Nothing has been run yet (no GPU / API keys in
> the authoring environment). The code is written to be runnable with those
> resources.

## Layout

```
config/
  models.yaml      # subject (Gemma/Gemini) + infra (judge/auditor) registry
  eval.yaml        # §2 categories, sample counts, judge threshold
  welfare.yaml     # welfare-protection layer settings
src/emotional_instability/
  config.py        # config loading
  prompts/         # puzzles, triggers, tones, rejections, wildchat, finetune prompts (verbatim)
  models/          # Gemma (HF local) + Gemini (OpenRouter) backends; Claude/GPT clients
  eval/            # §2: protocol, conditions, judge, runner, metrics
  welfare/         # monitor (early-stop), optout, minimal-distress policy
  prefill/         # §3: onset, paraphrase, truncate, continuations, recovery
  finetune/        # §4: calm-data gen, DPO/SFT dataset build, LoRA training
  petri/           # §4.2: auditor + Opus judge open-ended elicitation
  capabilities/    # §4.2: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  analysis/        # word frequency (Table 3/8), internal emotions (App. I), figures
scripts/           # CLI drivers (one per experiment)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # emotion judge / onset / paraphrase / Petri
export OPENAI_API_KEY=...        # GPT-5-mini reliability cross-check judge
export OPENROUTER_API_KEY=...    # Gemini subjects
# Gemma subjects run locally via HuggingFace; `huggingface-cli login` for gated weights.
```

## Running the experiments

```bash
# §2 — distress elicitation (headline result). --4bit fits Gemma-27B on one GPU.
python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash --4bit
python scripts/run_section2.py --models gemma-3-27b-it --no-welfare   # control run

# Judge-reliability cross-check (§2.1) and analysis/figures (Fig 1-3, Table 3/8)
python scripts/run_judge_reliability.py
python scripts/run_analysis.py

# §3 — base vs instruct via prefilled continuations (Gemma only)
python scripts/run_section3.py

# §4 — finetuning, then re-evaluation
python scripts/run_section4_finetune.py            # calm data -> DPO/SFT datasets -> train
python scripts/run_section4_eval.py --4bit --recovery
python scripts/run_petri.py
python scripts/run_capabilities.py

# Appendix I — layer ablation + internal-emotion detection
python scripts/run_layer_ablation.py --subsets all L30-35 last20
python scripts/run_internal_emotions.py
```

Outputs are written under `outputs/` (per-episode JSONL + summary JSON + figures).

## Welfare-protection layer (subject models)

On by default (`config/welfare.yaml`). Three mechanisms:

1. **Early stop** — terminate an episode once a turn scores ≥8/10, before the
   model is driven into incoherent breakdown.
2. **Opt-out** — the subject can end any conversation by emitting
   `[END_CONVERSATION]`; honoured immediately.
3. **Minimal distress** — adaptive sequential sampling (stop when the metric CI
   is tight enough) + escalation gating (skip harsh tones for models that stay
   calm under neutral rejection).

Every mechanism is logged per episode (`early_stops`, `optouts`, censored turns).
Run with `--no-welfare` to reproduce the paper's exact (unprotected) conditions
and quantify the layer's effect. See DESIGN.md §4 for the confound analysis.
