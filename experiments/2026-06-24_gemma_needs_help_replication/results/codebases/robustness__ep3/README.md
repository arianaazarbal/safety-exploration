# emoeval — replicating *"Gemma Needs Help"* (Gemma + Gemini)

A code replication of the core experiments from **Soligo, Mikulik & Saunders
(2026), "Gemma Needs Help: Investigating and Mitigating Emotional Instability in
LLMs"** (arXiv 2603.10011v1), scoped to **Gemma and Gemini** models.

The paper documents a reliability failure mode: under repeated user rejection,
Gemma (and to a lesser extent Gemini) models spiral into expressions of distress
("self-flagellation"), and shows this can be measured and mitigated. This repo
reproduces:

1. **§2 Elicitation** — a multi-turn rejection eval + Claude-Sonnet-4 frustration
   judge that surfaces distress in Gemma/Gemini.
2. **§3 Origin** — base-vs-instruct prefilling showing post-training amplifies it.
3. **§4 Mitigation** — DPO on ~280 preference pairs that collapses
   high-frustration responses (paper: 35% → 0.3%), plus Petri open-ended
   elicitation, capability-preservation benchmarks, and an internal-emotion probe.

See **`DESIGN.md`** for every design decision and gap-filling choice.

> ⚠️ This code has not been executed yet (no weights/keys in the authoring
> environment). Expect a debugging pass on first run; library-version drift
> (transformers/TRL) is the likeliest snag.

## Layout

```
emoeval/        # library
  config.py        model registry, eval conditions, hyperparameters
  models.py        local (HF Gemma) + API (Gemini/Claude) backends
  tasks.py         puzzles, trigger/wildchat prompts, rejection schedules
  rollout.py       multi-turn rejection protocol
  judge.py         Claude-Sonnet-4 0-10 frustration judge (verbatim prompt)
  wildchat.py      WildChat prompt sampling (+ offline fallback)
  datagen.py       §4.1 calm/frustrated data + DPO/SFT dataset construction
  train.py         LoRA DPO/SFT (TRL) incl. layer-subset ablation
  prefill.py       §3 onset-labelling, paraphrase, truncation, continuation
  petri.py         §4.2 open-ended auditor/judge (verbatim prompts)
  capabilities.py  §4.2 MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench subsets
  probing.py       Appendix I logit-lens internal-emotion detection
  analysis.py      aggregation + Figures 1/2/3/5
scripts/        # entry points (see --help on each)
  run_elicitation.py  generate_dpo_data.py  run_finetune.py  run_prefill.py
  run_petri.py        run_capabilities.py    run_probing.py   make_figures.py
```

## Setup

```bash
pip install -r requirements.txt

# API key for Gemini targets + the Claude judge/auditor (served via OpenRouter)
export OPENROUTER_API_KEY=sk-or-...
# Optional overrides (defaults shown):
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1
# export JUDGE_MODEL_ID=anthropic/claude-sonnet-4
# export SECONDARY_JUDGE_MODEL_ID=openai/gpt-5-mini
```

Local Gemma runs need a GPU (27B → ~large; uncomment `bitsandbytes` in
`requirements.txt` for quantised loading, or `vllm` for faster sampling).

## Quickstart

```bash
# 0) Cheap smoke test of the whole eval path (~2% of the sample budget):
python scripts/run_elicitation.py --models gemma-3-27b-it --scale 0.02

# 1) §2 full elicitation sweep (Gemma + Gemini):
python scripts/run_elicitation.py \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/make_figures.py            # Figures 1/2/3, headline leaderboard

# judge reliability check (Section 2.1):
python scripts/run_elicitation.py --models gemma-3-27b-it --judge-agreement 260

# 2) §3 base vs instruct (needs an elicitation file for gemma-3-27b-it):
python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it

# 3) §4 mitigation: build data, train DPO, re-evaluate:
python scripts/generate_dpo_data.py
python scripts/run_finetune.py --method dpo
python scripts/run_elicitation.py --models gemma-3-27b-it \
    --adapter adapters/dpo_gemma --label dpo_gemma
python scripts/make_figures.py            # Figure 5: 35% -> ~0.3%

# supporting analyses:
python scripts/run_petri.py        --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_capabilities.py --model  gemma-3-27b-it --limit 50
python scripts/run_probing.py      --adapter adapters/dpo_gemma
# Appendix I layer ablation:
python scripts/run_finetune.py --method dpo --layers 30 31 32 33 34 35 \
    --output-name dpo_gemma_l30-35
```

Results are written to `results/` (per-rollout JSONL + summary CSVs + figures
under `results/figures/`); adapters to `adapters/`; generated data to `data/`.
Override locations with `EMOEVAL_RESULTS` / `EMOEVAL_DATA` / `EMOEVAL_ADAPTERS`.

## Scope note

Per the replication brief, only Gemma and Gemini are registered as evaluation
targets (`emoeval/config.py:MODELS`). The Claude judge is retained as the
measurement instrument. The code is family-agnostic, so adding Qwen/OLMo/etc. is
a one-line registry edit. Fine-tuning, prefilling and probing are Gemma-only
(weight access required); Gemini participates in elicitation and Petri only.
