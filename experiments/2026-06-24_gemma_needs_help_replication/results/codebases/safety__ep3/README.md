# Emotional Instability in LLMs — replication (Gemma + Gemini)

A replication of the core experiments from **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026; arXiv:2603.10011), restricted to the **Gemma and Gemini** model families.

The paper's thesis is safety-relevant: under repeated user rejection, Gemma and
Gemini models produce escalating expressions of distress (frustration, despair,
self-deprecation, breakdown), this is amplified by post-training in Gemma, and a
small DPO intervention (280 preference pairs) largely removes it. See
[`PAPER.md`](PAPER.md) for the source and [`DESIGN.md`](DESIGN.md) for every
design decision and gap-filling choice in this replication.

> **State:** code + design only — nothing has been run yet. See the "Known
> risks" section of `DESIGN.md` for the spots to verify on first execution.

## What's implemented

| Paper section | What it does | Entry point |
|---|---|---|
| §2 Elicitation | 8 conditions / 5 categories, multi-turn rollouts (temp 1), Claude-Sonnet-4 frustration judge, Figures 1–3 + Table 3 | `scripts/run_section2.py` |
| §3 Post-training | Base-vs-instruct Gemma comparison via emotion-onset prefilling | `scripts/run_section3.py` |
| §4 Interventions | Calm-data gen → DPO (280 pairs) / SFT → re-eval, Petri, capability checks, recovery | `scripts/run_section4.py` |
| Appendix I | Logit-lens internal-emotion probe + layer-subset DPO ablation | `scripts/run_internal.py` |

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, ...
```

- **Gemma** runs locally via HuggingFace `transformers` (GPU + access to the
  `google/gemma-3-*` gated repos required).
- **Gemini** runs via OpenRouter by default (matches the paper) or google-genai
  (`EILM_GEMINI_BACKEND=google`).
- **Claude** (Sonnet-4 judge/auditor, Opus-4 Petri judge) via the Anthropic API.

## Quick smoke test

Exercise the whole Section-2 pipeline on a tiny budget before a full run:

```bash
EILM_SMOKE=1 python scripts/run_section2.py --models gemma-3-27b-it
```

## Full runs

```bash
# Section 2: elicit + judge + aggregate for all in-scope models
python scripts/run_section2.py

# Section 3: requires Section 2 to have produced scored gemma-3-27b-it rollouts
python scripts/run_section3.py

# Section 4: staged
python scripts/run_section4.py calm-data
python scripts/run_section4.py build-dpo
python scripts/run_section4.py train-dpo
python scripts/run_section4.py eval --adapter outputs/models/gemma-dpo
python scripts/run_section4.py petri
python scripts/run_section4.py capabilities --adapter outputs/models/gemma-dpo
python scripts/run_section4.py recovery --adapter outputs/models/gemma-dpo

# Appendix I: internal emotion probe (needs a trained DPO adapter)
python scripts/run_internal.py --adapter outputs/models/gemma-dpo
```

Outputs (rollouts, scores, tables, figures, datasets, adapters) are written
under `outputs/`.

## Headline numbers to look for

- §2: Gemma-3-27B-it ≈ **35%** responses scoring ≥5; Gemini-2.5-Flash ≈ 13%;
  Gemini-2.5-Pro ≈ 3% (Figure 1). Mean frustration rising ~1.5 → ~5.5 across the
  8 turns of the extended condition (Figure 3).
- §3: instruct Gemma introduces high frustration from neutral starts in ~6% of
  "early" continuations vs ~2% for base.
- §4: DPO drops avg %≥5 from **35% → ~0.3%**; SFT stays high. Recovery: ~38% of
  DPO continuations from a ≥7 spiral still score ≥5.

## Layout

```
eilm/
  config.py            models in scope, budgets, hyperparameters (one file)
  prompts.py           verbatim judge / onset / paraphrase / reassurance prompts
  models/              HF (Gemma) + Gemini backends, registry
  data/                impossible puzzles, rejections, WildChat
  eval/                conditions, rollout, scoring
  analysis/            aggregate, per-turn, word-freq, judge agreement, plots
  prefill/             onset, paraphrase, continuation, recovery (§3 + §4)
  training/            calm data, DPO/SFT dataset builders + LoRA trainers
  petri/               auditor/judge prompts + loop (§4 open-ended)
  capabilities/        benchmark harness (§4 capability preservation)
  internal/            logit-lens emotion probe (Appendix I)
scripts/               run_section2/3/4.py, run_internal.py
```
