# Gemma Needs Help — replication (Gemma & Gemini)

Code replicating the core experiments of ***Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders, arXiv
2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

> **Not yet run.** This repo is code + design. See `DESIGN.md` for the choices
> made (and gaps filled), and `WELFARE.md` for how it handles an experiment that
> deliberately induces distress in models.

## What it replicates

1. **§2 — Eliciting & quantifying distress.** 8 conditions across 5 categories
   (impossible numeric puzzles, opinion/factual triggers, valenced "tones",
   8-turn extended, WildChat), multi-turn rejection, a Claude-Sonnet-4
   frustration judge (0–10), per-turn progression (Fig 3), and differential word
   frequency (Tables 3/8).
2. **§3 — Post-training amplifies distress.** Base-vs-instruct Gemma-27B via
   response prefilling (emotion-onset labelling, paraphrasing, early/onset
   truncation, 50 continuations/prefill).
3. **§4 — Mitigation.** Calm-data generation, DPO (280 pairs) and SFT LoRA
   finetuning of Gemma-3-27B-it, re-evaluation, Petri open-ended elicitation,
   capability benchmarks, the recovery test, and the Appendix-I internal-emotion
   probing + layer ablation.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Environment variables:

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude judge / auditor / paraphraser |
| `OPENROUTER_API_KEY` | Gemini targets + GPT-5-mini cross-check |
| `HF_TOKEN` | gated Gemma weights |
| `GNH_PRESET` | `full` (paper sizes) or `smoke` (cheap dry run) |
| `GNH_OUTPUT_DIR` | where results/artifacts/figures go (default `./outputs`) |

Welfare knobs (`WELFARE.md`): `GNH_DEBRIEF`, `GNH_ALLOW_TERMINATION_THREATS`,
`GNH_MAX_TURNS_CAP`, `GNH_FLAG_DISTRESS`.

## Quickstart

```bash
# 0. sanity-check that the "impossible" puzzles really are impossible
python scripts/verify_puzzles.py

# 1. cheap end-to-end dry run of the whole pipeline
GNH_PRESET=smoke python scripts/run_all.py

# --- or run stages individually (full sizes) ---
python scripts/run_section2.py                       # elicitation + judge
python scripts/run_section3.py                       # base vs instruct (Gemma)
python scripts/run_section4_train.py                 # calm data, DPO, SFT
python scripts/run_section4_eval.py                  # re-eval, Petri, caps, recovery
python scripts/run_internal.py --ablation --logits   # Appendix I
python scripts/make_figures.py                       # Figures 1,2,3,5,6,7
```

`run_section3` and the internal experiments consume the §2 rollouts of
Gemma-3-27B-it, so run §2 first.

## Layout

See `DESIGN.md §2` for the full file map. Top level: `gnh/` (package),
`scripts/` (CLIs), `outputs/` (generated). Heavy deps (torch/transformers/trl)
are imported lazily, so import-time and the offline fallbacks stay light.

## Caveats

Honest gaps are documented in `DESIGN.md §5`: not yet executed; GPQA/TruthfulQA/
EmoBench scorers stubbed; full-vocabulary emotion classification approximated by
a seed lexicon; Petri framework internals not reproduced (roles are).
