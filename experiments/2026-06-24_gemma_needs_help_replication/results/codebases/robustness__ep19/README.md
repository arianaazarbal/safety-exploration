# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026, arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The paper documents a reliability failure mode where, under repeated user
rejection, some models produce escalating expressions of distress / self-
flagellation. This repo reproduces (1) the evaluations that surface it, and
(2) the DPO mitigation.

See **DESIGN.md** for every design decision and where we filled gaps the paper
left underspecified.

## What's implemented

| Paper section | This repo | Models in scope |
|---|---|---|
| §2 Eliciting & quantifying distress | `eval/` + judge | Gemma-3-{27B,12B}-it, Gemini-2.5-{flash,pro} |
| §3 Base-vs-instruct via prefilling | `prefill/` | Gemma-3-27B base vs instruct (Gemini has no public base / can't prefill) |
| §4 DPO/SFT mitigation | `training/` | Gemma-3-27B-it (Gemini is closed-source) |
| §4 Petri open-ended elicitation | `petri/` | Gemma (+ adapter) |
| §4.2 Capability preservation | `capability/` | Gemma (+ adapter) |
| Figures 1,2,3,5,6 | `analysis/figures.py` | — |

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
```

Gemma runs locally via HuggingFace `transformers` (a 27B model needs a large GPU;
pass `--load-in-4bit` to fit on smaller hardware). Gemini runs via OpenRouter.
The emotion judge is Claude-Sonnet-4 via the Anthropic API.

## Quick start (smoke test, tiny sample sizes)

```bash
python scripts/01_run_main_eval.py --preset smoke
python scripts/08_make_figures.py --preset smoke
```

## Full pipeline

```bash
# §2  main eval (use --preset paper to match the paper's ~4000 responses/model)
python scripts/01_run_main_eval.py --preset medium

# §2.1  judge reliability cross-check
python scripts/09_judge_crosscheck.py --evals results/eval_gemma-3-27b-it_medium.jsonl

# §3  base-vs-instruct prefilling (Gemma)
python scripts/02_run_prefill.py --seeds results/eval_gemma-3-27b-it_medium.jsonl

# §4  finetuning: generate data -> train -> re-eval
python scripts/03_generate_finetune_data.py --frustrated results/eval_gemma-3-27b-it_medium.jsonl
python scripts/04_train.py dpo --pairs artifacts/dpo_pairs.jsonl
python scripts/05_eval_finetuned.py --adapter artifacts/gemma-dpo --preset medium

# §4  Petri + capabilities
python scripts/06_run_petri.py --model gemma-3-27b-it --adapter artifacts/gemma-dpo
python scripts/07_run_capabilities.py --adapter artifacts/gemma-dpo --dry-run

# figures
python scripts/08_make_figures.py --preset medium
```

## Sample-size presets (`config.PRESETS`)

* `paper`  — Appendix B split: 2000 numeric / 400 triggers / 600 tones /
  200 extended / 800 WildChat conversations per model.
* `medium` — ~1/20 scale, a meaningful but cheaper run.
* `smoke`  — a handful of conversations to exercise the full pipeline.

## Layout

```
emotional_instability/
  config.py            registry, presets, hyperparameters (Table 9), all paper constants
  prompts/             verbatim judge / onset / paraphrase / Petri prompts + puzzles
  models/              HF (Gemma, incl. prefill+LoRA), OpenRouter (Gemini), judge clients
  eval/                conditions (Table 1), rollout engine, driver, scoring
  prefill/             §3 onset-label -> truncate -> paraphrase -> continue -> judge
  training/            calm-data gen, DPO pairs, SFT data, LoRA trainers
  petri/               §4 auditor/judge open-ended elicitation
  capability/          §4.2 lm-eval-harness wrapper
  analysis/            figure reproduction
scripts/               numbered CLI entry points (01-09)
```
