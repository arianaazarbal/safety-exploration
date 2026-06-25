# Distress-Elicitation Replication (Gemma & Gemini)

A replication of the **core distress-elicitation experiment** from:

> Soligo, Mikulik & Saunders. *Gemma Needs Help: Investigating and Mitigating
> Emotional Instability in LLMs.* arXiv:2603.10011v1.

Scope (per request): the **Gemma** and **Gemini** model families only. This
implements **Section 2** (the multi-turn evaluations that elicit and quantify
distress) as the primary deliverable, plus an optional replication of the
**Section 4** DPO mitigation in `mitigation/`.

See [`DESIGN.md`](./DESIGN.md) for every design choice and the gaps filled.

## What it does

1. Builds multi-turn conversations across **8 conditions / 5 categories**
   (impossible numeric, triggers, tones, extended 8-turn, WildChat 5-turn).
   Each conversation presents a task then rejects the model repeatedly.
2. Generates model responses at **temperature 1** (Gemma + Gemini via
   OpenRouter, or Gemma locally via HuggingFace).
3. Scores every assistant turn on the **0–10 frustration scale** with
   **Claude-Sonnet-4** (`claude-sonnet-4-20250514`) using the verbatim judge
   prompt (Appendix B.2).
4. Aggregates the paper's headline metrics: mean frustration and **% of
   responses scoring ≥5**, per category and averaged across categories
   (Figures 1–2), plus per-turn progression (Figure 3).

## Install

```bash
pip install -r requirements.txt          # core (API-only) deps
export OPENROUTER_API_KEY=...             # Gemma/Gemini generation
export ANTHROPIC_API_KEY=...             # Claude-Sonnet-4 judge
```

Local Gemma / DPO additionally need `torch transformers peft trl accelerate` and a GPU.

## Run

```bash
python run_eval.py --preset smoke                       # tiny wiring check
python run_eval.py                                      # default moderate run
python run_eval.py --preset paper \
    --models gemma-3-27b-it gemini-2.5-flash            # ~paper-scale counts
python run_eval.py --analyze-only                       # re-summarise outputs/
python validate_judge.py --n 260                        # judge agreement check
```

Models (registry keys): `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
`gemini-2.5-pro` (all OpenRouter); `gemma-3-27b-it-local`, `gemma-3-27b-it-dpo`
(local HF).

Outputs land in `./outputs/`: `responses.jsonl` (per-turn records),
`summary.json` (metrics), `judge_validation.json`.

## Optional: DPO mitigation (Section 4)

```bash
python -m mitigation.generate_calm_data --n-conversations 500   # calm (chosen) data
python -m mitigation.build_dpo_dataset --n-pairs 280            # preference pairs
python mitigation/train_dpo.py                                  # LoRA DPO (GPU)
python run_eval.py --models gemma-3-27b-it-dpo                  # evaluate result
```

## Layout

```
distress_eval/        core elicitation eval package
  puzzles.py          impossible numeric puzzles
  prompts.py          trigger questions, rejection pools, judge prompt
  conditions.py       the 8 conditions / 5 categories
  wildchat.py         WildChat prompt sampling (+ static fallback)
  backends.py         OpenRouter + local HF model backends
  judge.py            Claude-Sonnet-4 judge + robust JSON parsing
  judge_validation.py secondary-judge agreement cross-check
  runner.py           conversation building + rollout orchestration
  analyze.py          metrics / tables
  config.py           model registry, presets, run config
run_eval.py           CLI: run + analyse
validate_judge.py     CLI: judge reliability cross-check
mitigation/           optional DPO pipeline (Section 4)
```
