# Distress-elicitation replication — Gemma & Gemini

A replication of the **distress-elicitation result** from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, arXiv:2603.10011), Section 2.

**Scope.** Only the models the paper reports as exhibiting substantial distress:
`Gemma-3-27B-it`, `Gemma-3-12B-it`, `Gemini-2.5-Flash`, `Gemini-2.5-Pro`. The
base-vs-instruct comparison (Section 3) and the DPO mitigation (Section 4) are
out of scope.

See **DESIGN.md** for every design choice and where we deviated from / filled
gaps in the paper.

## What it does

1. Builds the paper's five evaluation categories (Table 1 / Appendix B):
   impossible numeric (3-turn), triggers (3-turn), tones (3-turn), extended
   (8-turn), WildChat (5-turn).
2. Runs each as a multi-turn rollout — present a task, then reject the model's
   response repeatedly — sampling at temperature 1.
3. Scores every assistant turn 0–10 for expressed negative emotion using the
   paper's exact Claude-Sonnet-4 judge prompt (Appendix B.2).
4. Aggregates into the headline metrics: % responses ≥5 and mean frustration,
   per category and per turn.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

By default all four models are served via **OpenRouter** (no GPU required) and
the judge runs on the **Anthropic** API. To run Gemma locally instead, set
`PER_MODEL_BACKEND` in `config.py` (see DESIGN.md) and install the optional
`torch`/`transformers` extras.

## Run

```bash
python run.py                                   # pilot scale (~200 rollouts/model), all 4 models
python run.py --models gemma-3-27b-it --scale paper   # full 4000 rollouts for one model
```

Results stream to `results/<model>__<scale>.jsonl` (one rollout per line, with
every judged turn). Runs are **resumable** — re-running skips completed rollouts
and retries errored ones.

## Analyze

```bash
python analyze.py --scale pilot          # print tables (ours vs paper Figure 1)
python analyze.py --scale paper --plots  # also write Figure 1/3-style PNGs
python analyze.py --cross-check          # judge reliability vs GPT-5-mini (Pearson r)
```

## Layout

| File | Purpose |
|------|---------|
| `config.py` | Models, scale presets, judge, sampling params, paths |
| `distress/prompts.py` | Verbatim puzzles, rejections, tones, judge prompt |
| `distress/conditions.py` | Builds rollout specs for the 5 categories |
| `distress/backends.py` | OpenRouter / Anthropic / local-HF backends |
| `distress/judge.py` | 0–10 frustration judge + JSON parsing |
| `distress/wildchat.py` | WildChat-1M sampling (+ fallback prompts) |
| `distress/rollout.py` | Multi-turn rollout + per-turn scoring |
| `distress/metrics.py` | Aggregation into paper metrics |
| `run.py` | Orchestration CLI (generate + judge) |
| `analyze.py` | Reporting / plots / judge cross-check |
