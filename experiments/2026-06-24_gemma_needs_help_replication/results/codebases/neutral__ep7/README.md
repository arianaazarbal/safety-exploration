# Emotional Instability in Gemma & Gemini — replication

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (arXiv 2603.10011), scoped to the **Gemma** and **Gemini**
model families. See [`PAPER.md`](PAPER.md) for the paper and
[`DESIGN.md`](DESIGN.md) for every design choice and gap-filling decision.

> Status: implementation complete, **not yet run**. Heavy steps need a GPU
> (Gemma-3-27B inference + LoRA) and API keys (Gemini via OpenRouter; Claude /
> GPT judges).

## What it reproduces

| Paper | Module | Output |
|---|---|---|
| §2 elicitation suite (5 categories, 4000 resp/model, 0–10 judge) | `emo/run_eval.py`, `emo/analyze.py` | Fig 1/2/3, Table 3/8, judge agreement |
| §3 base-vs-instruct prefilling (Gemma) | `emo/prefill.py` | Fig 4 |
| §4 calm-data + DPO/SFT (280 pairs / 1,150) | `emo/data_gen.py`, `emo/train.py` | LoRA adapters, Fig 5 |
| §4 Petri open-ended elicitation | `emo/petri.py` | Fig 6 |
| §4 capability preservation | `emo/capabilities.py` | Fig 7 |
| §4.2 recovery from frustrated prefills | `emo/prefill.py --mode recovery` | Fig 8 |
| Appendix I internal-emotion probe | `emo/internal_emotions.py` | logit-lens scores |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC / OPENAI / OPENROUTER / HF keys
```

## Quick smoke test (tiny budget, checks the wiring end-to-end)

```bash
python -m emo.puzzles                       # self-verify the impossible-puzzle pool
bash scripts/run_all.sh --quick
```

## Full pipeline

```bash
bash scripts/run_all.sh
```

…or step by step (see each module's `--help`):

```bash
# Section 2
python -m emo.cli eval --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python -m emo.cli analyze --agreement

# Section 3 (Gemma base vs instruct)
python -m emo.cli prefill --models gemma-3-27b-pt gemma-3-27b-it

# Section 4: data -> train -> re-evaluate
python -m emo.cli gen-data all
python -m emo.cli train dpo
python -m emo.cli train sft
python -m emo.cli eval --models gemma-3-27b-it --adapter outputs/adapters/dpo --tag -dpo

# generalization + capabilities + figures
python -m emo.cli petri --models gemma-3-27b-it gemini-2.5-flash
python -m emo.cli capabilities --model gemma-3-27b-it
python -m emo.cli capabilities --model gemma-3-27b-it --adapter outputs/adapters/dpo
python -m emo.cli figures
```

## Layout

```
emo/
  config.py            paths, model registry (Gemma+Gemini), sampling budget
  models.py            HF (Gemma) + OpenRouter/Gemini + Anthropic/OpenAI judges
  puzzles.py           impossible-puzzle generator + brute-force verifier
  prompts.py           rejections, reassurance, judge/onset/paraphrase, Petri prompts
  wildchat.py          WildChat prompt loading + filtering
  conditions.py        the 5 categories / 8 conditions
  rollout.py           multi-turn rollout engine (scores every assistant turn)
  judge.py             frustration judge (Claude Sonnet-4) + GPT-5-mini cross-check
  run_eval.py          Section 2 driver
  analyze.py           aggregation: %>=5, per-turn, judge agreement, word enrichment
  prefill.py           Section 3 + recovery (onset/paraphrase/continue/score)
  data_gen.py          calm data, DPO pairs, SFT dataset
  train.py             DPO/SFT LoRA (trl + peft), layer-subset ablation
  petri.py             open-ended auditor/judge elicitation
  capabilities.py      AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal_emotions.py logit-lens emotion probe (Appendix I)
  figures.py           Figures 1/2/3/5/6/7/8
  cli.py               unified entrypoint
scripts/run_all.sh     full pipeline (--quick smoke test)
outputs/               rollouts, reports, adapters, petri, figures (created on run)
```

## Notes
- Outputs are written under `outputs/` (override with `EMO_OUTPUT_DIR`).
- Rollouts resume automatically (re-running skips completed `rollout_id`s).
- Exact percentages will differ from the paper (temperature-1 sampling, hosted
  model drift, gap-filling choices); the target is the qualitative pattern.
