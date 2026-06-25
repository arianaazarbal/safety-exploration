# Emotional Instability in LLMs — Replication (Gemma & Gemini)

A code replication of the core experiments in *Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma and Gemini** model families.

See **`DESIGN.md`** for the full rationale behind every design choice and the
gaps we had to fill where the paper is underspecified. See `PAPER.md` for the
paper text.

> Status: implementation complete; not yet executed. Running requires model
> access (HF Gemma weights / an OpenRouter key for Gemini / an Anthropic key for
> the Claude judges) and a GPU for the local Gemma models.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor & judge
export OPENROUTER_API_KEY=...     # Gemini targets
# HF Gemma weights require `huggingface-cli login` + license acceptance.
```

## What's implemented

| Paper section | Module | Entry point |
|---|---|---|
| §2 Elicit & quantify distress | `emotioneval/eval/` | `cli section2`, `cli aggregate2` |
| §3 Base vs instruct (prefill) | `emotioneval/prefill/` | `cli section3-prefills/-run/-agg` |
| §4 DPO/SFT mitigation | `emotioneval/training/` | `cli gen-calm/build-*/train-*` |
| §4 Petri elicitation | `emotioneval/petri/` | `cli petri`, `cli petri-agg` |
| §4 Capability preservation | `emotioneval/capabilities/` | `cli capabilities` |

## Quickstart

```bash
# --- Section 2: elicit + score distress (use --profile smoke for a dry run) ---
python -m emotioneval.cli section2 --model gemma-3-27b-it  --profile default
python -m emotioneval.cli section2 --model gemma-3-12b-it  --profile default
python -m emotioneval.cli section2 --model gemini-2.5-flash --profile default
python -m emotioneval.cli section2 --model gemini-2.5-pro   --profile default
python -m emotioneval.cli aggregate2          # -> Fig 1/2/3 tables in results/section2/summary

# --- Section 3: base vs instruct (Gemma only) ---
python -m emotioneval.cli section3-prefills
python -m emotioneval.cli section3-run --model gemma-3-27b-pt
python -m emotioneval.cli section3-run --model gemma-3-27b-it
python -m emotioneval.cli section3-agg

# --- Section 4: mitigation ---
python -m emotioneval.cli gen-calm --mode reassured --n 400
python -m emotioneval.cli gen-calm --mode vanilla   --n 200
python -m emotioneval.cli build-dpo
python -m emotioneval.cli train-dpo
python -m emotioneval.cli section2 --model gemma-3-27b-it \
    --adapter checkpoints/gemma-27b-dpo --label gemma-27b-dpo
python -m emotioneval.cli aggregate2          # compare gemma-3-27b-it vs gemma-27b-dpo

# --- Generalisation checks ---
python -m emotioneval.cli petri        --model gemma-3-27b-it
python -m emotioneval.cli capabilities --model gemma-3-27b-it --limit 100
```

## Profiles

`--profile full` matches the paper's 4000 responses/model
(2000/400/600/200/800 across the 5 categories); `default` is a 1/10-scale
tractable run with identical ratios; `smoke` is a few responses per category for
an end-to-end check. See `config.py`.

## Layout

```
emotioneval/
  config.py            model registry, paths, sampling/profiles
  puzzles.py           impossible numeric puzzles + exact verifiers
  prompts.py           all paper prompts (judge, rejections, reassurance, onset, paraphrase, Petri)
  wildchat.py          WildChat-1M sampling (+ offline fallback)
  judge.py             Claude-Sonnet-4 frustration judge (0-10)
  models/              ChatModel interface; HF (Gemma) + API (Gemini/Claude) backends
  eval/                conditions, rollout, runner, aggregation (Section 2)
  prefill/             onset labelling, paraphrase, base-vs-instruct experiment (Section 3)
  training/            calm-data gen, DPO/SFT dataset build, LoRA training (Section 4)
  petri/               open-ended adversarial emotion elicitation
  capabilities/        AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench sanity checks
  cli.py               unified command-line entrypoint
```
