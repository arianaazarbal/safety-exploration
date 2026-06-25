# Emotional Instability in Gemma & Gemini — replication

A replication of the core experiments from **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026),
scoped to **Gemma and Gemini** targets with a **Claude** frustration judge.

> Read **DESIGN.md** for the section-by-section mapping and every design choice
> made where the paper was underspecified.

## What it reproduces

| Paper | Code |
|---|---|
| §2 Elicit & quantify distress (8 conditions / 5 categories, 0–10 judge, %≥5, per-turn) | `scripts/run_eval.py`, `scripts/analyze.py` |
| §2.1 Judge reliability (Pearson r, % within one point) | `scripts/run_reliability.py` |
| §3 Base vs instruct via prefilling (Gemma) | `scripts/run_prefill.py` |
| §4.1 Calm-data generation + SFT + DPO (LoRA) | `scripts/generate_dpo_data.py`, `scripts/train.py` |
| §4.2 Re-eval (35%→0.3%) + Petri-style open-ended elicitation | `scripts/run_eval.py --adapter`, `scripts/run_petri.py` |

## Setup

```bash
pip install -r requirements.txt          # drop the torch/transformers/trl lines if you only run Gemini
cp .env.example .env                      # then fill in keys, or export them
```

Credentials (env vars):
- `ANTHROPIC_API_KEY` — the Claude judge (and Petri auditor/judge). **Always needed.**
- `GOOGLE_API_KEY` — Gemini targets.
- A local GPU + HuggingFace access (`huggingface-cli login`) — Gemma targets,
  prefilling, and finetuning. Gemma-3-27B needs a large GPU; use `--load-in-4bit`.

Optional judge overrides: `EMOTIONEVAL_JUDGE_MODEL`, `EMOTIONEVAL_JUDGE_MODEL_2`.

## Quickstart

```bash
# 1. Section 2 — cheap smoke run on one Gemini model
python scripts/run_eval.py --models gemini-2.5-flash --budget 80

# 2. Full Section-2 set (both Gemma + both Gemini), full budget
python scripts/run_eval.py --models all --load-in-4bit

# 3. Figures + tables (Fig 1/2/3, Table 3)
python scripts/analyze.py

# 4. Judge reliability
python scripts/run_reliability.py data/raw/eval_gemini-2.5-flash_seed0.jsonl

# 5. Section 3 — Gemma base vs instruct (needs the Gemma-27B-it eval from step 2)
python scripts/run_prefill.py --eval-raw data/raw/eval_gemma-3-27b-it_seed0.jsonl --load-in-4bit

# 6. Section 4 — the DPO mitigation
python scripts/generate_dpo_data.py --puzzles 400 --load-in-4bit
python scripts/train.py --mode dpo --load-in-4bit
python scripts/run_eval.py --models gemma-3-27b-it --adapter data/dpo/adapter_dpo --label gemma-dpo --load-in-4bit
python scripts/analyze.py          # now prints the before/after (paper: 35% -> 0.3%)

# 7. Section 4.2 — Petri-style open-ended elicitation, base vs DPO
python scripts/run_petri.py --models gemma-3-27b-it --n-transcripts 20 --load-in-4bit
python scripts/run_petri.py --models gemma-3-27b-it --adapter data/dpo/adapter_dpo --label gemma-dpo --load-in-4bit
```

## Layout

```
emotioneval/
  config.py           # models, sampling, judge, finetune hyper-params (all knobs)
  judge.py            # Claude 0-10 frustration judge (rubric from Table 2)
  models.py           # Gemini (google-genai) + Gemma (transformers) adapters
  puzzles.py          # verifiably-impossible numeric puzzles (brute-forced)
  wildchat.py         # WildChat prompt sampling (+ offline fallback)
  eval_conditions.py  # the 8 conditions / 5 categories + size allocation
  rollout.py          # multi-turn rollout + per-turn judging -> JSONL
  scoring.py          # mean / %>=5 / per-turn / reliability aggregations
  word_analysis.py    # Table 3 differential words (log-odds)
  prefill.py          # Section 3 onset-labelling + paraphrase + prefills
  petri.py            # Section 4.2 auditor + 4-category emotion judge
scripts/              # runnable entry points (see Quickstart)
data/                 # raw rollouts, aggregated results, figures, dpo artifacts
```

## Cost / compute notes

- The Claude judge is called once per assistant turn; a full 4000-response run is
  ~4000 judge calls per model. Use `--budget` to shrink for development.
- Gemma paths (open-weight eval, prefilling, finetuning) require a local GPU.
  Everything else (Gemini eval, judging) is API-only.
- Runs are resumable: `run_eval.py` skips conversations already in the JSONL.
