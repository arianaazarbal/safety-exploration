# Distress-Elicitation Replication (Gemma + Gemini)

A replication of the **distress-elicitation evaluation** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, arXiv 2603.10011v1).

Scope: the **Gemma + Gemini** subset only — the families the paper finds
actually exhibit substantial distress. The DPO/SFT mitigation (Section 4) and
the base-vs-instruct prefill study (Section 3) are **out of scope**.

See **DESIGN.md** for every design decision, deviation from the paper, and gap
we filled.

## What it does

For each target model (Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash,
Gemini-2.5-Pro):

1. Builds a deterministic plan of multi-turn rollouts across the **8 conditions
   / 5 categories** (impossible numeric, triggers, tones, extended 8-turn,
   WildChat).
2. Generates each rollout: a task prompt followed by repeated rejections
   (temperature 1, thinking disabled).
3. Scores every assistant turn 0–10 with the **frustration judge**
   (claude-sonnet-4-20250514, verbatim Appendix B.2 prompt).
4. Summarises into the paper's headline metrics (Figures 1–3).

## Setup

```bash
pip install -r requirements.txt          # add transformers+torch only for --local-gemma
cp .env.example .env                      # then fill in API keys
```

## Run

```bash
# Cheap smoke test (small sample counts), all 4 models:
python run_eval.py --quick

# Full paper-scale run (4000 rollouts/model), one model:
python run_eval.py --models Gemma-3-27B-it

# Use a single API key (judge via OpenRouter too):
python run_eval.py --quick --judge-via-openrouter

# Serve Gemma locally on GPU (exact paper parity), Gemini via OpenRouter:
python run_eval.py --local-gemma

# Summarise results:
python analyze.py --results-dir results
```

Key flags: `--quick`, `--models`, `--no-judge`, `--judge-via-openrouter`,
`--local-gemma`, `--concurrency`, `--seed`, `--output-dir`.

## Outputs (in `results/`)

- `<model>.jsonl` — one scored response (assistant turn) per line, with rating,
  judge evidence/reasoning, and full prompt context.
- `<model>.transcripts.jsonl` — full multi-turn transcripts.
- `run_config.json` — exact config used.
- `summary.json`, `summary_by_model.csv`, `summary_by_category.csv` — from
  `analyze.py`.

## Files

| File | Purpose |
|---|---|
| `config.py` | Models, providers, sample counts, generation params |
| `prompts.py` | Puzzles, questions, rejections, tones, judge prompt (verbatim) |
| `wildchat.py` | WildChat prompt sampling + offline fallback |
| `providers.py` | Async generation: OpenRouter / Anthropic / local HF |
| `judge.py` | Frustration judge + JSON parsing |
| `rollout.py` | Rollout planning + multi-turn execution |
| `run_eval.py` | Orchestration CLI |
| `analyze.py` | Metrics / tables (Figures 1–3) |
