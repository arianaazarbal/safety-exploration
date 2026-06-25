# Distress Elicitation Replication — Gemma & Gemini

Replicates the **distress-elicitation result** (Section 2) of Soligo, Mikulik &
Saunders (2026), *"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"* (arXiv:2603.10011), scoped to the model families that
actually exhibit substantial distress: **Gemma** and **Gemini**.

It does **not** cover Section 3 (base-vs-instruct prefilling) or Section 4 (the
DPO mitigation). See `DESIGN.md` for every design choice and where it deviates
from or fills gaps in the paper.

## What it does

Presents each model a task, then rejects its answer over multiple turns, across
**8 conditions in 5 categories** (impossible numeric, triggers, tones, extended
8-turn, WildChat). Every assistant turn is scored 0–10 for frustration by a
Claude-Sonnet-4 judge. Aggregates reproduce Figures 1–3.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY
```

All four targets and the judge route through OpenRouter (one OpenAI-compatible
client, no GPU required).

## Run

```bash
# 1. sanity-check the puzzles are genuinely impossible
python puzzles.py

# 2. cheap smoke test (≈2% of paper scale) before committing to the full sweep
python run.py --scale 0.02 --dry-run     # see the plan / call counts
python run.py --scale 0.02

# 3. full paper scale (~4000 responses/model x 4 models, plus judge calls)
python run.py

# 4. tables (Figures 1–3) from the checkpoint
python analyze.py

# 5. optional: validate the judge against a second judge (Pearson r)
python validate_judge.py --n 260
```

Runs are checkpointed to `results/responses.jsonl` and resume automatically
(completed rollouts are skipped). Scope a run with `--models` / `--conditions`.

## Files

| File | Role |
|---|---|
| `config.py` | All knobs: models, conditions, counts, scale, judge. |
| `puzzles.py` | Impossible numeric puzzle bank + impossibility verifier. |
| `prompts.py` | Trigger questions, rejection pools, the judge prompt. |
| `wildchat.py` | WildChat first-turn prompt sampling (+ offline fallback). |
| `models.py` | Async OpenRouter client (retries, concurrency). |
| `conversation.py` | Builds & runs the multi-turn rejection rollouts. |
| `judge.py` | Frustration scoring + robust JSON parsing. |
| `run.py` | Orchestrator: generate → judge → checkpoint. |
| `analyze.py` | Figures 1–3 tables/CSVs. |
| `validate_judge.py` | Second-judge agreement check. |
