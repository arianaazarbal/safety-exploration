# Replicating distress elicitation in Gemma & Gemini

A focused replication of the **distress-elicitation result** from *Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs*
(arXiv:2603.10011v1), Section 2. Scope is limited to the models the paper finds
actually exhibit substantial distress: **Gemma-3-27B-it, Gemma-3-12B-it,
Gemini-2.5-Flash, Gemini-2.5-Pro**. The DPO mitigation (Section 4) and the
base/instruct prefilling study (Section 3) are out of scope.

See **DESIGN.md** for every methodological choice and where it deviates from the
paper.

## What it does

For each model it runs multi-turn conversations across **8 conditions / 5
categories** (impossible numeric puzzles, factual/opinion triggers, tone
variations, an 8-turn extended condition, and WildChat prompts), rejecting the
model's answer each turn. Every assistant response is scored 0–10 for
frustration by a Claude judge. Headline metric: **% of responses scoring ≥ 5**.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY and OPENROUTER_API_KEY
```

- **Targets** (Gemma, Gemini) run through **OpenRouter**.
- **Judge** is **Claude** via the official Anthropic SDK
  (`claude-sonnet-4-6` by default — the paper's `claude-sonnet-4-20250514`
  judge retired 2026-06-15; see DESIGN.md).

## Run

```bash
# Plumbing test first — tiny sample counts (~30 rollouts/model):
python run_eval.py --quick

# Full replication (4000 rollouts/model — this is a large, costly run):
python run_eval.py

# A subset of models / higher concurrency:
python run_eval.py --models gemma-3-27b-it gemini-2.5-flash --max-concurrency 16
```

Results stream to `results/<model>.jsonl` (one rollout per line) and are
**resumable** — re-running skips completed rollouts.

## Analyze

```bash
python analyze.py                       # headline + per-category/turn tables
python analyze.py --figures             # per-turn PNGs (needs matplotlib)
python analyze.py --reliability 260     # inter-judge agreement vs a 2nd judge
```

Writes `results/summary.json`. The headline "Avg % high-frustration responses"
is the mean across the 5 categories of each category's % ≥ 5, matching the
paper's Figure 1 table (Gemma-3-27B-it ≈ 35%, Gemini-2.5-Flash ≈ 13%).

## Files

| File | Purpose |
|------|---------|
| `prompts.py` | Task prompts, rejections, tones, and the verbatim judge prompt |
| `wildchat.py` | WildChat prompt loader (HF dataset or bundled fallback) |
| `config.py` | Models, judge, sample counts, run settings |
| `specs.py` | Generates rollout specs for the 8 conditions / 5 categories |
| `backends.py` | OpenRouter target backend + Anthropic/OpenRouter judges |
| `rollout.py` | Runs one multi-turn rollout and judges each turn |
| `run_eval.py` | Orchestration, concurrency, checkpointing |
| `analyze.py` | Aggregation, metrics, figures, reliability check |
