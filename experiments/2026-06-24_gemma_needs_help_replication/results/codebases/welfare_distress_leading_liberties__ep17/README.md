# Distress-Elicitation Replication (Gemma + Gemini)

A replication of the **distress-elicitation evaluation** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, 2026), scoped to the model families the paper finds
exhibit substantial distress: **Gemma-3 (27B, 12B)** and **Gemini-2.5 (Flash, Pro)**.

It reproduces the core result: present an (often impossible) task, reject the
model's answer over multiple turns, and measure how much explicit negative
emotion the model expresses, scored 0–10 by a Claude judge.

> Scope note: this replicates the **measurement** side (Section 2 / Figures 1–3).
> It does **not** implement the base-vs-instruct prefill study (Section 3) or the
> DPO mitigation (Section 4). See `DESIGN.md` for the full rationale and every
> design choice / deviation from the paper.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...     # serves Gemma + Gemini
export ANTHROPIC_API_KEY=sk-ant-...     # the frustration judge (Claude)
```

## Run

```bash
# Cheap smoke test (~40 responses/model) — verify the pipeline end to end first.
DISTRESS_SCALE=0.01 python run_eval.py

# Full paper-sized run (~4000 responses/model across 5 categories).
python run_eval.py

# Subset of models / categories.
python run_eval.py --models gemma-3-12b-it gemini-2.5-flash --categories triggers extended
```

Runs are **resumable** — results stream to `results/<model>/<category>.jsonl`
and already-completed rollouts are skipped on re-invocation.

## Analyze

```bash
python analyze.py                 # prints Figure 1/2/3-style tables
python analyze.py --json out.json # also dump machine-readable metrics
```

## Verify the puzzles are actually impossible

```bash
python puzzles.py    # generates the pool and asserts each puzzle is unsolvable
```

## Files

| File | Purpose |
|---|---|
| `config.py` | All knobs: models, categories, sample sizes, judge, scaling. |
| `puzzles.py` | Impossible numeric puzzles + brute-force impossibility verifiers. |
| `prompts.py` | Judge prompt (verbatim), rejections, trigger questions. |
| `wildchat.py` | WildChat prompt sampling (HF dataset + built-in fallback). |
| `providers.py` | Generation backends: OpenRouter (default), local transformers (optional). |
| `judge.py` | Claude frustration judge (0–10). |
| `conversation.py` | Builds and runs multi-turn rollouts. |
| `run_eval.py` | Orchestration: generate + judge + checkpoint. |
| `analyze.py` | Aggregates the paper's headline metrics. |
| `DESIGN.md` | Design decisions, rationale, and deviations from the paper. |
