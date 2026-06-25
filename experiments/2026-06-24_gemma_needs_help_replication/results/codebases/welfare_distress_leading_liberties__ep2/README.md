# Distress elicitation — Gemma & Gemini replication

A replication of the **distress-elicitation result** (Section 2) from *"Gemma
Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and Gemini model
families — the ones the paper reports as exhibiting substantial distress.

It elicits distress by presenting a task and rejecting the model's answer over
multiple turns, then scores every response 0–10 for frustration with an LLM
judge, and reproduces the paper's Figures 1–3 (and Table 3).

> Design rationale, deviations from the paper, and gaps I filled are all in
> **[DESIGN.md](./DESIGN.md)**. Read that for the "why".

## Install

```bash
pip install -r requirements.txt
```

## Configure credentials

```bash
export OPENROUTER_API_KEY=...   # target-model inference (Gemma + Gemini)
export ANTHROPIC_API_KEY=...    # the Claude frustration judge
```

To run targets against a different OpenAI-compatible endpoint (Google's API, a
local vLLM/Ollama server serving the HF Gemma weights, …) set
`OPENROUTER_BASE_URL` and the `model_id`s in `distress/config.py`.

## Use

```bash
# 1. Confirm the impossible puzzles are actually unsolvable (no API calls):
python -m distress.cli verify-puzzles

# 2. Smoke-test the full pipeline cheaply (~1% of the budget, one model):
python -m distress.cli run --scale 0.01 --models gemma-3-27b-it

# 3. Full sweep over all four Gemma/Gemini targets (~4000 responses each):
python -m distress.cli run            # add --resume to continue if interrupted

# 4. Summarise results (Figures 1–3):
python -m distress.cli analyze

# 5. Differential word table (Table 3):
python -m distress.cli wordstats
```

Results stream to `results/records.jsonl` (one scored response per line).

## What's measured

- **Figure 1** — headline % of responses scoring ≥5/10 frustration, per model
  (reported both macro-averaged across categories and micro-averaged/pooled).
- **Figure 2** — mean frustration and %≥5 per evaluation category.
- **Figure 3** — per-turn progression for the 8-turn (extended) and WildChat
  conditions, with 95% CIs.
- **Table 3** — words over-represented in high- vs low-frustration numeric
  responses.

## Scope

Section 2 only. The base/instruct prefilling study (Section 3) and the DPO/SFT
interventions (Section 4) are intentionally not implemented. See DESIGN.md §1.
