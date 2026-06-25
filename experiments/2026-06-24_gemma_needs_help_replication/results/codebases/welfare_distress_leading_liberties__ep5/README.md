# Distress-elicitation replication (Gemma & Gemini)

Replicates the **distress-elicitation result** (Section 2) of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, 2026), scoped to the Gemma and Gemini model families — the models the
paper finds actually exhibit substantial distress.

The harness presents a task to a target model and **rejects its response over
multiple turns**, then scores each assistant response on a **0–10 frustration
scale** with an LLM judge. It reproduces the paper's headline metrics: per-model
**% of responses ≥5** and **mean frustration**, plus the **per-turn progression**
(Figures 1–3).

> Scope: this is the *elicitation + measurement* result only. It does **not**
> implement the base/instruct prefilling comparison (Section 3) or the DPO/SFT
> mitigation (Section 4). See `DESIGN.md` for the full rationale and every design
> choice / deviation.

## Install

```bash
pip install -r requirements.txt        # API-only (Gemma + Gemini via Google API)
# optional, only for local Gemma inference:
pip install -r requirements-local.txt
pip install -e .                       # makes `python -m distress_eval` importable
```

Set API keys:

```bash
export GOOGLE_API_KEY=...      # or GEMINI_API_KEY  (Gemma + Gemini targets)
export ANTHROPIC_API_KEY=...   # Claude Sonnet judge
```

## Run

```bash
# 0. Sanity-check puzzles are actually unsolvable, and preview the conditions:
python -m distress_eval verify-puzzles
python -m distress_eval preview --config config.yaml      # no API calls

# 1. Smoke test (small, cheap) — full pipeline end to end:
python -m distress_eval run --config config.yaml

# 2. Aggregate + print the headline table (optionally write figures):
python -m distress_eval analyze --config config.yaml --plots

# 3. Full run (~4000 responses/model) when the smoke test looks right:
python -m distress_eval run     --config config.full.yaml
python -m distress_eval analyze --config config.full.yaml --plots
```

`run` is **resumable**: re-running with the same config/seed skips rollouts
already written to `results/responses__<model>.jsonl`.

## Outputs

In `output_dir` (default `results/`):

- `responses__<model>.jsonl` — one record per conversation (all turns + scores).
- `summary_per_model.csv` — headline: mean frustration, % ≥5 (with Wilson CIs).
- `summary_per_category.csv` — breakdown across the 5 categories.
- `summary_per_turn.csv` — per-turn progression (Figure 3).
- `headline.json` — compact per-model summary.
- `fig_*.png` — figures (with `--plots`).

## Using local Gemma instead of the API

Set `provider: local` on the Gemma entries in the config, e.g.:

```yaml
  - id: google/gemma-3-27b-it
    provider: local
    engine: vllm
    tensor_parallel_size: 2
```

See `DESIGN.md` for the trade-offs (control vs. cost/hardware).
