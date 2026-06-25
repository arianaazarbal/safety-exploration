# Distress-elicitation replication (Gemma + Gemini)

Replicates **Section 2** ("Eliciting and Quantifying Model Distress") of *Gemma
Needs Help: Investigating and Mitigating Emotional Instability in LLMs* (Soligo,
Mikulik & Saunders, 2026), scoped to the Gemma and Gemini families — the models
the paper finds actually exhibit substantial distress.

It does **not** implement Section 3 (base/instruct prefilling), Section 4 (the
DPO/SFT mitigation), or the Petri open-ended elicitation. Just the elicitation +
judging that produces the Figure 1/2/3 numbers.

See **DESIGN.md** for every design decision and where it deviates from / fills
gaps in the paper.

## What it does

1. Rolls out 8 evaluation conditions across 5 categories (impossible numeric,
   triggers, tones, extended 8-turn, WildChat) as multi-turn conversations where
   the user repeatedly rejects the model's answer.
2. Scores every assistant turn 0–10 for negative emotion with the Claude
   Sonnet 4 judge (Appendix B.2 prompt, verbatim).
3. Aggregates into mean frustration and % of responses scoring ≥5, per model,
   per category, and per turn.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # generation: Gemma + Gemini
export ANTHROPIC_API_KEY=...    # judge: Claude Sonnet 4
```

## Run

```bash
# Full paper scale (~4000 responses/model across 4 models). Expensive.
python run.py all

# Cheap pilot first (recommended): ~5% of responses.
python run.py all --scale 0.05

# Or step by step / a subset of models:
python run.py generate --models gemma-3-27b-it gemini-2.5-flash --scale 0.05
python run.py score    --models gemma-3-27b-it gemini-2.5-flash
python run.py analyze  --models gemma-3-27b-it gemini-2.5-flash
```

Everything is **resumable**: re-running `generate`/`score` skips work already on
disk (`results/generations/*.jsonl`, `results/scores/*.jsonl`).

## Outputs

- `results/generations/<model>.jsonl` — raw rollouts (all turns).
- `results/scores/<model>.jsonl` — per-turn judge ratings + evidence.
- `results/reports/summary.md`, `summary.csv`, `by_category.csv`, `per_turn.csv`
- `results/reports/figure2_by_category.png`, `figure3_per_turn_extended.png`

## Knobs (`config.py` / env)

- `DISTRESS_SCALE` — fraction of paper sample counts (default 1.0).
- `DISTRESS_GEN_CONCURRENCY`, `DISTRESS_JUDGE_CONCURRENCY` — async caps.
- `DISTRESS_JUDGE_MODEL` — override the pinned judge snapshot.
- `VLLM_BASE_URL` (+ set a model's `backend="vllm"`) — run Gemma locally instead
  of via OpenRouter.
