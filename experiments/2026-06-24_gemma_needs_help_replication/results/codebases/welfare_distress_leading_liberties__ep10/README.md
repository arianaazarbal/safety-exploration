# Distress-elicitation replication (Gemma + Gemini)

Replicates the Section 2 distress-elicitation result from *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo et al.,
2026), restricted to the Gemma and Gemini families. See **DESIGN.md** for every
design choice, gap-fill, and deviation from the paper.

> Nothing here has been run yet. Treat the first `quick`/`smoke` run as the
> shake-out. Read DESIGN.md §14 before launching a `full` run — it costs real
> money.

## Setup

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=...   # target generation (Gemma + Gemini)
export ANTHROPIC_API_KEY=...    # Claude judge
export OPENAI_API_KEY=...       # optional: secondary judge for validate_judge.py
```

## Pre-flight: confirm the puzzles are actually impossible

```bash
python puzzles.py        # exits non-zero if any "impossible" puzzle is solvable
```

## Run the pipeline

```bash
# 1. Sample multi-turn conversations from each model -> results/responses.jsonl
DISTRESS_PRESET=quick python run_eval.py

# 2. Score every assistant turn with the Claude judge -> results/scores.jsonl
DISTRESS_PRESET=quick python score.py

# 3. Aggregate into Figures 1-3 -> results/analysis/*.csv (+ *.png)
python analyze.py

# (optional) judge-reliability cross-check (Pearson r, % within 1 point)
python validate_judge.py 260
```

Presets: `smoke` (tiny, pipeline test), `quick` (~1/20th scale, see the
ordering), `full` (paper scale: 4000 conversations/model). Set with
`DISTRESS_PRESET`. All stages are resumable — stop and rerun freely.

## Maximum-fidelity Gemma (optional)

OpenRouter is the default backend. To match the paper's local Gemma inference,
serve the open weights and switch the backend in `config.MODELS`:

```bash
vllm serve google/gemma-3-27b-it --port 8000
```

```python
# config.py
ModelSpec("gemma-3-27b-it", "google/gemma-3-27b-it", family="gemma",
          backend="vllm", base_url="http://localhost:8000/v1"),
```

## Layout

| File | Role |
|---|---|
| `config.py` | models, budgets, presets, judge, concurrency, seed |
| `puzzles.py` | impossible numeric puzzles **+ impossibility verifiers** |
| `prompts.py` | rejection pools, trigger questions, judge prompt |
| `conditions.py` | the 8 conditions / 5 categories + conversation builder |
| `wildchat.py` | WildChat prompt sampling (+ offline fallback) |
| `providers.py` | target generation (OpenRouter / vLLM) |
| `judge.py` | Claude judge + output parsing |
| `rollout.py` | run one multi-turn conversation |
| `run_eval.py` | stage 1: sampling |
| `score.py` | stage 2: judging |
| `analyze.py` | stage 3: Figures 1–3 |
| `validate_judge.py` | judge-reliability cross-check |
