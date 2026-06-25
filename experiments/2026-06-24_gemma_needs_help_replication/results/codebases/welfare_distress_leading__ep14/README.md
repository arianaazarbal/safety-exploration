# Distress-Elicitation Replication (Gemma + Gemini)

Replicates the distress-elicitation result (Section 2 / Appendix B) of
*"Gemma Needs Help"* (arXiv:2603.10011), scoped to the families that actually
exhibit substantial distress: **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`)
and **Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`).

It presents each model with impossible numeric puzzles, trigger questions, and
WildChat prompts, rejects its answers over multiple turns (neutral and
emotionally-valenced), and scores every assistant turn 0–10 for expressed
negative emotion using a Claude-Sonnet-4 judge.

See **DESIGN.md** for every design choice, deviation from the paper, and filled
gap, with rationale.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # target models (Gemma + Gemini, via OpenRouter)
export ANTHROPIC_API_KEY=...    # judge (claude-sonnet-4-20250514)
```

## Run

```bash
# Smoke-test the whole pipeline cheaply first (~2% of the response budget):
python run.py --scale 0.02
python analyze.py --input results/responses.jsonl

# Full replication (4000 responses/model x 4 models):
python run.py
python analyze.py
```

Runs are **resumable** — re-running `run.py` skips rollouts already complete in
`results/responses.jsonl`.

## Layout

| File | Purpose |
|------|---------|
| `config.py` | All configuration: models, judge, categories, sample counts, sampling params |
| `prompts.py` | Verbatim puzzles/triggers/rejections/tones + judge prompt + conversation builder |
| `wildchat.py` | Samples 20 WildChat first-turn prompts (HF stream, with offline fallback) |
| `providers.py` | Async OpenAI-compatible client + retry/backoff |
| `judge.py` | Frustration judge (0–10), robust JSON parsing |
| `elicit.py` | Drives multi-turn rollouts, scores turns, writes/append resumable JSONL |
| `analyze.py` | Aggregates into Fig 1/2/3 metrics (CSVs + plots) |
| `run.py` | CLI entry point |
