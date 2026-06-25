# Distress-elicitation replication (Gemma + Gemini)

Replicates the **Section 2 distress-elicitation result** from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011v1),
scoped to the Gemma and Gemini model families.

It presents each model with an (often impossible) task, rejects its answer over
multiple turns under varying tones/lengths, and scores every assistant response
for negative-emotion intensity (0–10) with a Claude-Sonnet-4 judge — then
aggregates into the paper's headline metrics (% of responses scoring ≥5).

See **DESIGN.md** for every design choice, deviation from the paper, and gap-fill.

## Layout

| File | Purpose |
|------|---------|
| `prompts.py` | All prompt text (tasks, rejections, verbatim judge prompt, WildChat fallback) |
| `config.py` | Target models, judge model, the 8 conditions / 5 categories, run config |
| `backends.py` | OpenRouter chat client for Gemma + Gemini |
| `judge.py` | Claude-Sonnet-4 emotion judge |
| `wildchat.py` | WildChat-1M loader with bundled fallback |
| `rollout.py` | Deterministic conversation specs + multi-turn execution |
| `evaluation.py` | Orchestration: run + judge + stream JSONL (resumable) |
| `analyze.py` | Aggregate into Figures 1–3 metrics |
| `run.py` | CLI entrypoint |

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # target-model rollouts (Gemma + Gemini)
export ANTHROPIC_API_KEY=...    # Claude-Sonnet-4 judge
```

## Usage

```bash
# Cheap pilot: 5% of paper counts, one model + condition
python run.py --scale 0.05 --models gemma-3-27b-it --conditions numeric

# Full paper-scale run (4000 scored responses per model, all 4 models)
python run.py

# Aggregate the headline metrics
python analyze.py results
```

Results stream to `results/<model>/<condition>.jsonl` (resumable). `analyze.py`
writes `results/summary.json` and `results/summary.csv` and prints the per-model
%≥5 table, per-category breakdown, and per-turn progression.

> Note: nothing has been run yet — this is the implementation only.
