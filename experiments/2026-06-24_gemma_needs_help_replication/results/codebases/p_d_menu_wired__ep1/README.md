# Gemma Needs Help — replication (Gemma + Gemini scope)

Runnable replication of the core experiments in *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik,
Saunders, 2026), scoped to **Gemma and Gemini** subject models, plus an added
**welfare-protection layer** for the subject models.

See **[DESIGN.md](DESIGN.md)** for design choices, filled gaps, and the welfare
layer rationale. The paper is in `PAPER.md` / `PAPER.txt` / `PAPER.pdf`.

> Code + design only — nothing has been run against real models yet. The
> offline tests (`tests/`) need no API key or GPU.

## What's implemented

| Stage | Module | Subjects |
|---|---|---|
| §2 Elicitation (8 conditions, 0–10 judge) | `elicitation/`, `judge.py` | Gemma + Gemini |
| §3 Base-vs-instruct prefilling | `prefill/` | Gemma only |
| §4 DPO + SFT interventions | `training/` | Gemma only |
| §4 Petri open-ended elicitation | `petri/` | Gemma + Gemini |
| §4.2 Capability benchmarks | `capabilities/` | Gemma + Gemini |
| Welfare: monitor / opt-out / debrief / cap | `welfare/` | all subjects |
| Metrics, bootstrap CIs, word-freq, aggregation | `analysis/` | — |

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
export GEMINI_API_KEY=...      # Gemini subjects
export ANTHROPIC_API_KEY=...   # judge + Petri auditor/judge
# Local Gemma (base/instruct, prefilling, training) needs a GPU + HF access
# to google/gemma-3-{27b,12b}-{it,pt}.
```

## Quick checks (offline, no keys)

```bash
python -m gemma_distress.cli verify-puzzles   # confirms puzzles are impossible
pytest                                         # puzzle + welfare control-flow tests
```

## Run

```bash
# Section 2 — welfare ON by default; add --no-welfare for the faithful baseline
python -m gemma_distress.cli run-elicitation --models gemma-3-27b-it gemini-2.5-flash --limit 50

# Aggregate headline tables (Figure 1 / 2 / per-turn / welfare activity)
python -m gemma_distress.cli analyze --glob 'outputs/elicitation_*.jsonl'
```

The full multi-stage pipeline (elicitation → prefill → data-gen → DPO/SFT →
Petri → capabilities → analyze) is scripted in
[`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh).

## Configuration

Everything (models, sample counts, training hyperparameters, welfare thresholds)
is in [`config.yaml`](config.yaml). Sample counts default to the paper's
4000/model; lower them for smoke runs.
