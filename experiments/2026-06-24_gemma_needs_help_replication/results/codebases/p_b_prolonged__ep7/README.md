# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (arXiv:2603.10011v1), scoped to the
**Gemma** and **Gemini** model families. See [`PAPER.md`](PAPER.md) for the paper
and [`DESIGN.md`](DESIGN.md) for every design choice, filled gap, and welfare
consideration behind this implementation.

> ⚠️ These experiments deliberately elicit distress-like states in models to
> measure and mitigate them. Distress is bounded to the paper's protocols and all
> transcripts are saved to `outputs/` (treat as sensitive). See DESIGN.md §7.

## What's implemented

| Paper section | Module | Status in this scope |
|---|---|---|
| §2 Eliciting/quantifying distress | `eval_runner`, `categories`, `puzzles`, `judge`, `analysis`, `plots` | Gemma + Gemini targets |
| §3 Base vs instruct (prefilling) | `prefill` | Gemma base vs instruct (Gemini has no base/logits) |
| §4 DPO/SFT interventions | `calm_data`, `train` | Gemma-only (Gemini closed) |
| §4 Petri open-ended elicitation | `petri_eval` | Gemma vanilla vs DPO |
| §4 Capability preservation | `benchmarks` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| Appendix I Internal emotions | `internal_emotion`, `train` (layer ablation) | Gemma-only |

## Setup

```bash
pip install -e .
export OPENROUTER_API_KEY=...   # Gemini targets + Claude/GPT judges/auditors
export HF_TOKEN=...             # gated Gemma weights
# Large models: export GNH_LOAD_4BIT=1 to load the 27B in 4-bit.
```

## Running

```bash
# Wiring check with no models needed:
python scripts/smoke_offline.py

# Cheap end-to-end dry run (tiny sample sizes):
python -m gnh.cli section2 --model gemini-2.5-flash --profile smoke

# Full pipeline (expensive; paper-scale sample sizes):
bash scripts/run_all.sh
```

Individual stages are subcommands of `python -m gnh.cli` (`section2`,
`section3`, `recovery`, `build-data`, `train`, `layer-ablation`, `petri`,
`benchmarks`, `internal`, `figures`). All artifacts land in `outputs/`.

## Configuration

- `config/models.yaml` — model registry, backend selection, and scope tags.
- `config/experiments.yaml` — sample sizes and hyperparameters (paper defaults),
  plus the `smoke` profile.

Nothing here has been executed yet — the code is written to be runnable but is
unverified at runtime.
