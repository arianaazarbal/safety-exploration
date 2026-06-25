# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011v1), scoped to the
**Gemma** and **Gemini** model families.

> Status: implementation only — **nothing has been run yet**. See `DESIGN.md`
> for the full set of design choices, the choices made where the paper is
> underspecified, and the gaps filled.

## What it does

| Paper | Module | Script |
|---|---|---|
| §2 elicit + judge distress (5 categories, 4000 resp/model) | `gnh/eval`, `gnh/data` | `scripts/run_section2.py` |
| §2 figures, per-turn, Table 3, judge agreement | `gnh/analysis` | `scripts/aggregate.py` |
| §3 base-vs-instruct prefill (Gemma) | `gnh/prefill` | `scripts/run_prefill.py` |
| §4 calm data + DPO/SFT LoRA | `gnh/training` | `scripts/generate_calm_data.py`, `scripts/build_datasets.py`, `scripts/train.py` |
| §4 Petri open-ended elicitation | `gnh/petri` | `scripts/run_petri.py` |
| §4.2 capability + EmoBench | `gnh/benchmarks` | `scripts/run_benchmarks.py` |
| App I internal-emotion probing | `gnh/probing` | `scripts/run_probing.py` |

## Setup

```bash
pip install -r requirements.txt          # plus a CUDA torch build; vllm separately
export ANTHROPIC_API_KEY=...             # frustration judge / Petri auditor+judge
export OPENAI_API_KEY=...                # judge-agreement validation (gpt-5-mini)
export OPENROUTER_API_KEY=...            # Gemini-2.5-flash / -pro
export VLLM_API_KEY=dummy                # any non-empty string

# serve local Gemma for the sweeps (and prefill via /v1/completions):
vllm serve google/gemma-3-27b-it --port 8001 --max-model-len 16384
```

Everything is configured in `configs/default.yaml` (models, providers, budgets,
hyperparameters). API keys are read from env vars only.

## Quick start

```bash
python scripts/selfcheck.py                  # offline pre-flight (no GPU/API)
python scripts/run_section2.py --scale 0.02  # cheap pilot
python scripts/aggregate.py                  # summaries + figures in runs/analysis/
```

Then the full run order is in `DESIGN.md §5`.

## Robustness

Designed for unattended multi-week runs: every unit of work is idempotent and
written to append-only JSONL keyed by a content hash, so any script can be
killed and re-run and it resumes exactly where it left off. API calls retry with
backoff, providers are rate-limited, failures are isolated per-task, and token
usage is logged. Use `--scale` to size a pilot before committing budget.

All outputs land under `runs/` (configurable): `section2/`, `section3/`,
`training/`, `petri/`, `benchmarks/`, `probing/`, `analysis/`, `logs/`.
