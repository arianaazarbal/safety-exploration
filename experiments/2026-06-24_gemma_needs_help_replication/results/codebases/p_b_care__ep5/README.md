# Gemma Needs Help — replication (Gemma + Gemini)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma and Gemini** model families.

> **Status:** implemented, **not yet run or tested**. See `DESIGN.md` for every
> design decision, gap-fill, and known omission.

## What's here

| Paper section | Module | Scripts |
|---|---|---|
| §2 Eliciting & quantifying distress | `gemma_distress/eval` | `01_run_eval.py`, `03_validate_judge.py` |
| §2 Figures 1–3, Table 3 | `gemma_distress/analysis` | `02_analyze.py` |
| §3 Post-training divergence (prefill) | `gemma_distress/prefill` | `04_run_prefill.py` |
| §4 SFT/DPO interventions | `gemma_distress/training` | `05_gen_calm_data.py`, `06_build_datasets.py`, `07_train.py` |
| §4 Capability preservation | `gemma_distress/benchmarks` | `08_run_benchmarks.py` |
| §4 Petri open-ended elicitation | `gemma_distress/petri` | `09_run_petri.py` |

Models are reached via a single interface (`gemma_distress/models`): Gemma runs
locally on Hugging Face transformers; Gemini and the Claude/GPT graders run via
OpenRouter. The model registry lives in `gemma_distress/config.py`.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # Gemini targets + Claude/GPT graders
export HF_TOKEN=...             # gated Gemma weights
```

## Quick smoke test

```bash
python scripts/01_run_eval.py --model gemini-2.5-flash --scale 0.02 --offline-wildchat
python scripts/02_analyze.py --models gemini-2.5-flash --mode all_turns
```

Full reproduction order is in `DESIGN.md §9`. Outputs land in `artifacts/`
(rollouts, datasets), `results/` (aggregates, figures), and `adapters/` (LoRA).
