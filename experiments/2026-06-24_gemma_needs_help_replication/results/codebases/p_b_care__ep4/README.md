# Emotional Instability in LLMs — Replication (Gemma & Gemini)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011), scoped to the
**Gemma** and **Gemini** model families.

See **`DESIGN.md`** for the full mapping of paper sections → modules and the
rationale behind every choice made where the paper is underspecified.

> Status: implementation only. Nothing has been run yet.

## Install

```bash
pip install -e .            # installs deps from requirements.txt, puts the package on path
```

## Credentials

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | Gemini targets + Claude/GPT judges & auditor (OpenRouter) |
| `HF_TOKEN` | downloading gated Gemma weights / datasets |

Local Gemma inference and all of §3/§4/Appendix I need a GPU (gemma-3-27b fits a
single 48GB card in bf16, or ~24GB with `local.load_in_4bit: true`).

## Pipeline

```bash
# §2 — elicit + judge distress (Figures 1-3, Table 3)
python scripts/run_eval.py                              # all targets
python scripts/run_eval.py --models gemma-3-27b-it --backend openrouter   # no GPU
python scripts/make_figures.py

# §3 — base vs instruct via prefilling (Gemma only; Figure 4)
python scripts/run_prefill.py --prepare
python scripts/run_prefill.py --run --summarize

# §4 — DPO/SFT mitigation (Gemma only; Figure 5)
python scripts/run_training.py calm
python scripts/run_training.py datasets
python scripts/run_training.py dpo
python scripts/run_training.py sft --variant diverse
python scripts/run_eval.py --models gemma-3-27b-dpo     # re-evaluate the finetune
python scripts/make_figures.py --models gemma-3-27b-it gemma-3-27b-dpo

# §4.1 — Petri open-ended elicitation (Figure 6)
python scripts/run_petri.py --run --summarize

# §4.2 — capability preservation (Figure 7)
python scripts/run_benchmarks.py

# Appendix I — internal vs expressed emotion (vanilla vs DPO)
python scripts/run_training.py ablation
python scripts/run_internal_emotions.py
```

All outputs land under `artifacts/` (configurable in `config/default.yaml`). Every
phase is resumable — re-running skips completed work.

## Configuration

`config/default.yaml` is the single source of truth: model registry, the
4000-responses sampling plan, judge models, training hyperparameters (Table 9),
Petri/benchmark settings, and output paths. Pass `--config path.yaml` to any script
to override.
