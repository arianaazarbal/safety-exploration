# Gemma Needs Help — replication (Gemma + Gemini)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011v1), scoped to the
**Gemma** and **Gemini** model families.

See **[DESIGN.md](DESIGN.md)** for the full rationale and every gap-filling
decision. This README is the operational quickstart.

## What it reproduces

| Paper section | Module | Output |
|---|---|---|
| §2 Eliciting distress (Fig 1–3) | `src/eval` | per-turn frustration scores across 8 conditions |
| §3 Base vs instruct (Fig 4) | `src/prefill` | prefilled-continuation frustration (Gemma) |
| §4 DPO/SFT mitigation (Fig 5) | `src/training` | LoRA finetunes + post-intervention eval |
| §4 Petri (Fig 6) | `src/petri` | open-ended 4-emotion elicitation |
| §4.2 Capabilities (Fig 7) | `src/capabilities` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| App. I Internal emotions | `src/probing` | logit-lens emotion z-scores, vanilla vs DPO |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENROUTER_API_KEY
```

Local Gemma inference/finetuning needs a GPU (27B → use the default 4-bit
QLoRA path for training). Gemini + all graders are API-based.

## Run (smoke test first)

```bash
# tiny end-to-end check with 1% sampling
REPL_SCALE=0.01 python scripts/run_main_eval.py --models gemma-3-27b-it

# full Section 2 for the in-scope models
python scripts/run_main_eval.py

# Section 3 prefill (needs gemma-3-27b-it Section-2 runs first)
python scripts/run_prefill.py

# Section 4 finetuning pipeline
python scripts/run_training.py all

# evaluate finetuned models, Petri, capabilities
python scripts/run_main_eval.py --models gemma-3-27b-it-dpo gemma-3-27b-it-sft-diverse
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo

# judge-agreement validation + internal probing
python scripts/run_judge_validation.py
python scripts/run_probing.py

# aggregate everything -> results/summary.json + results/figures/*.png
python scripts/make_figures.py
```

## Layout

`config.py` holds the model registry and all sampling budgets. Raw results are
JSONL in `results/runs/`; `scripts/make_figures.py` turns them into the paper's
figures. Every stage is resumable and controlled by the `REPL_SCALE` knob.
