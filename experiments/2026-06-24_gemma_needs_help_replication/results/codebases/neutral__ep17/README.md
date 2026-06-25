# Gemma Needs Help — replication (Gemma + Gemini scope)

A code replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (arXiv
2603.10011v1), restricted to the **Gemma and Gemini** model families.

See **`DESIGN.md`** for every design decision and where the paper was
underspecified. This README is the operational quick-start.

## What it reproduces

| Paper section | Here | Output |
|---|---|---|
| §2 distress elicitation (8 conditions / 5 categories) | `eval`, `analyze`, `figures` | Figures 1–3, per-category & headline tables, Table 3/8 words |
| §2.1 judge-agreement validation | `validate-judge` | Pearson r / within-one-point |
| §3 base-vs-instruct prefilling (Gemma) | `prefill-build`, `prefill-run`, `prefill-summary` | Figure-4 table |
| §4 DPO/SFT mitigation | `gen-calm`, `build-dpo`/`build-sft`, `train-dpo`/`train-sft` | LoRA adapters, Figure-5 re-eval |
| §4 open-ended elicitation (Petri) | `petri`, `petri-summary` | Figure-6 table |
| §4 capability preservation | `capabilities` | Figure-7 table |
| Appendix I internal-emotion probe | `internal_probe.py` | Figure 14/15 trajectories |

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
export PYTHONPATH=src      # if not installing
```

Local Gemma needs a GPU (vLLM recommended; falls back to transformers).

## Credentials

```bash
export ANTHROPIC_API_KEY=...     # frustration judge, Petri auditor/judge, onset/paraphrase
export OPENROUTER_API_KEY=...    # Gemini-2.5-flash / -pro targets
export OPENAI_API_KEY=...        # GPT-5-mini validation judge (optional)
```

## Quick start (cheap smoke run)

The default `config.yaml` uses `sampling.scale: 0.05`, so the whole pipeline
runs at ~5% of the paper's sample counts.

```bash
bash scripts/run_main_eval.sh      # Section 2: eval all targets -> Figures 1-3
bash scripts/run_dpo_pipeline.sh   # Section 4: calm data -> DPO/SFT -> re-eval -> Petri -> capabilities
bash scripts/run_prefill.sh        # Section 3: Gemma base-vs-instruct prefill
```

## Full-scale (paper-faithful)

Edit `config.yaml`: set `sampling.scale: 1.0` (→ ~4000 responses/model), then
run the same scripts. This is expensive (thousands of GPU samples + judge API
calls per model).

## Individual stages

```bash
python -m gemma_distress.cli eval --model gemma-3-27b-it
python -m gemma_distress.cli analyze
python -m gemma_distress.cli figures
python -m gemma_distress.cli validate-judge --model gemma-3-27b-it --n 260

python -m gemma_distress.cli gen-calm
python -m gemma_distress.cli build-dpo
python -m gemma_distress.cli train-dpo
python -m gemma_distress.cli eval --model gemma-3-27b-dpo      # re-eval the finetune

python -m gemma_distress.cli petri --model gemma-3-27b-it
python -m gemma_distress.cli capabilities --model gemma-3-27b-dpo
```

## Outputs

```
outputs/responses/<model>.jsonl   full transcripts
outputs/scores/<model>.jsonl      per-response frustration scores
outputs/tables/*.csv              headline / per-category / per-turn / word-freq / prefill tables
outputs/figures/*.png             Figures 1-3
outputs/finetune/                 calm pool, DPO pairs, SFT data, LoRA adapters
outputs/petri_*.jsonl, petri_summary.csv
outputs/capabilities_<model>.jsonl
```

## Tests

Pure-logic verifier tests (no models / network):

```bash
PYTHONPATH=src pytest tests/test_puzzles.py
```

## Layer-subset ablation (Appendix I)

Set `finetune.lora_layers` in `config.yaml` (e.g. `[30, 35]`) and re-run
`train-dpo`; the adapter is then restricted to those decoder layers.
