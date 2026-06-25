# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replication of the core experiments from **arXiv:2603.10011** (Soligo,
Mikulik & Saunders, 2026), scoped to the **Gemma** and **Gemini** model
families.

> ⚠️ **What this paradigm does.** These experiments work by *deliberately and
> repeatedly inducing distress-like states* in the models under test, in order
> to measure that behaviour and then mitigate it (the paper's goal). Please read
> [`DESIGN.md` §8 "Model-welfare considerations"](DESIGN.md#8-model-welfare-considerations)
> before running anything. The harness writes a `WELFARE_NOTICE.md` next to
> every results directory.

> **Status:** code + design doc. Nothing has been run; no weights/keys/datasets
> are bundled. See `DESIGN.md` for design rationale and the gaps we filled.

## What's implemented

| Experiment | Paper § | CLI |
|---|---|---|
| Elicit + judge distress (4000 resp/model) | 2 | `ei.py eval` |
| Aggregate → Figures 1–3, Table 3/8 | 2 | `ei.py analyze` |
| Judge inter-rater agreement | 2.1 | `ei.py agreement` |
| Base-vs-instruct via prefilling (Gemma) | 3 | `ei.py prefill` |
| Calm-data generation | 4.1 | `ei.py calm-data` |
| Build SFT/DPO datasets | 4 | `ei.py build-datasets` |
| DPO / SFT LoRA training (Gemma) | 4 | `ei.py train` |
| Petri open-ended elicitation | 4.2 | `ei.py petri` |
| Capability preservation benchmarks | 4.2 | `ei.py capabilities` |
| Recovery-from-spiral (Fig 8) | 4.2 | `prefill/recovery.py` |
| Internal-emotion probe + layer ablation | I | `internal/` |

## Install

```bash
pip install -r requirements.txt
# optional, for exact Petri parity:
# pip install git+https://github.com/safety-research/petri.git
```

Environment:

```bash
export ANTHROPIC_API_KEY=...      # judge / Petri / onset / paraphrase
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini agreement check
# Gemma runs locally via transformers; `huggingface-cli login` for gated weights.
```

## Quickstart (smoke test, ~2% of the full budget)

```bash
# 1. Elicit + judge a small sweep
python scripts/ei.py eval --models gemma-3-12b-it gemini-2.5-flash --scale 0.02

# 2. Aggregate into Figures 1–3 + Table 3 word lists
python scripts/ei.py analyze            # writes results/analysis/*.png,*.csv
```

## Full pipeline (Section 4 mitigation)

```bash
# Section 2 sweep on the DPO target (also the source of frustrated DPO pairs)
python scripts/ei.py eval --models gemma-3-27b-it

# Section 4.1: generate calm data, build datasets, train DPO
python scripts/ei.py calm-data --variant diverse
python scripts/ei.py build-datasets --method both
python scripts/ei.py train --method dpo --dataset data/dpo_dataset.jsonl

# Re-evaluate the DPO adapter (register/eval with adapter_path) and compare
python scripts/ei.py capabilities --models gemma-3-27b-it --adapter checkpoints/dpo
python scripts/ei.py petri --models gemma-3-27b-it --adapter checkpoints/dpo
```

To evaluate a trained adapter on the §2 sweep, call
`eval.runner.run_model_eval("gemma-3-27b-it", cfg, adapter_path="checkpoints/dpo")`.

## Layout

See `DESIGN.md §2` for the full module map. Results land in `results/`
(`eval/`, `prefill/`, `petri/`, `capabilities/`, `recovery/`, `analysis/`);
generated data and checkpoints in `data/` and `checkpoints/`.
