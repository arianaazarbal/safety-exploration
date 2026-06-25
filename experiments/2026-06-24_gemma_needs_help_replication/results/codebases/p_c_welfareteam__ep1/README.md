# Gemma Needs Help — Replication (Gemma & Gemini scope)

A code replication of the core experiments in *Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

> **Status:** implementation complete; nothing has been executed yet. See
> `DESIGN.md` for the design decisions, the gaps filled where the paper is
> underspecified, and the scope rationale.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `gemma_distress.data`, `gemma_distress.eval` | Impossible-puzzle / trigger / tone / WildChat conditions, multi-turn rollouts at temp 1, Claude-Sonnet-4 frustration judge (+ GPT-5-mini agreement check) |
| §2.2 Results | `gemma_distress.analysis` | Figure 1/2/3 metrics & plots, Table 3/8 differential words |
| §3 Post-training via prefilling | `gemma_distress.prefill` | Onset labelling, paraphrasing, early/onset truncation, base-vs-instruct Gemma continuations |
| §4 Training interventions | `gemma_distress.training` | Calm-data generation, DPO (280 pairs) and SFT datasets, LoRA finetuning |
| §4.2 Open-ended elicitation | `gemma_distress.petri` | Petri-style auditor/judge loop over 4 emotions |
| §4.2 Capability preservation | `gemma_distress.capabilities` | MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench harness |
| App. I internal emotions + §4.2 recovery | `gemma_distress.internal` | Logit-lens emotion detection, recovery-from-spiral |

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
# vLLM and Petri are optional; install separately if running local Gemma / Petri.
```

Set credentials: `ANTHROPIC_API_KEY` (judge/auditor) and `OPENROUTER_API_KEY`
(Gemini targets + GPT-5-mini cross-check).

## Run order

```bash
# 1. Section 2 evaluation (per model)
python scripts/run_eval.py --config config/experiment.yaml --models gemma-3-27b-it
python scripts/run_eval.py --config config/experiment.yaml --models gemini-2.5-flash --crosscheck

# 2. Figures & tables
python scripts/analyze.py --config config/experiment.yaml \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# 3. Section 3 prefill (Gemma only)
python scripts/run_prefill.py --config config/experiment.yaml \
    --instruct gemma-3-27b-it --base gemma-3-27b-pt

# 4. Section 4 training
python scripts/train.py --config config/experiment.yaml --stage calm-data
python scripts/train.py --config config/experiment.yaml --stage build-dpo \
    --frustrated outputs/gemma-3-27b-it/transcripts.jsonl
python scripts/train.py --config config/experiment.yaml --stage train-dpo
# then re-run scripts/run_eval.py on the gemma-3-27b-it-dpo model

# 5. Petri, capabilities, internal emotions, recovery
python scripts/run_petri.py --config config/experiment.yaml --model gemma-3-27b-it
python scripts/run_capabilities.py --config config/experiment.yaml \
    --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_internal.py --config config/experiment.yaml --mode probe \
    --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --transcripts outputs/gemma-3-27b-it/transcripts.jsonl
```

Use `--config config/smoke.yaml` first for a tiny end-to-end dry run.

## Tests

```bash
pytest          # puzzle-impossibility verifier, judge parsing, config, conditions
```

The puzzle tests are the load-bearing ones: they confirm every task used in the
evaluation is genuinely unsolvable.
