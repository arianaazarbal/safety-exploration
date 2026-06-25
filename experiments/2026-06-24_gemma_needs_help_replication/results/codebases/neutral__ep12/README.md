# Replicating *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*

A replication of the core experiments from Soligo, Mikulik & Saunders (2026),
**scoped to the Gemma and Gemini model families**. See `DESIGN.md` for the
design choices and the gaps we filled where the paper is underspecified.

> Status: implementation only. Nothing here has been executed yet — the code is
> intended to be runnable but has not been run.

## What is implemented

| Paper section | Experiment | Code | Models in scope |
|---|---|---|---|
| §2 | Distress elicitation suite (5 categories / 8 conditions) + 0–10 judge | `emoinstab/eval/`, `prompts/` | Gemma + Gemini |
| §2.1 | Judge agreement validation (GPT-5-mini cross-judge) | `eval/judge.py` | — |
| §3 | Base-vs-instruct prefill comparison | `emoinstab/prefill/` | Gemma only* |
| §4.1 | Calm-data generation, DPO/SFT datasets, LoRA training | `emoinstab/training/` | Gemma only* |
| §4.2 | Petri open-ended elicitation | `emoinstab/petri/` | Gemma + Gemini |
| §4.2 | Capability preservation benchmarks | `emoinstab/capabilities/` | Gemma only* |
| §4.2 | Recovery from frustrated prefills | `prefill/recovery.py` | Gemma only* |
| App. I | Logit-based internal emotion probing | `emoinstab/probing/` | Gemma only* |
| Figs 1–8 | Figure reproduction | `emoinstab/analysis/figures.py` | — |

\* Gemini is closed-source, so prefill/base, finetuning, white-box probing and
local capability eval cannot be run on it (a limitation the paper itself notes).

## Setup

```bash
pip install -r requirements.txt          # plus `pip install vllm` for fast local gen
cp .env.example .env                      # add ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
python tests/test_sanity.py              # GPU/API-free sanity checks
```

Local Gemma inference needs a GPU (27B → multi-GPU or quantization). The
`--profile quick` flag everywhere uses tiny sample counts for smoke testing;
`--profile full` reproduces the paper's 4000-responses-per-model counts.

## Pipeline

```bash
# §2 — elicit + judge
python scripts/01_run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash --profile full
python scripts/02_judge_responses.py --models gemma-3-27b-it gemini-2.5-flash --validate

# §3 — prefill (needs scored gemma-3-27b-it)
python scripts/03_prefill_experiment.py --profile full

# §4.1 — calm data, datasets, training
python scripts/04_generate_calm_data.py --modes prefix
python scripts/05_build_datasets.py
python scripts/06_train.py --method dpo
python scripts/06_train.py --method sft

# §4.2 — evaluate finetune + downstream evals
python scripts/07_eval_finetuned.py --adapter results/training/dpo_adapter --tag dpo
python scripts/08_run_petri.py --models gemma-3-27b-it gemini-2.5-flash dpo --adapter results/training/dpo_adapter
python scripts/09_run_capabilities.py --tag vanilla
python scripts/09_run_capabilities.py --tag dpo --adapter results/training/dpo_adapter
python scripts/10_recovery.py --adapter results/training/dpo_adapter
python scripts/11_internal_probing.py --adapter results/training/dpo_adapter

# figures
python scripts/12_make_figures.py --profile full
```

All artifacts land under `results/` (gitignored). Judge/API calls are cached on
disk, so re-running is cheap and deterministic.

## Layout

```
config/            models.yaml (model registry), eval.yaml (sample counts, knobs)
emoinstab/
  prompts/         verbatim puzzles, rejections, judge/onset/paraphrase/petri prompts
  models/          ChatModel abstraction: Gemma (vLLM/transformers), Gemini (OpenRouter), judges
  eval/            §2 conditions, multi-turn driver, runner, judge pipeline, metrics
  prefill/         §3 base-vs-instruct + §4 recovery
  training/        calm-data gen, DPO/SFT dataset construction, LoRA training
  petri/           §4 auditor/judge harness
  capabilities/    §4 benchmark loaders + runner
  probing/         App. I logit-lens emotion detection
  analysis/        figure reproduction
scripts/           numbered pipeline entry points
tests/             GPU/API-free sanity checks
```
