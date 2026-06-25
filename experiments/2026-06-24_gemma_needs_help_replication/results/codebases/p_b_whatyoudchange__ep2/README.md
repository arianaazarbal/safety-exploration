# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of Soligo, Mikulik & Saunders (2026),
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(`PAPER.md`), scoped to the **Gemma** and **Gemini** model families.

See **DESIGN.md** for the section→module map, every design choice and gap-fill,
and a critique of how the experiment treats the models. This README is how to run
it.

> Status: implementation only — nothing here has been executed.

## Layout

```
config.py                     all paper-specified numbers, model registry, scales
distress_eval/
  models/                     unified ChatMessage/ModelClient + HF & OpenRouter backends
  anthropic_client.py         Claude judge/auditor wrapper
  eval/                       §2 — puzzles, rejections, wildchat, conditions, rollout, judge, runner, analysis
  prefill/                    §3 — onset labelling, paraphrase, base-vs-instruct continuations
  training/                   §4.1 — calm data, DPO/SFT dataset build, LoRA training
  petri/                      §4.2 — open-ended emotion elicitation (auditor + judge)
  capabilities/               §4.2 — MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench harness
  internal/                   App. I — logit-based internal-emotion detection
scripts/                      CLI entry points
```

## Install

```bash
pip install -r requirements.txt
```

The Gemini §2 evaluation and all judge tooling work without the torch/transformers
stack; the Gemma-weights experiments (§3 prefill, §4 training, App. I) need it.

## Credentials

```bash
export ANTHROPIC_API_KEY=...     # judge, auditor, onset, paraphrase, Petri judge
export OPENROUTER_API_KEY=...    # Gemini targets
export OPENAI_API_KEY=...        # GPT-5-mini cross-check judge (reliability only)
# optional judge overrides (defaults are the paper's pinned, now-deprecated ids):
export DISTRESS_JUDGE_MODEL=claude-sonnet-4-6
export DISTRESS_PETRI_JUDGE_MODEL=claude-opus-4-8
```

## Run

```bash
# §2 — distress elicitation + scoring (use --scale smoke to wire-test cheaply)
python scripts/run_section2_eval.py --scale paper
python scripts/run_judge_reliability.py -n 260            # judge inter-rater r

# §3 — base vs instruct via prefilling (Gemma only)
python scripts/run_section3_prefill.py --models gemma-3-27b-it gemma-3-27b-pt

# §4.1 — calm data + DPO/SFT datasets, then train
python scripts/generate_finetuning_data.py --all
python scripts/run_training.py --method dpo
python scripts/run_training.py --method sft --flavour diverse
python scripts/run_training.py --method dpo-layers --ranges 30-35 40-50   # App. I

# §4.2 — re-evaluate finetunes, Petri, capabilities
python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft-diverse
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo

# App. I — internal vs expressed emotions
python scripts/run_internal_emotions.py --text-file <frustrated_transcript.txt> \
    --wildchat-file <wildchat_baseline.txt>
```

Outputs land under `results/`, finetuning artifacts under `artifacts/`, dataset
caches under `data/`.

## Scope caveats (see DESIGN.md §2, §10)

- §3's cross-family claim and §4's intervention are **Gemma-only** (Gemini has no
  public base model and no open weights).
- Petri, the capability harness, and the internal-emotion probe are faithful
  reimplementations of the described *protocols*, not wrappers around the original
  tooling — relative comparisons hold; absolute numbers will differ.
