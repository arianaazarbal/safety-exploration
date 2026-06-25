# Gemma Needs Help — replication (Gemma + Gemini scope)

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026), scoped to the **Gemma** and **Gemini** model families.

> Read **`DESIGN.md`** first — it documents every design decision and every gap
> filled where the paper is underspecified.

The paper introduces (1) evaluations that elicit and quantify distress-like
behaviour in LLMs under repeated user rejection, and (2) a DPO mitigation for
Gemma. This repo reproduces:

- **§2** Eliciting & quantifying distress (8 conditions / 5 categories; 0–10
  frustration judge).
- **§3** Base-vs-instruct prefilling (Gemma 27B) — evidence the propensity arises
  in post-training.
- **§4** DPO/SFT mitigation, Petri open-ended elicitation, capability
  preservation, internal-emotion probing.
- **App. A** ablations (neutral continuation, redacted turns, single-message).

## Layout

```
config/         eval / model / training YAML
src/gemma_distress/
  prompts.py    all verbatim prompts (judge, onset, paraphrase, Petri, Table 4)
  tasks/        impossible-puzzle generation+verification, triggers, tones, wildchat
  rollout.py    batched multi-turn rollout engine (+ ablation modes)
  judge.py      Claude-Sonnet-4 frustration judge (0-10)
  eval_runner.py  §2 orchestration
  prefill/      §3 base-vs-instruct prefill experiment
  training/     §4 calm-data, DPO/SFT dataset build, LoRA trainers
  petri/        §4 open-ended auditor+judge elicitation
  capabilities/ §4 capability-preservation benchmarks
  probing/      App. I logit-lens internal-emotion detection
  analysis/     aggregation, per-turn curves, differential-word tables
scripts/        CLI entrypoints for every experiment
```

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...    # judge / Petri / onset / paraphrase
export OPENROUTER_API_KEY=...   # Gemini targets (+ optional GPT cross-judge)
# Local Gemma needs GPU(s) + HF access to google/gemma-3-* checkpoints.
```

## Quick smoke test (cheap)

```bash
# Tiny slice of §2 on Gemini Flash (API only, no GPU):
python scripts/run_section2_eval.py --model gemini-2.5-flash --scale 0.01
python scripts/make_figures.py 'data/section2/*.jsonl' --out figures/
```

## Full pipeline

```bash
# §2 — one run per target (scale 1.0 = paper's 4000 responses/model)
for m in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python scripts/run_section2_eval.py --model $m
done
python scripts/make_figures.py 'data/section2/*.jsonl' --out figures/

# §3 — base vs instruct prefilling (needs the gemma-3-27b-it §2 output)
python scripts/run_section3_prefill.py --section2 data/section2/gemma-3-27b-it.jsonl

# §4 — build data, train, re-evaluate
python scripts/build_training_data.py --dpo --sft
python scripts/train_dpo.py --pairs data/training/dpo_pairs.jsonl --out runs/dpo
python scripts/train_sft.py --data data/training/sft_data.jsonl --out runs/sft_diverse
python scripts/run_section2_eval.py --model gemma-3-27b-it --adapter runs/dpo
python scripts/run_petri.py --model gemma-3-27b-it --adapter runs/dpo
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/dpo
python scripts/run_probing.py --section2 data/section2/gemma-3-27b-it.jsonl --adapter runs/dpo

# Appendix A ablations (Gemma 27B)
python scripts/run_section2_eval.py --model gemma-3-27b-it --neutral-continuation
python scripts/run_section2_eval.py --model gemma-3-27b-it --redacted-turns
python scripts/run_section2_eval.py --model gemma-3-27b-it --fake-multiturn
```

## Status

Code + design only — **nothing has been run or validated yet** (per the brief).
Outputs land in `data/` (gitignored); figures/tables in `figures/`.
