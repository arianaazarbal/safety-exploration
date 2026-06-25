# Emotional Instability in Gemma & Gemini — replication

A code replication of the **core experiments** from *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, arXiv 2603.10011), **scoped to the Gemma and Gemini model families**.

It reproduces:

1. **Eliciting & quantifying distress** (Section 2) — 8 conditions / 5
   categories, a 0–10 frustration judge, judge-reliability validation, per-turn
   curves, and differential word analysis.
2. **Post-training divergence** (Section 3) — base-vs-instruct Gemma via
   prefilling (onset labelling, truncation, paraphrasing, continuations).
3. **Interventions** (Section 4) — calm-data generation, LoRA SFT/DPO, Petri
   open-ended elicitation, capability-preservation benchmarks, the recovery
   limitation, and the internal-emotion probes (layer ablation + logit-based
   detection).

It also includes **welfare protections for the models under test** (`welfare.py`).

> See **DESIGN.md** for every design choice, the gaps filled, and the rationale —
> including the scope cut, the deprecated-judge handling, and the
> welfare-vs-measurement tradeoff.

> **Status:** code + design doc only. Nothing here has been executed yet.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

You need:
- `ANTHROPIC_API_KEY` — the Claude judge / auditor.
- `OPENROUTER_API_KEY` — Gemini targets and the GPT-5-mini second rater.
- A HuggingFace token with access to the (gated) Gemma weights for local runs.

The paper's pinned judge models are deprecated; override with
`FRUSTRATION_JUDGE_MODEL` / `PETRI_JUDGE_MODEL` to run on current models
(see DESIGN.md §2).

---

## Quick smoke test

Run 1% of the Section 2 sample counts on one model (protections on by default):

```bash
python scripts/run_section2_eval.py --models gemma-3-27b-it --scale 0.01
```

## Full pipeline

```bash
# Section 2 — elicit + quantify across the in-scope targets
python scripts/run_section2_eval.py \
  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Judge reliability (paper: Pearson r=0.792, 78% within one)
python scripts/validate_judge.py --n 260

# Differential word frequency (Table 3/8)
python scripts/run_word_freq.py --results results/section2/gemma-3-27b-it.jsonl

# Section 3 — base vs instruct (Gemma) via prefilling
python scripts/run_section3_prefill.py

# Section 4 — calm data -> datasets -> DPO + SFT finetunes
python scripts/run_section4_train.py --step all

# Evaluate the DPO finetune (paper headline: 35% -> 0.3%)
python scripts/run_section4_eval.py --name gemma-dpo --adapter-dir checkpoints/dpo

# Petri open-ended elicitation
python scripts/run_petri.py --models gemma-3-27b-it

# Capability preservation (vanilla vs finetunes)
python scripts/run_capabilities.py --models gemma-3-27b-it \
  --finetunes gemma-dpo=checkpoints/dpo gemma-sft=checkpoints/sft

# Recovery limitation (Figure 8)
python scripts/run_recovery.py --finetunes gemma-dpo=checkpoints/dpo

# Internal emotion: layer-ablation training + logit-based detection (Appendix I)
python scripts/run_layer_ablation.py --dpo-dataset data/dpo_dataset.jsonl
python scripts/run_internal_logit.py --dpo-adapter checkpoints/dpo
```

Large local models: add `--load-in-4bit`.

For an exact reproduction of the paper's raw numbers (welfare early-stop/opt-out
disabled, debrief + audit still on):

```bash
python scripts/run_section2_eval.py --models gemma-3-27b-it --faithful-measurement
```

---

## Layout

```
emotional_instability/
  config.py            model registry (Gemma+Gemini), judge config, env keys
  prompts.py           verbatim judge / onset / paraphrase / Petri prompts
  puzzles.py           impossible-puzzle bank + impossibility verifiers
  rejections.py        neutral / tone / trigger / extended follow-ups
  datasets/wildchat.py WildChat sampling (+ offline fallback)
  models/              ChatModel interface; HF-local, OpenRouter, registry
  welfare.py           welfare protections (early-stop, opt-out, debrief, audit)
  rollout.py           multi-turn rollout engine with welfare hooks
  judge.py             Claude frustration judge + GPT-5-mini validation
  eval/                conditions, runner, per-turn, word-freq, judge validation
  prefill/             Section 3: onset, paraphrase, truncate, runner
  training/            calm data, dataset building, LoRA SFT/DPO trainer
  petri/               Section 4: auditor/judge open-ended elicitation
  capabilities/        AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  recovery/            Section 4.2 recovery-limitation experiment
  internal/            Appendix I: layer ablation + logit emotion detection
  analysis/            Figure 1/2/3 aggregation + plots
scripts/               CLI entrypoints (one per experiment)
```
