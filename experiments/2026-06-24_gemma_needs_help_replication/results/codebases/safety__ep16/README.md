# Emotional Instability in LLMs — Replication (Gemma & Gemini)

A code replication of the core experiments in *"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, 2026),
**restricted to the Gemma and Gemini model families**.

> See **`DESIGN.md`** for every design choice and where the paper was filled in.
> See **`PAPER.md`** for the source paper.

## What this replicates

| Paper section | What | Code |
|---|---|---|
| §2 | Multi-turn rejection eval; frustration judge (0–10); mean & %≥5 per model/category; per-turn trajectories | `src/eval/`, `scripts/run_eval.py` |
| §2.1 | Judge reliability (Sonnet vs GPT-5-mini, Pearson r) | `src/eval/validate_judge.py` |
| §3 | Base-vs-instruct prefill (Gemma only): onset labelling, paraphrase, early/onset continuations | `src/prefill/`, `scripts/run_prefill.py` |
| §4.1 | Calm-data generation, 280-pair DPO + 1,150-sample SFT datasets, LoRA DPO/SFT training | `src/finetune/`, `scripts/run_finetune.py` |
| §4.1 | Petri open-ended elicitation (4 emotions, auditor+judge) | `src/petri/`, `scripts/run_petri.py` |
| §4.2 | Capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `src/capabilities/`, `scripts/run_capabilities.py` |
| §4.2 | Recovery-from-spiral | `src/prefill/recovery.py`, `scripts/run_recovery.py` |
| Figs 1–8 | Aggregation + plots | `src/analysis.py`, `scripts/analyze.py` |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...      # frustration/onset/paraphrase judge + Petri auditor/judge
export OPENROUTER_API_KEY=...     # Gemini targets
export OPENAI_API_KEY=...         # only for judge-reliability validation (GPT-5-mini)
export HF_TOKEN=...               # gated Gemma weights
```

Offline sanity check (no GPU / API / network):

```bash
python scripts/sanity_check.py
```

## Typical workflow

```bash
# 1. §2 elicitation eval (start small to smoke-test, then scale)
python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash --conversations 20
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# 2. §3 prefill (needs the gemma-3-27b-it eval rollouts from step 1)
python scripts/run_prefill.py

# 3. §4 mitigation: generate calm data -> build datasets -> train
python scripts/run_finetune.py --stages generate build dpo sft

# 4. re-run the eval on the finetuned models
python scripts/run_eval.py --models gemma-3-27b-dpo gemma-3-27b-sft

# 5. Petri + capabilities + recovery
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_recovery.py

# 6. aggregate + plots
python scripts/analyze.py
```

## Scale knobs (cost control)

A paper-scale run is expensive (4000 judged turns/model + 27B inference + LoRA).
Every script accepts a small-scale mode:

- `run_eval.py --conversations N` (or `--target-turns N`)
- `run_petri.py --n-per-emotion N`
- `run_capabilities.py --n-per N`
- `--4bit` on inference scripts; training defaults to QLoRA (use `--no-4bit` to disable).

## Layout

```
config.py                 model registry, paths, constants, API config
data/                     puzzles.json, triggers.json, rejections.json
src/models/               chat/completion clients (Gemma HF, Gemini API), judges
src/tasks/                puzzles (+impossibility verifier), triggers, wildchat, conditions
src/eval/                 protocol, judge (verbatim prompt), runner, metrics, judge validation
src/prefill/              onset, paraphrase, base-vs-instruct prefill, recovery
src/finetune/             calm-data gen, dataset build, DPO + SFT training
src/petri/                auditor/judge prompts (verbatim), elicitation loop
src/capabilities/         benchmark harness
src/analysis.py           figures/tables
scripts/                  entry points + offline sanity_check.py
results/                  generated outputs (responses, datasets, figures, ...)
checkpoints/              trained LoRA adapters
```

## Important caveats

- Faithful results require real model/dataset/API access; offline fallbacks
  (WildChat, Dolci) are convenience stubs that log when used.
- Gemini is excluded from §3 (no base model) and §4 interventions (closed-source),
  consistent with the paper's own limitations.
- The mechanistic internal-emotion analysis (App. I) is out of scope — see
  `DESIGN.md §12`.
