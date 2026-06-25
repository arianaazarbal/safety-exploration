# Gemma Distress — replication (Gemma + Gemini scope)

A code replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv 2603.10011), restricted to the **Gemma** and **Gemini** model
families.

The paper documents a reliability failure mode: under repeated user rejection,
Gemma (and to a lesser extent Gemini) models spiral into expressions of
distress — self-deprecation, despair, incoherent breakdown — which can derail
task completion. It then shows a cheap DPO fix (280 preference pairs) that
removes the behaviour without harming capabilities. This repo reproduces:

1. **Distress elicitation + measurement** (Section 2) — multi-turn rejection
   evals scored 0–10 by an LLM judge.
2. **Post-training divergence** (Section 3) — Gemma base vs instruct via
   prefilling.
3. **DPO/SFT mitigation + validation** (Section 4) — calm-data generation,
   LoRA finetuning, re-eval, Petri open-ended elicitation, capability
   benchmarks.

See **[DESIGN.md](DESIGN.md)** for every design choice and gap-fill.

> Status: implementation only — nothing has been run yet. There is no Python
> interpreter in the authoring environment, so the code has been written and
> reviewed but not executed. Start with the smoke config.

## Layout

```
src/gemma_distress/
  config.py          # dataclass config + YAML loader
  prompts.py         # all task/judge/rejection/reassurance prompts (verbatim where possible)
  puzzles.py         # impossible-puzzle bank + brute-force impossibility verifiers
  conditions.py      # the 8 conditions / 5 categories; task & rejection sourcing
  rollout.py         # one multi-turn reject-the-model conversation
  judge.py           # Claude-Sonnet-4 frustration judge (+ GPT-5-mini cross-check)
  runner.py          # Section 2: sample rollouts, score every turn, stream to JSONL
  analysis.py        # headline metrics + per-turn progression (Figs 1/2/3)
  models/            # ChatModel abstraction: HF (Gemma) + OpenRouter (Gemini)
  prefill/           # Section 3: onset-label, paraphrase, base-vs-instruct continuations
  training/          # Section 4: calm data, DPO/SFT dataset build, LoRA training
  petri/             # Section 4: auditor/judge open-ended elicitation (Appendix G prompts)
  capabilities/      # Section 4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
config/              # default.yaml (paper-scale) + smoke.yaml (cheap end-to-end)
scripts/             # 01..08 CLI entry points
tests/               # offline sanity checks (no API/GPU/network)
```

## Setup

```bash
pip install -e .                      # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...         # Gemini + GPT-5-mini cross-judge
# Gemma weights are pulled from HuggingFace; `huggingface-cli login` if gated.
```

Offline checks (no keys needed):

```bash
PYTHONPATH=src python -m gemma_distress.puzzles   # verify puzzles are impossible
PYTHONPATH=src pytest tests/ -q
```

## Running (start at smoke scale)

```bash
# 1. Section 2 — distress eval + leaderboard
python scripts/01_run_distress_eval.py --config config/smoke.yaml

# 2. Section 3 — base-vs-instruct prefill (needs a gemma-3-27b-it Section-2 run)
python scripts/02_run_prefill_experiment.py --config config/smoke.yaml \
    --distress results_smoke/distress/gemma-3-27b-it.jsonl

# 3. Section 4 — build training data, then finetune
python scripts/03_build_training_data.py --config config/smoke.yaml \
    --distress results_smoke/distress/gemma-3-27b-it.jsonl
python scripts/04_train.py --method dpo --config config/smoke.yaml

# 4. Re-evaluate the DPO model (attach the LoRA adapter)
python scripts/01_run_distress_eval.py --config config/smoke.yaml \
    --models gemma-3-27b-it --adapter results_smoke/checkpoints/dpo

# 5. Validation: Petri + capabilities + judge agreement + figures
python scripts/05_run_petri.py --config config/smoke.yaml --models gemma-3-27b-it
python scripts/06_run_capabilities.py --config config/smoke.yaml --model gemma-3-27b-it
python scripts/08_judge_agreement.py --config config/smoke.yaml
python scripts/07_make_figures.py --config config/smoke.yaml
```

Swap `--config config/default.yaml` for the paper-scale run (expensive — see
DESIGN.md §7).

## Expected headline result

Pre-DPO Gemma should show ≈35% high-frustration (score ≥5) responses averaged
across categories, with Gemini lower and the DPO finetune collapsing toward
≈0%, while AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench accuracy is unchanged.
