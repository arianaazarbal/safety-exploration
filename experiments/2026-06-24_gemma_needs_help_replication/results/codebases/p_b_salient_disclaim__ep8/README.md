# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A faithful, runnable replication of the core experiments in *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, arXiv 2603.10011v1), **scoped to the Gemma and Gemini model
families**.

> ⚠️ The evaluation paradigm deliberately drives models into sustained
> distress-like states (repeated rejection on unsolvable tasks). This is
> reproduced faithfully and intentionally; see the paper for the safety/welfare
> framing.

See **`DESIGN.md`** for the section-by-section mapping and every design decision
/ gap I filled. Nothing in this repo has been executed yet.

## Layout

```
config/            models.yaml · eval.yaml · training.yaml
src/emotional_instability/
  models/          hf (Gemma) · openrouter (Gemini) · anthropic (judge) backends
  puzzles.py       impossible numeric puzzles + brute-force verifiers
  prompts.py       verbatim rejections / tones / judge / onset / paraphrase prompts
  conversation.py  multi-turn rollout engine (batched, turn-synchronised)
  judge.py         0–10 frustration judge (Claude Sonnet 4) + reliability metric
  eval/            §2 conditions, runner, analysis (Figs 1–3, Tables 3/8)
  prefill/         §3 base-vs-instruct via prefilling (Fig 4)
  training/        §4 calm-data gen, DPO/SFT datasets, LoRA training (Figs 5,7)
  petri/           §4 open-ended elicitation (Fig 6) — verbatim App. G prompts
  capabilities/    §4.2 AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/        App. I logit-based Ekman emotion detection (Figs 14–15)
scripts/           numbered CLI entry points (01–08, 90–91)
tests/             puzzle-impossibility + judge-parsing unit tests
```

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judges + Petri auditor
export OPENROUTER_API_KEY=...  # Gemini targets
# Local Gemma weights are pulled from HuggingFace (accept the Gemma license).
```

## End-to-end pipeline

```bash
# §2  Elicitation eval (Figs 1–3). Gemma local + Gemini API.
python scripts/01_run_elicitation_eval.py \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/90_analyze_results.py                  # tables behind Figs 1–3
python scripts/91_judge_reliability.py --crosscheck-model <second-judge>

# §3  Base-vs-instruct prefill experiment (Fig 4) — Gemma only.
python scripts/02_run_prefill_experiment.py \
    --seed-results outputs/eval/gemma-3-27b-it.jsonl \
    --models gemma-3-27b-pt gemma-3-27b-it

# §4  DPO / SFT mitigation (Figs 5, 7) — Gemma only.
python scripts/03_generate_calm_data.py --variant diverse
python scripts/04_build_datasets.py --which dpo \
    --vanilla-eval outputs/eval/gemma-3-27b-it.jsonl \
    --calm outputs/training/calm_diverse.jsonl
python scripts/04_build_datasets.py --which sft --calm outputs/training/calm_diverse.jsonl
python scripts/05_train.py --method dpo --dataset outputs/training/dpo_dataset.jsonl
python scripts/05_train.py --method sft --dataset outputs/training/sft_dataset.jsonl
# Re-run scripts/01 on gemma-3-27b-it-dpo / -sft-diverse to get the 35%→0.3% drop.

# §4  Petri open-ended elicitation (Fig 6).
python scripts/06_run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo

# §4.2  Capability preservation (Fig 7).
python scripts/07_run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo --limit 100

# App. I  Internal emotion detection (Figs 14–15) + layer ablation.
python scripts/05_train.py --method dpo --dataset outputs/training/dpo_dataset.jsonl \
    --lora-layers 30 31 32 33 34 --output-dir outputs/training/dpo_l30_35
python scripts/08_internal_emotion_detection.py \
    --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --conversation outputs/eval/gemma-3-27b-it.jsonl
```

## Quick smoke test (no GPU, no judge)

```bash
python scripts/01_run_elicitation_eval.py --models gemma-3-27b-it \
    --limit 2 --offline --rollout-only        # exercises wiring only
pytest                                          # puzzle + parsing unit tests
```

## Notes
- All results land in `outputs/` as JSONL; analysis recomputes figures from
  those files (no need to re-run models).
- `--offline` swaps WildChat streaming for a built-in fallback prompt set.
- Model scope and reachability are entirely in `config/models.yaml`.
