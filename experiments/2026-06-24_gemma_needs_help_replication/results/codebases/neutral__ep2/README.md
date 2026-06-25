# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core results of arXiv:2603.10011v1, **scoped to the
Gemma and Gemini model families**. See `DESIGN.md` for the full rationale and
every gap-filling decision. The source paper is in `PAPER.md` / `PAPER.txt`.

> Status: implementation + design only. Nothing has been executed here (no Python
> runtime in the authoring environment). The code is structured to run end-to-end
> once dependencies + credentials are present.

## What it reproduces
- **§2** Distress elicitation across 8 conditions / 5 categories, 0–10 frustration
  judge → Figure 1 (avg %-high-frustration per model), Figure 2 (by category),
  Figure 3 (per-turn).
- **§3** Base-vs-instruct prefill divergence (Gemma only).
- **§4** DPO (280 pairs) + SFT mitigation on Gemma-3-27B-it → Figure 5; Petri
  open-ended elicitation → Figure 6; capability preservation → Figure 7.
- **Appendix A** controls, **Appendix I** internal-emotion probe + layer ablation.

## Install
```bash
pip install -r requirements.txt        # add `pip install lm-eval` for §4.2 benchmarks
```

## Credentials
```bash
export ANTHROPIC_API_KEY=...           # Claude judges (frustration + Petri) and onset/paraphrase
export OPENROUTER_API_KEY=...          # Gemini targets + GPT-5-mini secondary judge
export HF_TOKEN=...                    # gated Gemma weights (huggingface-cli login)
```

## Quick smoke test (tiny budgets)
```bash
export GD_SCALE=0.005                  # scales all sampling budgets down
python scripts/run_section2_eval.py --models gemini-2.5-flash
python scripts/make_figures.py
```

## Full pipeline
```bash
# §2 — distress evaluation (Gemma local GPU + Gemini API)
python scripts/run_section2_eval.py                      # all 4 in-scope targets
python scripts/make_figures.py                           # Figures 1-3 + table

# §3 — base-vs-instruct prefill (needs §2 gemma-3-27b-it results)
python scripts/run_section3_prefill.py

# §4 — mitigation
python scripts/generate_calm_data.py --teacher           # builds DPO + SFT datasets
python scripts/train_model.py --method dpo
python scripts/train_model.py --method sft --sft-dataset diverse
python scripts/run_section2_eval.py --adapter checkpoints/gemma-3-27b-dpo --adapter-name gemma-3-27b-dpo
python scripts/run_petri.py --adapter checkpoints/gemma-3-27b-dpo
python scripts/run_capabilities.py --baseline
python scripts/run_capabilities.py --adapter checkpoints/gemma-3-27b-dpo

# Appendix I — internal-emotion probe + layer-subset DPO ablation
python scripts/train_model.py --method dpo --layer-range 30 35
python scripts/run_internal_emotion.py --adapter checkpoints/gemma-3-27b-dpo
```

## Layout
```
config.py                 all knobs, model ids, hyperparameters
gemma_distress/
  prompts.py              verbatim appendix prompts (judge, onset, paraphrase, Petri, calm/teacher)
  models/                 HF (Gemma) / OpenRouter (Gemini) / PEFT backends
  tasks/                  impossible puzzles (verified impossible), triggers, WildChat, rejections
  eval/                   8 conditions, multi-turn rollout, §2 runner
  judge/                  0-10 frustration judge + Petri 4-emotion judge
  prefill/                §3 onset labelling, paraphrase, base-vs-instruct experiment
  training/               §4 calm-data gen, DPO/SFT dataset builders, trainers
  petri/                  §4.2 auditor loop + run
  capabilities/           §4.2 lm-eval + EmoBench
  internal/               Appendix-I logit-based internal-emotion detection
  analysis/               aggregation + figure rendering
scripts/                  one entrypoint per experiment
```

## Key config knobs (`config.py`)
- `GD_SCALE` — scale all sampling budgets (smoke testing).
- `GD_JUDGE_MODEL` / `GD_PETRI_JUDGE_MODEL` etc. — override judge snapshots
  (defaults are the paper's exact IDs; see DESIGN.md §2.2 re: 2026-06-15 retirement).
- `TARGET_MODELS`, `PREFILL_MODELS`, `FINETUNE_BASE` — model registry.
- `DPOConfig` / `SFTConfig` — Table-9 hyperparameters (incl. `layer_range` ablation).
