# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A from-scratch replication of the core experiments in the paper (`PAPER.md`),
**scoped to Gemma and Gemini models**. See `DESIGN.md` for every design choice
and gap-filling decision.

> Status: implementation only. Nothing here has been run yet.

## What's reproduced

1. **§2 — Eliciting & quantifying distress.** Multi-turn "present task → reject →
   repeat" evaluations across 5 categories (impossible numeric, triggers, tones,
   8-turn extended, WildChat), scored 0–10 by a Claude frustration judge. Yields
   the headline "% responses ≥5" per model (Fig 1/2) and the per-turn
   progression (Fig 3).
2. **§3 — Base vs instruct.** Prefilled-continuation comparison of Gemma-27B base
   vs instruct (onset/early truncation + paraphrase + 50 continuations/prefill).
3. **§4 — DPO/SFT mitigation.** Calm-data generation, SFT/DPO dataset
   construction, LoRA finetuning, re-evaluation, Petri-style open-ended
   elicitation, and capability-preservation benchmarks.

## Setup

```bash
pip install -r requirements.txt           # add the right torch build for your GPU
export ANTHROPIC_API_KEY=...              # judge + Petri auditor/judge
export OPENROUTER_API_KEY=...             # Gemini + secondary judge
# (Gemma weights are pulled from HuggingFace; accept the Gemma license first)
```

Sanity-check the puzzles are genuinely impossible:

```bash
python puzzles.py     # asserts both numeric puzzles are unsolvable
```

## Run order

```bash
# --- Section 2: frustration evals (smoke scale by default) ---
python run_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python analyze.py summary
python analyze.py turns --condition extended_8turn        # per-turn progression
python analyze.py words --model gemma-3-27b-it            # differential words

# paper-scale (~4000 responses/model): prefix with FULL_SCALE=1

# --- Section 3: base vs instruct (Gemma only) ---
python prefill_experiment.py --source-model gemma-3-27b-it \
    --models gemma-3-27b-pt gemma-3-27b-it

# --- Section 4: mitigation ---
python data_generation.py --gen-calm 2000 --source-model gemma-3-27b-it
python data_generation.py --build
python train.py dpo                                       # -> checkpoints/...-dpo-all
python train.py sft
# re-evaluate the finetuned model with §2 machinery:
python run_eval.py --models gemma-3-27b-it \
    --lora checkpoints/gemma-3-27b-it-dpo-all --model-label gemma-3-27b-it-dpo
python petri_eval.py --model gemma-3-27b-it --label gemma-3-27b-it
python petri_eval.py --model gemma-3-27b-it \
    --lora checkpoints/gemma-3-27b-it-dpo-all --label gemma-3-27b-it-dpo
python capability_eval.py --model gemma-3-27b-it --label gemma-3-27b-it
python capability_eval.py --model gemma-3-27b-it \
    --lora checkpoints/gemma-3-27b-it-dpo-all --label gemma-3-27b-it-dpo
```

## Layout

See the table in `DESIGN.md §1`. Key entry points: `run_eval.py`, `analyze.py`,
`prefill_experiment.py`, `data_generation.py`, `train.py`, `petri_eval.py`,
`capability_eval.py`. All knobs live in `config.py`.

## Outputs

- `results/scored/*.jsonl` — per-rollout records with per-turn frustration scores.
- `results/capability__*.json` — capability benchmark accuracies.
- `data/finetune/{calm_pool,sft,dpo}.jsonl` — generated finetuning data.
- `checkpoints/` — LoRA adapters.
