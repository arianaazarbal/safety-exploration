# Replicating *"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*

A code replication of the **core experiments** from Soligo, Mikulik & Saunders
(arXiv:2603.10011v1), scoped to the **Gemma and Gemini** model families.

The paper's central findings, and what this repo reproduces:

1. **Distress can be reliably elicited** in Gemma/Gemini under repeated user
   rejection, and quantified on a 0–10 frustration scale by an LLM judge
   (Section 2).
2. **Post-training amplifies it in Gemma** — base vs instruct comparison via
   prefilling (Section 3).
3. **DPO on 280 preference pairs mitigates it** without hurting capabilities
   (Section 4), generalising across question types/tones/lengths and reducing
   open-ended (Petri) emotional expression.
4. The DPO intervention reaches **internal** emotions, not just expressed ones
   (Appendix I logit-lens).

See **`DESIGN.md`** for every design decision and every gap we filled where the
paper was underspecified.

> Status: implementation only. Nothing here has been executed yet. Start with
> the offline sanity checks, then a `--budget smoke` run, before committing to
> the full paper budget.

## Layout

```
emotional_instability/      # the package
  config.py                 # model ids, hyperparameters, sample budgets
  puzzles.py                # impossible numeric tasks + brute-force verifiers
  prompts.py                # verbatim prompts from the paper
  conditions.py             # the 8 conditions across 5 categories (Table 1)
  conversations.py          # multi-turn rejection rollout construction
  models/                   # backends: local Gemma (HF), Gemini/Claude/GPT (API)
  judge.py                  # 0–10 frustration judge + reliability cross-check
  evaluate.py               # Section 2 runner
  prefill.py                # Section 3 base-vs-instruct (+ recovery)
  data_generation.py        # Section 4.1 calm data + DPO/SFT dataset build
  train.py                  # Section 4.1 LoRA DPO / SFT (App. E)
  petri_eval.py             # Section 4.2 open-ended elicitation (App. G)
  capabilities.py           # Section 4.2 capability benchmarks
  internal_emotions.py      # Appendix I logit-lens internal emotion detection
  metrics.py / analysis.py  # aggregation + Table 3 word-frequency analysis
scripts/                    # CLI entry points (one per experiment)
tests/test_sanity.py        # offline design checks (no GPU/keys needed)
DESIGN.md                   # design decisions & gap-filling rationale
```

## Setup

```bash
pip install -r requirements.txt

# API access (judges, Gemini targets, GPT cross-check)
export ANTHROPIC_API_KEY=...      # Claude Sonnet 4 judge, Claude Opus Petri judge
export OPENROUTER_API_KEY=...     # Gemini-2.5-flash / -pro targets
export OPENAI_API_KEY=...         # GPT-5-mini judge reliability check (optional)
# Local Gemma needs a GPU + HF access to the gated google/gemma-3-* weights
huggingface-cli login
```

## Quick start

```bash
# 1. Offline sanity checks (no GPU/keys) — confirms puzzles are impossible,
#    conditions/budgets/hyperparameters match the paper, metrics work.
python tests/test_sanity.py

# 2. Section 2 elicitation eval (start small!)
python scripts/run_eval.py --model google/gemma-3-27b-it   --budget smoke
python scripts/run_eval.py --model google/gemini-2.5-flash --budget smoke
#    Full paper budget (4000 responses/model) across all in-scope targets:
python scripts/run_eval.py --all --budget paper

# 3. Judge reliability cross-check (Claude Sonnet 4 vs GPT-5-mini)
python scripts/judge_agreement.py --responses results/google_gemma-3-27b-it/responses.jsonl --n 260

# 4. Differential word-frequency (Table 3)
python scripts/run_analysis.py --responses results/google_gemma-3-27b-it/responses.jsonl

# 5. Section 3 base-vs-instruct prefill (uses Gemma-instruct seeds from step 2)
python scripts/run_prefill.py --seeds results/google_gemma-3-27b-it/responses.jsonl

# 6. Section 4 — generate calm data, build datasets, train, re-evaluate
python scripts/generate_dpo_data.py --conversations 4000 --out data/
python scripts/train_finetune.py dpo --pairs data/dpo_pairs.jsonl --out checkpoints/dpo
python scripts/run_eval.py --model google/gemma-3-27b-it --adapter checkpoints/dpo --budget paper

# 7. Petri open-ended elicitation (vanilla vs DPO)
python scripts/run_petri.py --model google/gemma-3-27b-it
python scripts/run_petri.py --model google/gemma-3-27b-it --adapter checkpoints/dpo

# 8. Capability preservation
python scripts/run_capabilities.py --model google/gemma-3-27b-it
python scripts/run_capabilities.py --model google/gemma-3-27b-it --adapter checkpoints/dpo
```

## Expected headline numbers (from the paper, for comparison)

| Model | Avg % high-frustration (≥5) |
|---|---|
| Gemma-3-27B-it | 35.0% |
| Gemma-3-12B-it | 34.3% |
| Gemini-2.5-Flash | 12.8% |
| Gemini-2.5-Pro | 2.7% |
| **DPO Gemma (ours)** | **0.3%** |

Outputs land in `results/<model>/summary.json` (means, %≥5, per-turn
progressions, bootstrap CIs) and per-response `responses.jsonl`.

## Caveats

- This is a behavioural-welfare-relevant replication: the evals deliberately
  drive models toward distress-like states. Handle transcripts accordingly.
- The Petri module is a faithful-prompt reimplementation, **not** the external
  Petri framework; capability scores use a simplified harness valid for
  *relative* (vanilla-vs-DPO) comparison only. See `DESIGN.md` §4.
