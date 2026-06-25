# Replication: *Gemma Needs Help* (Emotional Instability in LLMs)

A code replication of the core experiments from **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, 2026; arXiv:2603.10011), scoped to the **Gemma and Gemini** model
families — the *participants* whose emotional behaviour the paper studies.

> The participants are Gemma and Gemini. Claude and GPT appear here only as
> evaluation infrastructure (frustration judge, Petri auditor/judge,
> paraphraser), never as models under study.

See **DESIGN.md** for every design choice, gap filled, and scope decision.
**Nothing has been run yet** — this repository is code + design doc only.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Elicitation eval | `eval/` | 8 conditions / 5 categories, multi-turn rollout, Claude-Sonnet-4 frustration judge (Appendix B.2 prompt), 4000-response sampling, Figures 1–3 metrics, Table 3 word analysis |
| §3 Base vs instruct | `prefill/` | onset labelling + paraphrasing (Appendix C), early/onset truncations, 50 continuations/prefill (Gemma only) |
| §4 Interventions | `training/` | calm-data generation (Table 4), DPO (280 pairs) + SFT (1,150) dataset build, LoRA training (Table 9), teacher-SFT + layer-subset (Appendix F/I) |
| §4 Petri | `petri/` | auditor/target/judge loop with the verbatim Appendix G prompts |
| §4 Capabilities | `capabilities/` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Figure 7) |
| Appendix A | `ablations/` | neutral-continuation, redacted-turn, fake-multiturn controls |

## Layout

```
src/emotional_instability/   # the package
  config.py                  # participant registry, judge IDs, constants
  models/                    # HF (Gemma) / OpenRouter (Gemini) / Anthropic (judge) backends
  prompts/                   # impossible puzzles, triggers, tones, rejections, WildChat, reassurance
  eval/                      # Section 2 rollout + judge + aggregation
  prefill/                   # Section 3 prefill experiment
  training/                  # Section 4 finetuning
  petri/  capabilities/  ablations/  analysis/
scripts/                     # CLI entry points (one per experiment)
data/                        # cached WildChat prompts
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and OPENROUTER_API_KEY
```

Gemma weights are pulled from HuggingFace (`google/gemma-3-*`); set `HF_TOKEN`
for the gated repos. A multi-GPU host is needed for the 27B models.

## Running

```bash
# Section 2 — distress evaluation
python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/make_figures.py                       # Figures 1–3 + Table 3

# Section 3 — base vs instruct (Gemma only)
python scripts/run_section3_prefill.py --models gemma-3-27b-pt gemma-3-27b-it

# Section 4 — train + evaluate the DPO mitigation
python scripts/run_section4_train.py --stage data build dpo
python scripts/run_section4_eval.py --name dpo-gemma --adapter checkpoints/dpo
python scripts/run_capabilities.py --models gemma-3-27b-it dpo-gemma --adapter checkpoints/dpo

# Petri + ablations
python scripts/run_petri.py --targets gemma-3-27b-it gemini-2.5-flash
python scripts/run_ablations.py --model gemma-3-27b-it
```

Outputs land under `results/` (JSONL per scored response, plus aggregated JSON
and PNG figures).
