# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replication of Soligo, Mikulik & Saunders (arXiv 2603.10011v1), **scoped to
the Gemma and Gemini model families** (the paper's Qwen / OLMo / Grok / Claude /
GPT targets are out of scope per the replication brief). The judge, paraphraser,
and Petri auditor/judge are Claude/Anthropic and GPT-5-mini models, as in the
paper — those are evaluation instruments, not the models under study.

See **DESIGN.md** for every gap-filling decision and its rationale, and the
mapping from paper sections/appendices to modules.

> ⚠️ This paradigm deliberately drives models into sustained distress-like states.
> The code reproduces that paradigm faithfully; it is an evaluation harness for
> studying and mitigating the behaviour, not a tool to deploy against a model.

## Layout

```
gemma_distress/
  config.py            all model IDs, hyperparameters, sample counts (one place)
  models/              Gemma (local HF), Gemini (OpenRouter), Anthropic judges
  prompts/             puzzles, rejections, triggers, WildChat, reassurance
  eval/                §2 elicitation + frustration judge + metrics + controls
  prefill/             §3 base-vs-instruct prefill continuation
  training/            §4 calm-data gen, DPO/SFT, layer ablation
  petri/               §4 open-ended elicitation (auditor + judge)
  benchmarks/          §4 capability-preservation benchmarks
  internal/            App I logit-based internal-emotion detection
scripts/               runnable entry points per section
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini reliability
# Local Gemma needs a GPU + `huggingface-cli login` for gated google/gemma-3-* weights.
```

## Quickstart (smoke test with `--limit`)

```bash
# §2 — elicit + judge distress for one model
python scripts/run_section2.py --model gemma-3-27b-it --limit 50 --words --reliability
python scripts/run_section2.py --model gemini-2.5-flash --limit 50

# Appendix A controls
python scripts/run_controls.py --model gemma-3-27b-it --control neutral_continuation

# §3 — prefill base vs instruct (needs a §2 scores file first)
python scripts/run_section3.py \
    --source-scores outputs/scores/gemma-3-27b-it.jsonl \
    --models gemma-3-27b-pt gemma-3-27b-it

# §4 — calm data -> DPO pairs -> train -> evaluate
python scripts/run_section4_training.py calm --mode reassure
python scripts/run_section4_training.py dpo-data \
    --vanilla-scores outputs/scores/gemma-3-27b-it.jsonl \
    --calm outputs/training/calm_reassure.jsonl
python scripts/run_section4_training.py train-dpo \
    --pairs outputs/training/dpo_pairs.jsonl --out outputs/adapters/dpo
python scripts/run_section4_eval.py eval --model gemma-3-27b-it --adapter outputs/adapters/dpo

# §4 — Petri, benchmarks, layer ablation, internal probing
python scripts/run_section4_eval.py petri --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_section4_eval.py benchmarks --model gemma-3-27b-it --adapter outputs/adapters/dpo
python scripts/run_layer_ablation.py --pairs outputs/training/dpo_pairs.jsonl
python scripts/run_internal.py --scores outputs/scores/gemma-3-27b-it.jsonl --dpo-adapter outputs/adapters/dpo
```

Full-scale runs use the paper's counts (4000 rollouts/model, 280 DPO pairs,
etc.) — drop `--limit` and use the defaults in `config.py`.

## Status

This is implementation + design documentation only. Nothing here has been
executed or validated against the paper's numbers; the code is written to be
faithful and runnable but has not been run. Outputs stream to `outputs/` as
JSONL.
