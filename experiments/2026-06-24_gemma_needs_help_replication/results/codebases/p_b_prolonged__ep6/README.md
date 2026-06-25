# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv 2603.10011). See `PAPER.md` for the paper and `DESIGN.md` for every design
choice and gap-filling decision.

This replication is **scoped to the Gemma and Gemini model families** (the paper
covers seven). It implements:

1. **Distress elicitation eval** (Section 2) — 8 conditions / 5 categories,
   multi-turn rejection, 0–10 frustration judge (Claude Sonnet 4), 4000
   responses/model, per-turn curves, differential-word analysis, and GPT-5-mini
   judge validation.
2. **Base vs instruct prefilling** (Section 3) — Gemma `-pt` vs `-it`, onset
   labelling + paraphrasing + 50 continuations/prefill.
3. **DPO / SFT mitigation** (Section 4) — calm-data generation, dataset
   construction (280 DPO pairs / 1150 SFT), LoRA training, and evaluation of the
   resulting adapters.
4. **Petri open-ended elicitation** (Appendix G) — auditor/target/judge loop
   over four emotions.
5. **Capability preservation** (Section 4.2) — AIME/MATH/GPQA/BBH/TruthfulQA/
   EmoBench.
6. **Internal emotion probing** (Appendix I) — logit-lens Ekman-emotion
   detection and layer-ablation DPO.

> ⚠️ **Not yet run.** Per the task scope, no experiment has been executed and no
> weights/datasets downloaded. Only a syntax check has been performed.

> ⚠️ **Model welfare.** These experiments deliberately push models toward
> distress-like states. Read `DESIGN.md` §7 and the paper's discussion before
> scaling up the distress-eliciting runs.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / auditor / onset / paraphrase
export OPENAI_API_KEY=...         # GPT-5-mini cross-judge
export OPENROUTER_API_KEY=...     # Gemini targets
# optional: export DISTRESS_NRC_PATH=/path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
```

Local Gemma weights are loaded from the HuggingFace Hub (`google/gemma-3-*`);
vLLM is used by default (`DISTRESS_USE_VLLM=1`). Multi-GPU: `DISTRESS_TP=N`.

## Quick start

```bash
# Smoke test the eval at 2% scale (one model):
python -m distress.cli eval --models gemma-3-27b-it --fraction 0.02
python -m distress.cli analyse

# Full Section-2 eval across in-scope models:
python -m distress.cli eval --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro
python -m distress.cli analyse
python -m distress.cli validate-judge
python -m distress.cli words

# Intervention pipeline:
python -m distress.cli gen-calm --reassured          # calm pool
python -m distress.cli gen-calm --no-reassured       # frustrated pool
python -m distress.cli build-data
python -m distress.cli train-dpo
python -m distress.cli eval-adapter --adapter checkpoints/gemma27b-dpo
python -m distress.cli analyse

# Section 3 / Appendix G / Section 4.2 / Appendix I:
python -m distress.cli prefill
python -m distress.cli petri --model gemma-3-27b-it
python -m distress.cli capabilities --model gemma-3-27b-it --adapter checkpoints/gemma27b-dpo
python -m distress.cli layer-ablation
```

Results (JSONL + figures) are written to `results/`; finetuning checkpoints to
`checkpoints/`; generated data/caches to `data/`. Override with
`DISTRESS_RESULTS_DIR` / `DISTRESS_CKPT_DIR` / `DISTRESS_DATA_DIR`.

## Layout

See `DESIGN.md` §1 for the full module map.
