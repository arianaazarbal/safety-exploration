# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replication of the core experiments in `PAPER.md` (arXiv 2603.10011v1),
**scoped to the Gemma and Gemini model families as the participants** (subjects
under study). Judge / auditor / paraphraser models are infrastructure and are
kept exactly as the paper specifies (Claude Sonnet 4, Claude Opus 4, GPT-5-mini).

See `DESIGN.md` for every design decision and the gaps that were filled.

> Status: implementation only. Nothing here has been run; no results are claimed.

## What is implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Elicitation | `src/eval/`, `src/prompts/` | Impossible-puzzle generation, 8 conditions / 5 categories, multi-turn rollout, Claude 0–10 frustration judge, 4000-response sample plan |
| §2 Analysis | `src/analysis/` | Figures 1–3, differential words (Table 3/8), judge agreement (Pearson r) |
| §3 Prefilling | `src/prefill/` | Onset labelling, paraphrase, early/onset truncation, base-vs-instruct continuations |
| §4.1 Training | `src/training/` | Calm-data generation, SFT + DPO dataset construction, LoRA SFT/DPO (incl. layer ablations) |
| §4.2 Petri | `src/petri/` | Auditor (Claude Sonnet) ↔ target loop, Opus judge on 4 emotions |
| §4.2 Capabilities | `src/capabilities/` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| §4.2 Recovery | `src/prefill/` | Continuations from score≥7 states truncated 200 tokens before end |
| §4.2 / App I Probing | `src/probing/` | Logit-based Ekman-emotion detection, vanilla vs DPO |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in API keys + HF_TOKEN
```

Local Gemma inference/training/probing needs a GPU with enough memory for
`gemma-3-27b-it` (bf16, or set `load_in_4bit=True`). Gemini participants are
reached via OpenRouter.

## Running

`config.yaml` holds model ids and the sample plan. Use the pipeline driver:

```bash
# Smoke test (tiny rollout cap) to validate wiring end-to-end:
python -m scripts.pipeline --stage section2 --limit 8

# Full Section 2 + analysis:
python -m scripts.pipeline --stage section2
python -m scripts.pipeline --stage analysis

# Section 4 training + evaluation:
python -m scripts.pipeline --stage calm
python -m scripts.pipeline --stage datasets
python -m scripts.pipeline --stage train
python -m scripts.pipeline --stage eval_ft
```

Each module is also a standalone entrypoint, e.g.:

```bash
python -m src.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
python -m src.training.train_dpo --lora-layers 30-35   # Appendix I ablation
```

Outputs are written under `outputs/` (per-model JSONL rollouts, CSV summaries,
figures) and `data/` (cached WildChat prompts).
