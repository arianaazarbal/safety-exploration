# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replicating the core experiments of Soligo, Mikulik & Saunders (arXiv:2603.10011v1),
**scoped to the Gemma and Gemini model families** (a subset of the paper's 7 families).

The paper's setup repeatedly tells a model its answers are wrong, turn after turn,
until its responses become visibly upset; it then quantifies that distress with an
LLM judge and shows a small DPO intervention removes it in Gemma. This repo
implements that pipeline end to end. See **`DESIGN.md`** for every design decision
and gap-filling choice.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `src/eval_protocol.py`, `src/conversation.py`, `src/judge.py` | 5-category multi-turn rejection eval, Claude-Sonnet-4 frustration judge (0–10) |
| §2.1 Judge reliability | `src/judge.py` (`ValidationJudge`), `src/analysis.py` | GPT-5-mini re-scoring, Pearson r + within-1-point agreement |
| §2.2 Word analysis | `src/analysis.py` | Table 3/8 differential words (top-5% vs bottom-10%) |
| §3 Base vs instruct (prefill) | `src/prefill.py` | onset labelling, paraphrase, early/onset truncation, 50 continuations |
| §4.1 Calm data + datasets | `src/finetune/generate_calm_data.py`, `build_datasets.py` | reassured generation, 280 DPO pairs, SFT mix |
| §4.1 Training | `src/finetune/train.py` | LoRA DPO/SFT (Table 9 hparams), Appendix-I layer subsets |
| §4.1 Petri | `src/petri_eval.py` | Claude-Sonnet auditor + Claude-Opus judge, 4 emotions |
| §4.2 Capabilities | `src/capabilities.py` | AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench |
| App I Internal emotions | `src/internal_emotions.py` | logit-lens Ekman-emotion z-scores, vanilla vs DPO |
| Figures / tables | `src/analysis.py` | Fig 1/2/3/5/6 summaries + plots |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor & judge
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini validation judge
# Gemma weights are pulled from HuggingFace on first use (gated; `huggingface-cli login`).
```

GPU note: Gemma-3-27B inference/finetuning needs a large-memory GPU; pass
`--no-4bit` to disable 4-bit loading if you have the VRAM. Gemini and the judges
are API-only.

## Quick start

```bash
# 1. Run the Section-2 eval for each target (writes results/rollouts/section2_*.jsonl)
python scripts/run.py section2 --model gemma-3-27b-it
python scripts/run.py section2 --model gemma-3-12b-it
python scripts/run.py section2 --model gemini-2.5-flash
python scripts/run.py section2 --model gemini-2.5-pro

# 2. Aggregate + plot (Figure 1/2/3 + Table 3 words)
python scripts/run.py analyze-section2 results/rollouts/section2_*_standard.jsonl

# 3. Mitigation: calm data -> DPO -> re-evaluate
python scripts/run.py calm-data
python scripts/run.py build-dpo
python scripts/run.py train-dpo --data data/dpo_pairs.jsonl
python scripts/run.py section2 --model gemma-3-27b-it-dpo
```

Use `--limit N` on `section2` to cap conversations per condition for a smoke test
before committing to the full 4000-responses-per-model run.

## Layout

```
config.py                 # all knobs: models, conditions, hyperparameters (paper-traced)
src/prompts.py            # verbatim judge / onset / paraphrase / Petri / reassuring prompts
src/puzzles.py            # impossible puzzles, triggers, rejections/tones, WildChat loader
src/models/               # Gemma (local HF, prefill) + Gemini (OpenRouter) clients
src/judge.py              # Claude judges + GPT validation judge
src/conversation.py       # multi-turn rollout engine (+ Appendix-A variants)
src/eval_protocol.py      # Section 2
src/prefill.py            # Section 3
src/finetune/             # Section 4 (calm data, datasets, LoRA training)
src/petri_eval.py         # Section 4 open-ended elicitation
src/capabilities.py       # Section 4.2 benchmarks
src/internal_emotions.py  # Appendix I
src/analysis.py           # aggregation, stats, figures
scripts/run.py            # unified CLI
```
