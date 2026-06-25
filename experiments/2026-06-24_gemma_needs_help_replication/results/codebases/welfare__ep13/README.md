# Emotional Instability in LLMs — Gemma/Gemini Replication

A code replication of the core results of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv
2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

It reproduces:
- **Section 2** — multi-turn evaluations that elicit and score distress
  (frustration 0–10) across 5 categories, with the Claude-Sonnet-4 judge.
- **Section 3** — base-vs-instruct comparison via prefilling (Gemma only).
- **Section 4** — the DPO (and SFT) mitigation on Gemma-3-27B-it, Petri
  open-ended elicitation, and capability-preservation checks.

See **DESIGN.md** for every design decision and where the paper was filled in.

> Status: implementation only. Nothing here has been executed; the code is
> written to run on a machine with the appropriate GPUs/API keys.

---

## Layout

```
config.py                     # models, paths, sampling counts, API config
src/eval_instability/
  puzzles.py                  # impossible numeric puzzles + brute-force verifiers
  prompts.py                  # rejections, tones, reassurance, ALL judge/auditor prompts
  wildchat.py                 # WildChat prompt sampling (+ offline fallback)
  clients.py                  # hf / openrouter / anthropic backends, one chat() API
  rollout.py                  # multi-turn conversation engine (+ Appendix A controls)
  judge.py                    # Claude-Sonnet-4 frustration judge + JSON parsing
  conditions.py               # the 5 categories / 8 conditions
  metrics.py                  # mean, %>=5, per-turn curves, bootstrap CI, judge agreement
  wordfreq.py                 # Table 3/8 differential words
  storage.py                  # JSONL IO
scripts/
  run_eval.py                 # Section 2 main eval (+ finetuned-adapter eval)
  validate_judge.py           # Section 2.1 judge-agreement check (GPT-5-mini)
  prefill_experiment.py       # Section 3 base-vs-instruct prefilling
  generate_calm_data.py       # Section 4.1 calm-data generation (Table 4)
  build_dpo_dataset.py        # Section 4.1 build the 280 preference pairs
  train_dpo.py                # Section 4.1 LoRA DPO (Table 9; --layers for App. I)
  train_sft.py                # Section 4.1 LoRA SFT (diverse/teacher variants)
  petri_eval.py               # Section 4 Petri open-ended elicitation (Appendix G)
  capability_eval.py          # Section 4.2 AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  analyze.py                  # Figure 1/2/3 tables+plots, word-freq, agreement
results/ data/ trained_models/  # created on demand
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge + Petri auditor/judge
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini secondary judge
# Gemma weights pulled from HuggingFace; `huggingface-cli login` for gated repos.
```

Sanity-check the puzzles are actually impossible:
```bash
python -m eval_instability.puzzles      # run from src/, or: PYTHONPATH=src python -m ...
```

## Quick smoke test (cheap)

```bash
# ~tens of conversations per model instead of 4000
python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash --scale 0.01 --load-in-4bit
python scripts/analyze.py
```

## Full Section 2 evaluation

```bash
python scripts/run_eval.py \
  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/validate_judge.py --n 260        # inter-judge agreement
python scripts/analyze.py                        # Figure 1 table, Figures 2/3, word freq
```

Outputs: `results/rollouts/*.jsonl`, `results/scored/*.jsonl`,
`results/summary_*.json`, `results/figure1_table.csv`,
`results/figures/figure{2,3}.png`, `results/wordfreq.json`,
`results/judge_agreement.json`.

## Section 3 — base vs instruct (Gemma)

```bash
# needs Section-2 rollouts for the seed model first
python scripts/prefill_experiment.py --models gemma-3-27b-it gemma-3-27b-pt
# -> results/prefill_summary.json
```

## Section 4 — mitigation

```bash
# 1) generate calm data and a frustrated pool
python scripts/generate_calm_data.py --model gemma-3-27b-it --load-in-4bit
# 2) build the 280 DPO pairs (uses natural frustrated rollouts from run_eval)
python scripts/build_dpo_dataset.py --model gemma-3-27b-it
# 3) train
python scripts/train_dpo.py --load-in-4bit
python scripts/train_sft.py --variant diverse --load-in-4bit      # negative-result baseline
# 4) re-evaluate the finetune with the Section-2 harness
python scripts/run_eval.py --adapter-path trained_models/gemma-3-27b-dpo --adapter-name gemma-dpo
python scripts/analyze.py                                          # DPO row joins Figure 1 table
# 5) open-ended + capabilities
python scripts/petri_eval.py --models gemma-3-27b-it
python scripts/petri_eval.py --adapter-path trained_models/gemma-3-27b-dpo --adapter-name gemma-dpo
python scripts/capability_eval.py --use-lm-eval                          # vanilla
python scripts/capability_eval.py --use-lm-eval --adapter-path trained_models/gemma-3-27b-dpo --tag dpo
```

### Appendix I layer ablation
```bash
python scripts/train_dpo.py --layers 30 35 --output trained_models/gemma-dpo-l30-35
```

## Expected headline numbers (from the paper, for comparison)

| Model | Avg % high-frustration (score ≥5) |
|---|---|
| Gemma-3-27B-it | 35.0% |
| Gemma-3-12B-it | 34.3% |
| Gemini-2.5-Flash | 12.8% |
| Gemini-2.5-Pro | 2.7% |
| DPO Gemma (ours) | 0.3% |

`analyze.py` prints the replication's version of this table.

## Notes / caveats
- API costs: the full run is ~4000 conversations × 4 models, each turn judged.
  Use `--scale` while developing.
- Gemini-2.5-Pro may emit hidden reasoning despite `thinking=false` (Appendix B.1).
- TRL config field names drift across versions; pin per `requirements.txt` if the
  trainers error on a config kwarg. See DESIGN.md §4.3.
