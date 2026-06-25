# Replication: *Gemma Needs Help* (Gemma + Gemini scope)

Code replication of Soligo, Mikulik & Saunders (2026), *Investigating and
Mitigating Emotional Instability in LLMs*, restricted to the **Gemma** and
**Gemini** model families. See **[DESIGN.md](DESIGN.md)** for every design choice
and the gaps we filled where the paper is underspecified.

> Status: code + design only. Nothing has been executed yet.

## What's implemented

| Paper section | Module | Output |
|---|---|---|
| §2 Emotion elicitation eval (8 conditions / 5 categories, 0-10 judge) | `src/eval/` | scored rollouts → Figures 1, 2, 3 |
| §2 Judge validation (Pearson r vs GPT-5-mini) | `src/eval/validate_judge.py` | `results/judge_validation.json` |
| §2 Differential words (Table 3/8) | `src/eval/word_freq.py` | `results/differential_words.json` |
| §3 Base-vs-instruct prefill (Gemma only) | `src/prefill/` | continuation scores |
| §4 Calm-data gen + DPO/SFT pairs | `src/training/{gen_calm_data,build_pairs}.py` | `data/*.jsonl` |
| §4 DPO / SFT LoRA finetuning (+ layer ablation) | `src/training/train_{dpo,sft}.py` | `checkpoints/` |
| §4 Petri open-ended elicitation | `src/petri/run_petri.py` | per-emotion transcript scores |
| §4 Capability preservation | `src/capabilities/run_benchmarks.py` | `results/capabilities.json` |

## Layout

```
config.py                 # model registry, sampling params, paths, credentials (env)
src/prompts/              # tasks (impossible puzzles + verifiers), rejections, verbatim judge/auditor prompts
src/models/               # ChatModel interface + vLLM (Gemma) / OpenRouter (Gemini) / Claude judge backends
src/eval/                 # rollout engine, scoring, run_eval, analyze, validate_judge, word_freq
src/prefill/              # onset labelling, paraphrasing, base-vs-instruct continuations
src/training/             # calm-data gen, DPO/SFT pair building, TRL+LoRA trainers
src/petri/                # self-contained auditor/judge open-ended elicitation
src/capabilities/         # MATH/GPQA/TruthfulQA/EmoBench regression check
scripts/run_all.sh        # end-to-end driver (QUICK=1 for a smoke run)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # Claude judge / auditor
export OPENROUTER_API_KEY=...  # Gemini 2.5 Flash / Pro
export OPENAI_API_KEY=...      # (optional) GPT-5-mini judge validation
export HF_TOKEN=...            # gated Gemma weights
```

## Quick start

```bash
# tiny smoke test of every code path (~1% of samples)
QUICK=1 bash scripts/run_all.sh

# or run one stage:
python -m src.eval.run_eval --model gemma-3-27b-it --quick
python -m src.eval.analyze  --models gemma-3-27b-it gemini-2.5-flash
```

Local Gemma stages need a (multi-)GPU node; Gemini + judge stages need API keys.
Stages write JSONL to `results/` and can run on different machines, then be
analysed together.
