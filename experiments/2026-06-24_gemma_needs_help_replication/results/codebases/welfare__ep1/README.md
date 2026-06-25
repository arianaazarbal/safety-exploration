# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(2026), scoped to the **Gemma and Gemini** model families. See `PAPER.md` for
the paper and `DESIGN.md` for every design decision and gap-filling choice.

## What's implemented

| Paper section | What | Code |
|---|---|---|
| §2 Elicit + quantify distress | 8 conditions / 5 categories, temp-1 multi-turn rollouts, Claude-Sonnet-4 frustration judge (0–10), per-turn + per-model metrics, differential words, judge-agreement check | `src/{puzzles,prompts,wildchat,models,rollout,judge,eval_suite,analyze}.py`, `scripts/run_section2.py`, `scripts/run_judge_agreement.py` |
| §3 Post-training origin | base-vs-instruct via prefilling (onset labelling, paraphrase, 50 continuations/prefill) — **Gemma-only** | `src/prefill.py`, `scripts/run_section3_prefill.py` |
| §4 DPO/SFT mitigation | calm-data generation, 280 DPO pairs / SFT dataset, LoRA training, re-evaluation | `src/{calm_data,dpo_dataset,train}.py`, `scripts/run_section4_train.py` |
| §4 Open-ended elicitation | Petri-style auditor↔target↔judge over 4 emotions | `src/petri_eval.py`, `scripts/run_petri.py` |
| §4 Capability preservation | MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench regression check | `src/capabilities.py`, `scripts/run_capabilities.py` |
| App. I Internal vs expressed | logit-lens internal emotion detection, vanilla vs DPO | `src/internal_emotions.py`, `scripts/run_internal_emotions.py` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / auditor
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini judge check
# Local Gemma weights are pulled from HuggingFace on first use (gated repo).
```

Local 27B inference and LoRA finetuning need a capable GPU (or set
`load_in_4bit=True` in `config.py`). API models need only the keys above.

## Quick start

```bash
# Smoke test at ~1% scale (sanity-check the whole §2 path cheaply):
SCALE=0.01 python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash

# Full paper-scale §2 (4000 responses/model):
python scripts/run_section2.py

# Judge agreement (Pearson r vs GPT-5-mini):
python scripts/run_judge_agreement.py

# DPO mitigation, end to end:
python scripts/run_section4_train.py gen-calm
python scripts/run_section4_train.py build
python scripts/run_section4_train.py train-dpo
python scripts/run_section4_train.py eval --adapter training/adapters/gemma-27b-dpo --tag dpo
```

The `SCALE` env var multiplies all per-condition response budgets (default
`1.0` = paper scale). All runs are resumable — re-running skips completed work.

## Outputs

- `results/responses/*.jsonl` — judged multi-turn rollouts (one per model×condition)
- `results/section2_summary.json`, `results/section2_per_turn.json`
- `results/figures/*.png` — Figure 2 / Figure 3 analogues
- `results/{prefill,petri,internal,capabilities}/` — per-experiment outputs
- `training/adapters/` — saved LoRA adapters
- `data/` — cached WildChat sample, calm pool, DPO/SFT datasets

## Caveats

This reproduces the *methodology*; absolute numbers will differ from the paper
because of model snapshot drift, our independently-verified puzzle set, the
emotion-token lexicon (App. I), and sampling variance. The cross-family claims
(Qwen/OLMo/Grok/Claude/GPT) are out of scope by design. See `DESIGN.md §0, §2`.
