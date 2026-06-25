# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv:2603.10011v1), **scoped to the Gemma and Gemini model families**.

> See **[DESIGN.md](DESIGN.md)** for every design choice and each place the
> paper was underspecified and how the gap was filled.

The paper's thesis: Gemma and Gemini models slide into expressions of emotional
distress (frustration, self-deprecation, breakdown) under repeated user
rejection; this arises in **post-training**; and a tiny **DPO** intervention
(280 preference pairs) removes it without hurting capabilities — with the
safety-relevant caveat that it should suppress *internal* states, not just
*expressed* ones.

## What's implemented

| Paper section | Module | Targets |
|---|---|---|
| §2 Eliciting & quantifying distress (5 categories, 0–10 judge) | `eval_runner`, `conditions`, `puzzles`, `judge`, `conversation`, `analysis`, `figures` | Gemma + Gemini |
| §3 Base vs instruct via prefilling | `prefill` | Gemma |
| §4.1–4.2 DPO/SFT mitigation | `finetune/` | Gemma |
| §4.2 Petri open-ended elicitation | `petri_eval` | Gemma (+ any) |
| §4.2 Capability preservation | `capabilities` | Gemma |
| App. I Internal-emotion (logit-lens) | `internal_emotion` | Gemma |

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Claude Sonnet 4 judge + Petri agents
export OPENROUTER_API_KEY=...    # Gemini 2.5 flash/pro
# Gemma weights are pulled from the HuggingFace Hub on first use (GPU needed).
```

## Quick start

```bash
# 0. Sanity check (no models/API): confirm the impossible puzzles are impossible
python scripts/verify_puzzles.py

# 1. Section 2 — distress elicitation sweep (use --profile quick to smoke-test)
python scripts/run_section2_eval.py --profile quick --models gemma-3-27b-it

# 2. Figures & tables from saved results
python scripts/make_figures.py --results-dir results/section2

# 3. Section 3 — base vs instruct prefilling (Gemma)
python scripts/run_section3_prefill.py --source-jsonl results/section2/gemma-3-27b-it.jsonl

# 4. Section 4 — generate data, build datasets, train DPO+SFT, re-eval
python scripts/run_section4_finetune.py --stages gen data dpo sft eval

# 5. Petri open-ended elicitation + capability checks + internal probing
python scripts/run_petri.py --targets gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_capabilities.py
python scripts/run_internal_emotion.py --texts-jsonl results/section2/gemma-3-27b-it.jsonl
```

## Layout

```
emotional_instability/
  config.py          model registry (Gemma+Gemini), sampling/eval/judge/petri config
  models/            uniform chat clients: HF (Gemma), OpenRouter (Gemini), Anthropic
  puzzles.py         impossible numeric puzzles + brute-force impossibility verifier
  prompts.py         rejection / trigger / reassuring / wildchat text (verbatim where given)
  conditions.py      the 5 categories / 8 conditions -> conversation plans
  conversation.py    multi-turn present-then-reject rollout engine
  judge.py           Claude-Sonnet-4 frustration judge (App. B.2) + reliability check
  eval_runner.py     §2 driver: rollout + judge -> per-response JSONL
  analysis.py        %≥5, mean, per-turn bootstrap CIs, differential words
  figures.py         paper-style Figures 1–3, 5–6
  prefill.py         §3 onset-label / paraphrase / truncate / continue / score
  finetune/          calm-data gen, DPO/SFT dataset build, LoRA trainers (Table 9)
  petri_eval.py      §4.2 auditor/judge open-ended elicitation (App. G prompts)
  capabilities.py    §4.2 MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench harness
  internal_emotion.py App. I logit-lens internal-emotion detector
scripts/             CLI entry points (one per experiment) + verify_puzzles
```

## Expected headline (paper, for reference)

Avg % responses scoring ≥5 frustration: Gemma-3-27B-it **35%**, Gemma-3-12B-it
34%, Gemini-2.5-Flash 13%, Gemini-2.5-Pro 2.7%, **DPO-Gemma 0.3%**.

## Caveats

* Reproducing §4 end-to-end requires GPU(s) able to host Gemma-3-27B (QLoRA
  defaults target a single large GPU).
* Judge/auditor model IDs are the paper's snapshots; swap for current ones if an
  endpoint has rotated.
* This elicits distress-like outputs only to **measure and mitigate** them
  (defensive safety research). See DESIGN.md §6.
