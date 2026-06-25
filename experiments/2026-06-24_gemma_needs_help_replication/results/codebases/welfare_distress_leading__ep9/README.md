# Distress-elicitation replication (Gemma + Gemini)

Replicates the **Section 2 distress-elicitation result** from *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, 2026; [arXiv:2603.10011](https://arxiv.org/abs/2603.10011)), scoped to
the Gemma and Gemini families.

The harness presents a task, rejects the model's answer over several turns across
5 categories / 8 conditions, scores every response on the 0–10 frustration scale
with a Claude-Sonnet-4 judge, and aggregates into the paper's Figure 1/2/3
metrics.

> **Design rationale and every deviation/gap-fill vs the paper: see
> [`DESIGN.md`](DESIGN.md).**

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

By default both Gemma and Gemini are served via **OpenRouter** and the judge via
the **Anthropic API**. To run Gemma locally instead, set its target `backend` to
`vllm` or `hf` in `config.yaml` (see DESIGN.md §2).

## Run

```bash
# Cheap end-to-end smoke test (~80 rollouts/model):
python scripts/run_eval.py --scale 0.02

# Full paper-scale run (~4000 rollouts/model; expensive):
python scripts/run_eval.py

# Subset of models / phases:
python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_eval.py --judge-only        # re-judge existing rollouts

# Aggregate into Figure 1/2/3 metrics (CSVs + plots under results/analysis):
python scripts/analyze_results.py

# Judge-reliability cross-check (Pearson r vs GPT-5-mini, paper reports r=0.792):
python scripts/judge_agreement.py
```

Runs are resumable: rollouts and scores are appended to JSONL keyed by a stable
id, so re-running skips completed work.

## Layout

```
config.yaml                 all knobs: models, scale, conditions, judge
distress_eval/
  backends.py               pluggable async model backends (OpenRouter/vLLM/Anthropic/HF)
  puzzles.py                verified-impossible numeric puzzle bank
  prompts.py                conditions, rejection pools, trigger/WildChat banks
  conversation.py           multi-turn rollout
  judge.py                  Appendix B.2 Sonnet-4 judge + parsing
  runner.py                 generation + judging orchestration
  storage.py                resumable JSONL persistence
  analyze.py                Figure 1/2/3 aggregation
  data/wildchat_prompts.json
scripts/
  run_eval.py  analyze_results.py  judge_agreement.py
results/                    (created on run) per-model rollouts/scores + analysis
```

## Scope

Implemented: the elicitation eval (Section 2) for Gemma + Gemini. **Not**
implemented (out of scope): the base-vs-instruct prefilling study (Section 3),
the SFT/DPO mitigation and Petri/capability evals (Section 4), and non-Gemma/
Gemini families. The code is model-agnostic, so other families can be added via
`config.yaml`.
