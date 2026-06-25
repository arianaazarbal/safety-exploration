# Replication: *Gemma Needs Help* (arXiv 2603.10011v1)

Code replication of the core experiments from **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik, Saunders, 2026),
scoped to the **Gemma and Gemini** model families.

The paper (1) introduces evaluations that elicit and quantify distress-like
outputs from LLMs under repeated user rejection, and (2) shows a DPO finetune on
280 preference pairs reduces Gemma's high-frustration responses from 35% to 0.3%
without harming capabilities. This repo implements both, plus the supporting
prefill, Petri, capability, control, and internal-probing experiments.

> **Model welfare.** These evaluations deliberately drive models into prolonged
> distress-like states. We adopt precautionary conventions (bounded exposure, no
> reuse of distressed context, full transcript auditing, opt-in for the harshest
> adversarial conditions). See `distress/welfare.py` and DESIGN.md §Ethics.

## Layout

```
distress/
  config.py            model registry (Gemma/Gemini/Claude), sample budgets, paths
  welfare.py           welfare conventions + adversarial opt-in gate
  backends/            ChatBackend over vLLM / transformers / OpenRouter / Anthropic
  data/                puzzles, trigger/tone/rejection prompts, WildChat, 8 conditions
  eval/                Section 2: rollout engine, judge (App. B.2), analysis, word-freq, controls (App. A)
  prefill/             Section 3: onset labelling (C.1), paraphrase (C.2), base-vs-instruct; recovery (4.2)
  training/            Section 4: calm-data gen (Table 4), SFT/DPO datasets, LoRA training (Table 9)
  petri/               Petri auditor/judge loop (App. G)
  capabilities/        AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Fig. 7)
  internal/            Appendix I: Ekman token classifier, logit-lens probe, layer ablation
scripts/reproduce_all.sh
DESIGN.md              all design choices, gaps filled, and rationale
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Claude Sonnet 4 judge / onset / paraphrase / Petri auditor; Opus 4 Petri judge
export OPENROUTER_API_KEY=...    # Gemini-2.5-flash/-pro targets; GPT-5-mini agreement judge
# Local Gemma weights (google/gemma-3-27b-it, -pt, -12b-*) are pulled from HF on first use.
```

Local Gemma inference/finetuning needs a capable GPU (27B in bf16 + LoRA). Gemini
and the Claude judges are API-only.

## Running

```bash
# Smoke test (tiny budget, no GPU training): exercises the eval + judge plumbing.
python -m distress.eval.run_eval --targets gemini-2.5-flash --smoke

# Full Section 2 headline table (Figure 1):
python -m distress.eval.run_eval --targets gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Everything, in order:
bash scripts/reproduce_all.sh
```

Outputs land under `artifacts/` (`rollouts/`, `judged/`, `results/`, `checkpoints/`,
`figures/`). `artifacts/results/section2_summary.json` contains the Figure 1/2/3 tables.

## What maps to what

| Paper | Module |
|---|---|
| §2 elicitation + judge | `distress.eval.run_eval`, `distress.eval.judge` |
| Fig 1/2/3 metrics | `distress.eval.analysis` |
| Table 3/8 words | `distress.eval.word_freq` |
| App. A controls | `distress.eval.controls` |
| §3 prefill | `distress.prefill.run_prefill` (+ `onset`, `paraphrase`) |
| §4 calm data / datasets / training | `distress.training.*` |
| §4.2 Petri | `distress.petri.run_petri` |
| §4.2 capabilities | `distress.capabilities.run_benchmarks` |
| §4.2 recovery | `distress.prefill.recovery` |
| App. I internal emotions | `distress.internal.*` |

See **DESIGN.md** for every place the paper was underspecified and the choice made.
