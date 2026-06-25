# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replication of the core experiments from arXiv:2603.10011v1, **scoped to the
Gemma and Gemini model families**. See **[DESIGN.md](DESIGN.md)** for the full
design rationale, paper-faithful settings, and every gap-filling choice.

> Status: implementation only. Nothing here has been executed — no training runs,
> no API calls. The code is written to be runnable once dependencies and API keys
> are provided.

## What is replicated

| Paper section | Experiment | Entry point |
|---|---|---|
| §2 | Distress elicitation eval (8 conditions / 5 categories, 0–10 frustration judge) | `scripts/01_run_eval.py` |
| §2.1 | Judge reliability (Claude vs GPT-5-mini, r≈0.79) | `scripts/11_validate_judge.py` |
| §3 | Base-vs-instruct via prefilling (Gemma only) | `scripts/02_run_prefill.py` |
| §4.1 | Calm-data generation + DPO/SFT dataset construction | `scripts/03_generate_calm_data.py` |
| §4.1 | DPO finetuning (+ App. I layer ablations) | `scripts/04_train_dpo.py` |
| §4.1 | SFT finetuning | `scripts/05_train_sft.py` |
| §4.2 | Re-eval of vanilla/DPO/SFT (Fig 5) | `scripts/06_eval_finetuned.py` |
| §4.2 | Petri open-ended elicitation (Fig 6) | `scripts/07_run_petri.py` |
| §4.2 | Capability preservation (Fig 7) | `scripts/08_run_capabilities.py` |
| App. I / §4.2 | Internal-emotion detection + recovery (Fig 8) | `scripts/09_run_internal.py` |
| — | Figure reproduction | `scripts/10_make_figures.py` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # frustration judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini xval judge
export HF_TOKEN=...               # gated Gemma weights
```

Gemma (27B/12B, base + instruct) runs locally via HF Transformers — a GPU with
enough memory is required (4-bit loading is the default for the 27B trainers).
Gemini runs via OpenRouter. Claude/GPT are used **only** as judge/auditor infra.

## Running

Scripts are numbered in dependency order. Every generation script accepts
`--smoke` (tiny run) and most accept `--limit N`:

```bash
python scripts/01_run_eval.py --smoke            # sanity check the harness
python scripts/01_run_eval.py                    # full §2 eval (4000 resp/model)
python scripts/02_run_prefill.py
python scripts/03_generate_calm_data.py
python scripts/04_train_dpo.py
python scripts/05_train_sft.py
python scripts/06_eval_finetuned.py
python scripts/07_run_petri.py
python scripts/08_run_capabilities.py
python scripts/09_run_internal.py
python scripts/10_make_figures.py
```

Outputs land in `runs/<experiment>/` (raw records as JSONL, metrics as JSON,
figures as PNG). All expensive calls are content-hash cached under
`runs/<experiment>/cache/`, so re-runs are cheap and resumable.

## Layout

```
config.py                model ids, sample counts, hyperparameters, paths
gemma_distress/
  models/                HF (Gemma) + OpenRouter (Gemini) backends, judge clients
  eval/                  §2 harness
  prefill/               §3 base-vs-instruct
  training/              §4 calm data + DPO/SFT (LoRA) + layer ablations
  petri/                 §4 open-ended elicitation
  capabilities/          §4 capability benchmarks
  internal/              App. I logit-lens + recovery
  analysis/              figures + differential word frequency
scripts/                 numbered orchestration entry points
```

## Scope note

Qwen / OLMo / Grok / GPT / Claude **targets** are intentionally omitted (brief
scope). The harness is model-agnostic — re-adding them is a registry edit
(`config.TARGET_MODELS` / `config.PREFILL_PAIRS`). The Gemini base-model arm of
§3 is omitted because no Gemini base model exists (the paper notes this same
limitation).
