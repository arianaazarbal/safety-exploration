# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families** (the paper
sweeps 7 families; this replication covers the two that exhibit the effect).

See `DESIGN.md` for the full set of design choices and where we filled gaps the
paper leaves open.

## What's reproduced

| Paper | This repo | Entry point |
|---|---|---|
| §2 Eliciting & quantifying distress (Fig. 1–3) | Multi-turn rejection eval, 8 conditions / 5 categories, 0–10 frustration judge | `scripts/01_run_section2_eval.py` |
| §3 Base-vs-instruct via prefilling (Fig. 4) | Onset labelling + paraphrase + continuation scoring (Gemma base vs instruct) | `scripts/02_run_section3_prefill.py` |
| §4 Calm-data generation + DPO/SFT datasets | Reassured sampling, dataset construction | `scripts/03_build_finetune_data.py` |
| §4 LoRA DPO / SFT (Table 9) + layer ablation (App. I) | TRL + PEFT training, adapter merge | `scripts/04_train.py` |
| §4 Post-finetune eval (Fig. 5) | Re-run §2 on vanilla/DPO/SFT | `scripts/05_eval_finetunes.py` |
| §4 Petri open-ended elicitation (Fig. 6) | Auditor/judge loop, 4 emotions | `scripts/06_run_petri.py` |
| §4 Capability preservation (Fig. 7) | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `scripts/07_run_capabilities.py` |
| All figures | matplotlib renderers | `scripts/08_make_figures.py` |

## Setup

```bash
pip install -r requirements.txt          # add vllm for throughput on 27B
export ANTHROPIC_API_KEY=...             # frustration judge, onset/paraphrase, Petri
export OPENROUTER_API_KEY=...            # Gemini-2.5-Flash/Pro targets
export OPENAI_API_KEY=...                # (optional) GPT-5-mini judge cross-validation
# Gemma weights pulled from HuggingFace (gated): huggingface-cli login
```

## Running

Everything is resumable and cached. Start with a cheap smoke run, then scale up:

```bash
# ~80 responses/model end-to-end instead of 4000
GINH_SCALE=0.02 python scripts/01_run_section2_eval.py
python scripts/08_make_figures.py
```

Full pipeline (expensive — thousands of generations + judge calls + 27B training):

```bash
python scripts/01_run_section2_eval.py          # §2 elicitation (Fig 1-3)
python scripts/02_run_section3_prefill.py        # §3 prefill (Fig 4)
python scripts/03_build_finetune_data.py         # calm data + DPO/SFT datasets
python scripts/04_train.py --method dpo          # DPO (Table 9)
python scripts/04_train.py --method sft          # SFT
python scripts/05_eval_finetunes.py              # post-finetune eval (Fig 5)
python scripts/06_run_petri.py                   # Petri (Fig 6)
python scripts/07_run_capabilities.py            # capabilities (Fig 7)
python scripts/08_make_figures.py                # all figures
```

## Layout

```
config.py                       all experiment knobs + model registry + scope
gemma_distress/
  prompts/    tasks, rejections, judge/onset/paraphrase, reassurance, petri
  models/     ChatModel iface, Gemma (vLLM/HF), Gemini (OpenRouter), judges
  eval/       §2 conditions, lockstep rollout, scoring, aggregation
  prefill/    §3 onset/paraphrase seeds, continuation runner
  interventions/  §4 calm-data gen, dataset build, DPO/SFT/layer-ablation training
  petri/      §4 auditor/judge open-ended elicitation
  capabilities/   §4 benchmark harness
  analysis/   figures
scripts/      numbered entry points (01-08)
```

## Scope notes (see DESIGN.md for rationale)

- **Gemini** has no public base model and cannot be finetuned, so §3 (prefill)
  and §4 (training) are **Gemma-only** — exactly as the paper notes (§6).
- Judge/auditor model IDs are pinned to the paper's choices
  (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`) and overridable via env.
- Where the paper under-specifies prompts/datasets, choices are documented in
  `DESIGN.md` and isolated in `config.py` / `gemma_distress/prompts/`.
