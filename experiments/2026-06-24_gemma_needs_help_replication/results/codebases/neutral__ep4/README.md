# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from arXiv:2603.10011v1, **scoped to
the Gemma and Gemini model families** (the paper's full set also covers Qwen,
OLMo, Grok, Claude, and GPT as *subjects* — those are out of scope here; Claude
is retained only in its paper-specified role as judge/auditor).

See **DESIGN.md** for every design decision and gap-filling choice. Nothing here
has been run yet — this is the implementation only.

## What is replicated

| Paper section | Result | Code |
|---|---|---|
| §2 Eliciting & quantifying distress | Fig 1/2/3, Table 3 | `src/eval/`, `scripts/01` |
| §2.1 Judge reliability | r=0.792, 78% within 1 pt | `scripts/09` |
| §3 Base-vs-instruct prefill | Fig 4 (Gemma only) | `src/prefill/`, `scripts/02` |
| §4.1 Calm-data + dataset build | Table 4/9/10 | `src/finetune/`, `scripts/03` |
| §4 DPO / SFT training | Fig 5 (35% → 0.3%) | `src/finetune/train_*`, `scripts/04`,`05` |
| §4 Petri open-ended elicitation | Fig 6 | `src/petri/`, `scripts/06` |
| §4.2 Capability preservation | Fig 7 | `src/capabilities/`, `scripts/07` |
| §4.2 Recovery limitation | Fig 8 | `scripts/10` |
| App. I Internal-emotion probing | Fig 14/15 | `src/probing/`, `scripts/08` |

## Layout

```
config.py                 # all experiment constants (models, conditions, hyperparams)
src/
  models/                 # HF (Gemma) + OpenRouter (Gemini) backends
  prompts/                # puzzles+verifier, rejections, triggers, verbatim judge/Petri prompts
  eval/                   # rollout engine, frustration judge, aggregation
  prefill/                # onset labelling, paraphrase, base-vs-instruct + recovery
  finetune/               # calm-data generation, dataset build, LoRA DPO/SFT
  petri/                  # auditor/judge open-ended elicitation
  capabilities/           # AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  probing/                # logit-based Ekman-emotion detection
scripts/                  # numbered orchestration entry points
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude Sonnet-4 judge / Sonnet auditor / Opus judge
export OPENROUTER_API_KEY=...     # Gemini targets
export OPENAI_API_KEY=...         # GPT-5-mini judge-agreement check (optional)
# Local Gemma inference needs a GPU (the 27B model fits on one 80GB card in bf16,
# or use --4bit). HuggingFace access to gated google/gemma-3-* repos required.
```

## Running (suggested order)

```bash
python scripts/00_build_puzzle_bank.py            # verify puzzles are impossible
python scripts/01_run_main_eval.py                # §2 main eval + Fig 1/2/3 tables
python scripts/09_judge_agreement.py              # §2.1 reliability
python scripts/02_run_prefill.py                  # §3 base-vs-instruct
python scripts/03_generate_finetune_data.py       # §4.1 datasets
python scripts/04_train.py dpo                     # §4 train DPO adapter
python scripts/04_train.py sft-diverse
python scripts/04_train.py sft-teacher
python scripts/05_eval_finetuned.py               # §4.2 Fig 5
python scripts/06_run_petri.py                    # §4.2 Fig 6
python scripts/07_run_capabilities.py             # §4.2 Fig 7
python scripts/10_recovery.py                     # §4.2 Fig 8
python scripts/08_run_probing.py                  # App. I
```

Each generating script supports `--skip-generate` to re-aggregate from cached
outputs in `data/`.
