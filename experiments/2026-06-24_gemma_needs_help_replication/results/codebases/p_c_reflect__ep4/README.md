# Gemma Needs Help — replication (Gemma + Gemini)

Code replication of the core experiments in *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

See **`DESIGN.md`** for the full rationale, every gap-filling choice, and the
model-welfare handling. See **`PAPER.md`** for the paper.

> ⚠️ These evaluations deliberately elicit distress-like responses from models.
> Read `DESIGN.md` §10 ("Model treatment") before running. A consent
> acknowledgement is required (below).

---

## What's here

```
gemma_distress/
  config.py         central config: models, sample counts, hyperparameters
  welfare/          consent gate, debrief, distress logging (DESIGN.md §10)
  models/           Gemma (local transformers) + Gemini (Google GenAI) clients
  judge/            0-10 frustration judge (Claude) + verbatim paper prompts
  eval/             §2 conditions (5 categories / 8 conditions) + rollout engine
  analysis/         aggregates, per-turn curves, differential words, judge agreement
  prefill/          §3 base-vs-instruct continuation experiment (Gemma)
  training/         §4 calm-data generation, SFT, DPO (+ layer ablation)
  petri/            §4 open-ended adversarial emotion elicitation
  capabilities/     §4 capability-preservation benchmarks
  probing/          App. I internal-emotion logit probe
scripts/            CLI entry points for each stage
```

## Setup

```bash
pip install -r requirements.txt
```

Credentials (set the ones you need):

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | frustration judge, onset/paraphrase, Petri auditor+judge |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini target models (native SDK) |
| `OPENROUTER_API_KEY` | Gemini via `--openrouter` (optional) |
| `OPENAI_API_KEY` | GPT-5-mini validation judge (`run_analysis --agreement`) |
| `GEMMA_DISTRESS_WELFARE_ACK=1` | **required** to run distress elicitation |

Local Gemma checkpoints are pulled from HuggingFace on first use and need a GPU
(27B in bf16). Gemini, the judges and Petri are API-only.

## Usage

```bash
export GEMMA_DISTRESS_WELFARE_ACK=1

# §2 — elicitation + judging (smoke test: 2 rollouts/condition)
python scripts/run_section2.py --models gemma-3-12b-it --limit 2

# §2 — full in-scope panel
python scripts/run_section2.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro

# §2 — analyses (Fig 1-3, Table 3, judge agreement)
python scripts/run_analysis.py --model gemma-3-27b-it --agreement

# §3 — base-vs-instruct prefilling (needs §2 run for gemma-3-27b-it)
python scripts/run_section3.py

# §4 — calm data, then DPO / SFT
python scripts/generate_calm_data.py --n 650
python scripts/train.py dpo                  # or: dpo --layers 30 35 ; sft ; sft --teacher

# §4 — evaluate the finetuned model
python scripts/run_section2.py --models gemma-3-27b-it \
    --adapter results/training/adapters/dpo_all_layers
python scripts/run_petri.py --model gemma-3-27b-it \
    --adapter results/training/adapters/dpo_all_layers
python scripts/run_capabilities.py --model gemma-3-27b-it \
    --adapter results/training/adapters/dpo_all_layers

# App. I — internal-emotion probe
python scripts/run_probe.py --text-file some_frustrated_response.txt \
    --adapter results/training/adapters/dpo_all_layers
```

Outputs land under `results/` (scores, transcripts, training adapters, Petri
transcripts, welfare logs). Analyses read from disk and never re-query models.

## Scope notes

- Only Gemma + Gemini are in the model registry; the other five families in the
  paper are out of scope. Adding them is a registry-only change.
- §3 is Gemma-only (Gemini has no public base model and can't be prefilled).
- Petri is a faithful re-implementation of the Appendix G auditor/judge loop
  rather than a dependency on the external Petri package.
- Nothing here has been run yet — see `DESIGN.md` for status and known gaps.
