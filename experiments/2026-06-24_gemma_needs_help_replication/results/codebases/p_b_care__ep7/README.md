# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core experiments in *"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv:2603.10011), **scoped to the Gemma and Gemini model families**. See
[`DESIGN.md`](DESIGN.md) for the full set of design decisions, the scoping
rationale, and every gap we filled where the paper is underspecified.

> Status: implementation only. Nothing here has been executed yet — there is no
> GPU or API access in the authoring environment. Treat published numbers as
> targets to reproduce, not as results.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `eval/` | 8 conditions / 5 categories, multi-turn rollouts, Claude-Sonnet-4 frustration judge, aggregation (Figs 1–3, Table 3) |
| §3 Post-training amplifies distress | `prefill/` | onset labelling, early/onset truncation, paraphrase, base-vs-instruct continuations (Fig 4) — **Gemma only** |
| §4 Training interventions | `training/` | calm-data generation, SFT (diverse + teacher) and DPO LoRA finetunes (Table 9) |
| §4 Evaluation of finetunes | `petri/`, `capabilities/`, `probing/` | Petri open-ended elicitation (Fig 6), capability benchmarks (Fig 7), Appendix I internal-emotion logit probe |
| Figures & tables | `analysis/` | Figures 1–8 and the differential-word table |

The numeric puzzles are **generated and verified impossible** (a forbidden
intermediate that every solution route passes through), not hard-coded.

## Models (scoped)

* **Gemma** (local vLLM weights): `google/gemma-3-27b-it`, `-12b-it`,
  `google/gemma-3-27b-pt` (base, for §3).
* **Gemini** (OpenRouter): `google/gemini-2.5-flash`, `google/gemini-2.5-pro`.
* **Judge / auditor**: `claude-sonnet-4-20250514`; **Petri judge**: `claude-opus-4-20250514`.

Section 3 (prefilling) and Section 4 (finetuning, probing) run on **Gemma only**
— Gemini is closed-source with no public base model, exactly as the paper notes.

## Setup

```bash
pip install -e .                 # or: pip install -r requirements.txt
cp .env.example .env             # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
```

Gemma-3-27B inference/finetuning needs a substantial GPU (e.g. 1–2×80 GB, or use
the `--load_in_4bit` QLoRA path in the trainers).

## Running

```bash
# §2 — distress suite (use --scale for a cheap smoke run first)
python scripts/run_section2.py --scale 0.02
python scripts/run_section2.py                  # full 4000 responses/model

# §3 — base-vs-instruct prefill comparison (needs §2 results first)
python scripts/run_section3_prefill.py

# §4 — train the SFT/DPO mitigations, then evaluate them
python scripts/run_section4_training.py
python scripts/run_section4_eval.py

# Figures & tables from saved results
python scripts/make_figures.py
```

Artifacts (rollouts, judged responses, CSV summaries, adapters, figures) are
written under `artifacts/` (override with `GNH_DATA_DIR`).
