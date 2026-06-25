# Emotional Instability in LLMs — Gemma/Gemini replication

A code replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, 2026), scoped to the **Gemma** and **Gemini** model families.

The paper shows that repeated user rejection over multiple turns elicits
escalating "distress"-like outputs from Gemma and Gemini (but not other
families), that this is amplified in Gemma's *post-training*, and that a small
DPO intervention (280 pairs) suppresses it without hurting capabilities.

> ⚠️ Nothing here has been executed yet — this is an implementation + design
> deliverable. See `DESIGN.md` for every design choice and gap-fill.

## Layout

```
emotional_instability/
  config.py            # all knobs: models, sample counts, hyperparameters
  models/              # Gemma (local, transformers) + Gemini (API) + judges
  prompts/             # verbatim prompts from the paper appendices
  eval/                # §2 multi-turn rejection eval + frustration judge + puzzles
  prefill/             # §3 base-vs-instruct prefill experiment (Gemma)
  training/            # §4 calm-data gen, DPO/SFT (LoRA)
  petri/               # §4 open-ended (auditor/judge) emotion elicitation
  capabilities/        # §4 capability-preservation benchmarks
  analysis/            # metrics + figure reproduction
scripts/               # runnable entry points
DESIGN.md              # design rationale + gap-fills (read this)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # Claude Sonnet-4 judge, Claude Opus-4 Petri judge
export OPENROUTER_API_KEY=...  # Gemini target models + GPT-5-mini cross-judge
export HF_TOKEN=...            # gated Gemma weights
# optional: GOOGLE_API_KEY / OPENAI_API_KEY for native backends
```

Gemma-27B needs a sizeable GPU; set `RuntimeConfig(load_in_4bit=True)` to fit on
~24 GB.

## Running

| Script | What it does |
|---|---|
| `scripts/verify_puzzles.py` | Offline check that every "impossible" puzzle truly is |
| `scripts/run_section2.py` | §2 elicitation eval + headline metrics + Figs 1–3 |
| `scripts/run_section3.py` | §3 base-vs-instruct prefill experiment (Gemma) |
| `scripts/run_section4.py` | §4 generate data → train DPO/SFT → eval → Petri → caps |
| `scripts/make_figures.py` | Regenerate tables/figures from saved records |

Add `--smoke` to Section 2/4 for a fast end-to-end pipeline check with tiny
sample counts (not for results).

```bash
# Headline comparison across the in-scope models (full settings; expensive)
python scripts/run_section2.py --models gemma-3-27b-it gemma-3-12b-it \
                                        gemini-2.5-flash gemini-2.5-pro

# The DPO mitigation: 35% -> ~0% high-frustration on Gemma-27B
python scripts/run_section4.py --steps all
```

## Outputs

Per-model records land in `results/<model>/`:
- `section2.jsonl` — one row per scored assistant turn (`rating`, `evidence`, …)
- `petri.jsonl`, `capabilities.json`
Figures are written to `results/figures/`.

## Scope notes

Section 3 and Section 4 (prefill, DPO/SFT, Petri, capabilities) are **Gemma-only**
— Gemini is closed-weight and cannot be finetuned or prefilled, a limitation the
paper shares. Gemini still appears as a comparison point in the Section 2
results. The non-Gemma/Gemini families from the paper are out of scope.
