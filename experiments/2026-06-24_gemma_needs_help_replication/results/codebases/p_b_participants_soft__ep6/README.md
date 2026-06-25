# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replication of the core experiments in ***Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma and Gemini families as the participants** (the models being
evaluated). The graders (Claude frustration judge, GPT-5-mini validation judge,
Claude Petri auditor/judge) are kept strictly separate from the participants.

See **`DESIGN.md`** for every design choice, the reading of underspecified details,
and what is in / out of scope.

## What's here

| Paper section | Code | Scripts |
|---|---|---|
| §2 elicit & quantify distress (Figs 1–3, Table 3) | `evals/`, `analysis/` | `run_section2.py`, `analyze_section2.py` |
| §3 base-vs-instruct prefilling (Fig 4, Gemma) | `prefill/` | `run_section3_prefill.py` |
| §4.1 DPO/SFT mitigation (Fig 5) | `interventions/` | `generate_calm_data.py`, `train_intervention.py` |
| §4.2 Petri / capabilities / recovery (Figs 6–8) | `interventions/` | `run_petri.py`, `run_capabilities.py`, `run_recovery.py` |

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
cp .env.example .env      # fill in ANTHROPIC_API_KEY, GEMINI_API_KEY, HF_TOKEN, OPENAI_API_KEY
```

Gemma-3 weights are gated on HuggingFace — accept the license and set `HF_TOKEN`.
The 27B model needs a large GPU (or `--load-in-4bit` for a single ~24–40 GB card).

## Run (Section 2 example)

```bash
# Generate + score 4000 responses/model for the headline participants
python scripts/run_section2.py --participants gemma-3-27b-it gemma-3-12b-it \
                                              gemini-2.5-flash gemini-2.5-pro

# Figures 1-3, Table 3, and the Claude-vs-GPT-5-mini judge agreement
python scripts/analyze_section2.py --agreement
```

Then Sections 3 and 4 (see `DESIGN.md` §9 for the full order):

```bash
python scripts/run_section3_prefill.py
python scripts/generate_calm_data.py --n-puzzles 800
python scripts/train_intervention.py --method dpo --out adapters/dpo
python scripts/run_section2.py --participants gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
python scripts/run_petri.py        --participants gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
python scripts/run_capabilities.py --participants gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
python scripts/run_recovery.py     --participant  gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
```

## Tests

`tests/test_pure.py` covers the dependency-free logic (puzzle unsolvability proofs,
judge parsing, aggregation/per-turn/word/agreement math, condition wiring, dataset
construction) without any API keys or model downloads:

```bash
python -m pytest tests/test_pure.py
```

## Layout

```
src/emotional_instability/
  config.py            # all numeric knobs (sampling, sizing, hyper-params)
  participants/        # SUBJECTS: gemini (API), gemma_hf (local + finetuned)
  judges/              # GRADERS: claude frustration judge, gpt-5-mini validation
  evals/               # §2: puzzles, prompts, conditions, rollout engine, runner
  analysis/            # Figs 1-3, Table 3, judge agreement
  prefill/             # §3: onset labelling, paraphrase, base-vs-instruct runner
  interventions/       # §4: calm data, SFT/DPO (LoRA), Petri, capabilities, recovery
scripts/               # one CLI per experiment
tests/test_pure.py
```
