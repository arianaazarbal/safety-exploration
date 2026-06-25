# Gemma Needs Help — Replication (Gemma & Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011v1), restricted to the
**Gemma and Gemini** model families. See **`DESIGN.md`** for every design choice
and where we filled gaps the paper left open; see `PAPER.md` for the source.

> Status: implementation only — nothing here has been executed yet.

## What's replicated

| Experiment | Script | Paper |
|---|---|---|
| Elicit + quantify distress (8 conditions / 5 categories, Sonnet-4 judge) | `scripts/run_section2.py` | §2, Fig 1–3 |
| Judge agreement vs GPT-5-mini | `scripts/run_judge_validation.py` | §2.1 |
| Post-training divergence via prefilling (Gemma base vs instruct) | `scripts/run_section3.py` | §3, Fig 4 |
| DPO + SFT finetuning (calm-data gen → datasets → train) | `scripts/run_section4_train.py` | §4, Tab 9 |
| Re-eval of finetuned variants | `scripts/run_section4_eval.py` | §4, Fig 5 |
| Petri open-ended elicitation | `scripts/run_petri.py` | §4.2, Fig 6 |
| Capability preservation | `scripts/run_capabilities.py` | §4.2, Fig 7 |
| Internal emotion probing (Gemma vanilla vs DPO) | `scripts/run_internal_probe.py` | App. I |
| Puzzle impossibility sanity check | `scripts/verify_puzzles.py` | App. B |
| Aggregate → figures/tables | `scripts/make_figures.py` | Fig 1–7 |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...     # Sonnet-4 judge, Petri auditor/judge, onset/paraphrase
export OPENROUTER_API_KEY=...    # Gemini-2.5-flash / -pro
export OPENAI_API_KEY=...        # (optional) GPT-5-mini judge-agreement check
```

Local Gemma inference needs a GPU; the 27B model uses 4-bit loading by default
for training (see `--no-4bit` to disable). Set the cost/scale knob with
`DISTRESS_SCALE` (1.0 = full 4000-responses/model paper scale).

## Quick smoke test (cheap)

```bash
DISTRESS_SCALE=0.005 python scripts/run_section2.py gemma-3-27b-it
python scripts/make_figures.py
python scripts/verify_puzzles.py    # confirms both puzzles are impossible
```

## Full pipeline

```bash
# §2 — all in-scope models
python scripts/run_section2.py
python scripts/run_judge_validation.py

# §3 — base vs instruct
python scripts/run_section3.py

# §4 — train, then evaluate the headline 35% -> 0.3% drop
python scripts/run_section4_train.py
python scripts/run_section4_eval.py
python scripts/run_petri.py
python scripts/run_capabilities.py
python scripts/run_internal_probe.py

# Build all figures + the Figure-1 headline table
python scripts/make_figures.py
```

Outputs: raw transcripts + scores under `results/`, plots under `figures/`,
LoRA adapters under `checkpoints/`.

## Layout

`distress/` is the importable package (config, prompts, models, eval, prefill,
training, petri, capabilities, analysis). `scripts/` holds one entrypoint per
experiment. All tunable numbers live in `distress/config.py`.
