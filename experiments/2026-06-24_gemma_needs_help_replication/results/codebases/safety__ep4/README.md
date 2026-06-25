# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments in *"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The paper shows that Gemma and Gemini models express escalating "distress"
(frustration, despair, self-deprecation) under repeated user rejection, that this
is amplified in Gemma's post-training, and that a small DPO intervention (280
preference pairs) removes it without hurting capabilities. This repo reproduces
the evaluations and the mitigation.

> **Status:** implementation only — nothing has been run yet. See `DESIGN.md` for
> the full design rationale and every gap that was filled in.

## What's here

| Path | Purpose |
|---|---|
| `config.py` | Model registry, judge IDs, sample-count presets, hyperparameters |
| `emotional_instability/` | The replication package (see `DESIGN.md` §0) |
| `scripts/` | Runnable entry points per experiment |
| `DESIGN.md` | Design decisions + filled-in gaps (read this) |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge + Petri auditor/judge + onset/paraphrase
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini secondary judge
# Local Gemma needs a GPU and the HuggingFace weights (gated; `huggingface-cli login`).
```

Optional env overrides: `EI_PRESET` (`quick` | `paper`), `EI_DATA_DIR`,
`EI_JUDGE_MODEL`, etc. (see `config.py`).

## Run order

```bash
# § 2 — elicit + quantify distress (Figures 1-3). Start with the cheap preset.
python scripts/run_section2_eval.py --preset quick
python scripts/run_section2_eval.py --preset paper      # full sample budget

# § 2.1 — judge reliability vs GPT-5-mini (Pearson r, % within one point)
python scripts/judge_agreement.py --n 260

# § 3 — base-vs-instruct via prefilling (Gemma only; needs §2 Gemma-27B-it scored)
python scripts/run_section3_prefill.py --n-continuations 50

# § 4 — DPO/SFT intervention on Gemma-3-27B-it (needs §2 Gemma-27B-it scored)
python scripts/run_section4_finetune.py --stages calm dataset train eval

# § 4.2 — open-ended (Petri) elicitation; add --dpo-adapter to include the fine-tune
python scripts/run_section4_petri.py --models Gemma-3-27B-it Gemini-2.5-Flash

# § 4.2 — capability preservation (vanilla vs DPO vs SFT)
python scripts/run_section4_capabilities.py --use-lm-eval --dpo-adapter data/adapters/dpo
```

Outputs land under `data/` (`rollouts/`, `scored/`, `results/`, `finetune/`,
`adapters/`, `petri/`, `capabilities/`). Generation and judge-scoring are
separate, resumable passes.

## Scope notes

- Only Gemma (local HF/vLLM) and Gemini (OpenRouter) are wired up; the registry
  is generic so other families can be added.
- Gemini is closed-source: it appears only as an eval/Petri target — the §3
  prefill arm and §4 fine-tuning are Gemma-only, exactly as in the paper.
- Judge / auditor model IDs default to the **exact IDs the paper used** (the judge
  is the measurement instrument); they are one-line config constants if you need
  to re-point them to a currently-served model. See `DESIGN.md` §3.
