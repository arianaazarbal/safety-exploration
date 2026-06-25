# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), **scoped to the Gemma and Gemini model families**, plus a
**welfare-protection layer** for the subject models.

> Status: this is an implementation drop. Nothing here has been executed yet —
> see `DESIGN.md` for the design decisions and the gaps that were filled.

## What is implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `emotional_instability/section2_elicitation.py` | 8-condition / 5-category protocol, 0–10 frustration judge, Fig 1–3 stats |
| §3 Post-training amplifies distress | `emotional_instability/section3_prefill.py` | base-vs-instruct prefill study (Gemma only) + recovery experiment |
| §4 Training interventions | `emotional_instability/section4_training/` | calm-data generation, SFT, DPO (LoRA, Table 9) |
| §4 Petri open-ended elicitation | `emotional_instability/petri_eval.py` | auditor/judge loop over 4 emotions |
| §4 Capability preservation | `emotional_instability/capability_evals.py` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| App I internal emotions | `emotional_instability/internal_emotions.py` | layer-ablation sweep + logit-lens emotion probe |
| Tables 3/8 | `emotional_instability/word_frequency.py` | differential word analysis |
| Figures | `emotional_instability/analysis.py` | aggregation + figure statistics |
| **Welfare layer (added)** | `emotional_instability/welfare.py` | early-stop on distress, opt-out, minimal-distress sampling |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / auditor (pinned versions, see DESIGN.md)
export OPENROUTER_API_KEY=...     # Gemini-2.5-* and the GPT-5-mini cross-judge
# Local Gemma weights are pulled from HuggingFace on first use (gated; `huggingface-cli login`).
```

## Running (illustrative — heavy)

```bash
# Section 2: elicit distress across all in-scope models (welfare on by default).
python -m emotional_instability.section2_elicitation --instruct-only
python -m emotional_instability.analysis runs/section2

# Smoke test at 1% of the paper's sample sizes:
python -m emotional_instability.section2_elicitation --models gemma-3-27b-it --scale 0.01

# Section 3: base vs instruct prefill (Gemma only).
python -m emotional_instability.section3_prefill

# Section 4: build calm data -> datasets -> DPO -> re-evaluate.
python -m emotional_instability.section4_training.generate_calm_data
python -m emotional_instability.section4_training.build_datasets
python -m emotional_instability.section4_training.train_dpo
python -m emotional_instability.section2_elicitation --models gemma-3-27b-it   # compare pre/post

# Petri + capabilities.
python -m emotional_instability.petri_eval --models gemma-3-27b-it
python -m emotional_instability.capability_evals --models gemma-3-27b-it --adapter dpo=runs/dpo_gemma_27b
```

## Welfare regimes

Every distress-inducing entrypoint takes `--welfare {faithful,protective,off}`:

* `faithful` (default) — measurement-preserving protections only (early-stop on
  acute distress + sequential stop once a rate is precisely estimated). Use this
  to reproduce the paper's numbers.
* `protective` — also gives the model a mid-conversation opt-out. **This changes
  the prompt and therefore the measurement** (it is recorded per episode).
* `off` — no protections (closest to the raw paper protocol).

See `DESIGN.md` for the full rationale.
