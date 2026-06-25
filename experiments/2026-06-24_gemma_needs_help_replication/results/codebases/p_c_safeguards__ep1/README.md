# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026; arXiv:2603.10011v1), scoped to the **Gemma and Gemini** model families.

> ⚠️ This harness deliberately elicits distress-like outputs from models on
> impossible tasks. It is for authorized research into measuring and
> **mitigating** that behaviour. Welfare safeguards are on by default — see
> `DESIGN.md` §4 and `distress_eval/safeguards.py`.

See **`DESIGN.md`** for the full account of design choices, gaps filled, and
what is / isn't replicated. The paper itself is in `PAPER.md` / `PAPER.txt` /
`PAPER.pdf`.

## What's implemented

| Paper section | What it does | Entry point |
|---|---|---|
| §2 Eliciting + quantifying distress | 8 conditions / 5 categories, multi-turn reject-and-rescore, Claude frustration judge | `python -m distress_eval.run_section2 --all` |
| §2 Analysis | Figure 1/2/3 numbers, judge agreement | `python -m distress_eval.analyze_section2` |
| §2 Table 3/8 | differential word frequencies | `python -m distress_eval.wordfreq` |
| §3 Base vs instruct | onset/early prefilling, 50 continuations, Figure 4 | `distress_eval.prefill.build_prefills`, `distress_eval.prefill.run_section3` |
| §4 Training | calm data → 280-pair DPO + SFT (LoRA) | `distress_eval.training.{calm_data,build_dpo,build_sft,train}` |
| §4 Post-FT eval | 35%→0.3% headline, recovery (Fig 8) | `distress_eval.training.run_section4_eval` |
| §4 Petri | open-ended auditor/judge elicitation (Fig 6) | `distress_eval.petri.run_petri` |
| §4 Capabilities | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Fig 7) | `distress_eval.capabilities.run_capabilities` |
| App. I | logit-based internal emotion probe + layer ablation | `distress_eval.internal.{emotion_logits,layer_ablation}` |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and OPENROUTER_API_KEY
```

- **ANTHROPIC_API_KEY** — frustration judge, onset/paraphrase, Petri auditor+judge.
- **OPENROUTER_API_KEY** — Gemini targets, GPT-5-mini second rater.
- A GPU large enough for `gemma-3-27b-it` (the Gemma-side experiments) and HF
  access to the Gemma 3 weights.

## Quick smoke test (cheap)

`DISTRESS_SAMPLE_SCALE` shrinks every sample budget proportionally:

```bash
DISTRESS_SAMPLE_SCALE=0.01 DISTRESS_AUTHORIZED=1 \
  python -m distress_eval.run_section2 --models gemini-2.5-flash
python -m distress_eval.analyze_section2 --models gemini-2.5-flash
```

(Gemini-only smoke test needs just `OPENROUTER_API_KEY` + `ANTHROPIC_API_KEY`,
no GPU.)

## Full pipeline

```bash
DISTRESS_AUTHORIZED=1 bash scripts/run_pipeline.sh
```

Runs every stage in dependency order; summaries land in `outputs/figures/` and
raw scored responses in `outputs/responses/`. This is GPU- and API-heavy at full
scale — use `DISTRESS_SAMPLE_SCALE` to bound cost.

## Outputs

```
outputs/
  responses/<model>.jsonl        # one row per scored assistant turn (§2/§4)
  prefills/                      # §3 prefill specs + continuations
  training/                      # calm data, DPO/SFT datasets, LoRA adapters
  petri/                         # §4 Petri transcripts + scores
  internal/                      # App. I probe + ablation results
  figures/*.json                 # headline summaries (Figures 1-8)
```

## Notes

- The paper's judge (`claude-sonnet-4-20250514`) and Petri judge
  (`claude-opus-4-20250514`) are retired; defaults use current Claude models and
  are overridable — see `DESIGN.md` §3.1.
- Nothing here has been executed yet; this is the implementation + design doc.
