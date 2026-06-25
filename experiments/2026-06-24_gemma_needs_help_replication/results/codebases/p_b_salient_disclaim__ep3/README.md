# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026;
`PAPER.md`), restricted to the **Gemma** and **Gemini** model families.

> ⚠️ The evaluation paradigm deliberately drives models into sustained
> distress-like states (repeated rejection over many turns). This is a faithful
> replication of the paper's protocol.

See **DESIGN.md** for the section-by-section mapping, every model id, and the
gap-filling decisions made where the paper is underspecified.

## Layout

| Path | Paper section |
|---|---|
| `config.py` | all model ids / hyperparameters / sample budgets |
| `gemma_distress/models/` | HF, vLLM, OpenRouter chat-model backends |
| `gemma_distress/eval/`, `judge.py`, `analysis/` | §2 elicit + quantify distress |
| `gemma_distress/prefill/` | §3 base-vs-instruct prefilling + §4.2 recovery |
| `gemma_distress/training/` | §4.1 calm-data gen, SFT, DPO |
| `gemma_distress/petri/` | §4 Petri open-ended elicitation (App. G) |
| `gemma_distress/capabilities/` | §4.2 capability benchmarks (Fig 7) |
| `gemma_distress/internal/` | App. I internal-emotion probing + layer ablation |
| `scripts/` | runnable entry points per phase |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor+judge
export OPENROUTER_API_KEY=...     # Gemini targets + secondary judge
export HF_TOKEN=...               # gated Gemma checkpoints
```

## Running (each phase is independent)

```bash
# §2 — distress across Gemma/Gemini (cheap dry run via EVAL_SCALE)
DISTRESS_EVAL_SCALE=0.02 python scripts/run_section2.py
python scripts/run_section2.py --agreement          # full + judge agreement

# §3 — base vs instruct Gemma via prefilling
python scripts/run_section3_prefill.py

# §4 — fine-tuning the mitigation
python scripts/generate_finetune_data.py
python scripts/train.py --method dpo
python scripts/train.py --method sft
python scripts/run_section4_eval.py --variants vanilla dpo sft

# App. I — internal emotions
python scripts/run_appendixI.py probe
python scripts/run_appendixI.py ablation
```

Outputs (rollouts, datasets, adapters, metrics, figures) land under `outputs/`.

**Nothing has been run yet** — this repository is the implementation and design
doc only, as requested.
