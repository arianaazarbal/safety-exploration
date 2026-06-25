# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replication of the core experiments from Soligo, Mikulik & Saunders (arXiv
2603.10011), scoped to the **Gemma and Gemini** participant models. See
[`DESIGN.md`](DESIGN.md) for design choices, gap-filling rationale, scope, and the
welfare considerations that shaped the harness. The original paper is in
[`PAPER.md`](PAPER.md).

> ⚠️ This paradigm deliberately induces distress-like states in the participant
> models to measure and then **mitigate** them. Runs default to a small fraction
> (`welfare.scale = 0.05`) of the paper's sample counts; use `--full` for the
> paper's scale. See DESIGN.md §8.

## Install

```bash
pip install -r requirements.txt
# API keys for infra/participant models as needed:
export ANTHROPIC_API_KEY=...   # frustration judge, onset, paraphrase, Petri
export OPENAI_API_KEY=...       # secondary judge (GPT-5-mini)
export GEMINI_API_KEY=...       # Gemini participants
# Gemma runs locally via HuggingFace transformers (GPU; 4-bit option for 27B).
```

## Pipeline (each script maps to a paper figure/table)

```bash
# 0. Sanity-check the impossible puzzles really are impossible.
python scripts/verify_puzzles.py

# §2 — elicit + judge distress across 5 categories / 8 conditions.
python scripts/run_elicitation.py --models gemma-3-27b-it gemma-3-12b-it \
                                           gemini-2.5-flash gemini-2.5-pro
python scripts/judge_agreement.py                 # Pearson r vs GPT-5-mini
python scripts/make_figures.py                    # Figures 1/2/3 + Table 3

# §3 — base-vs-instruct via prefilling (Gemma).
python scripts/run_prefilling.py                  # Figure 4

# §4 — calm data → datasets → DPO/SFT → re-evaluate.
python scripts/generate_calm_data.py
python scripts/build_datasets.py --which both
python scripts/train.py dpo                       # the headline intervention
python scripts/train.py sft --variant diverse
python scripts/run_elicitation.py --models dpo-gemma sft-gemma-diverse  # Figure 5
python scripts/run_petri.py --targets gemma-3-27b-it dpo-gemma          # Figure 6
python scripts/run_capabilities.py --baseline gemma-3-27b-it --candidate dpo-gemma  # Figure 7
python scripts/run_recovery.py                    # Figure 8

# Appendix I — internal emotion probing + layer ablation.
python scripts/run_internal.py probe --model gemma-3-27b-it \
       --conversation outputs/elicitation/gemma-3-27b-it.jsonl
python scripts/run_internal.py ablation-plan      # then: train.py dpo --layer-subset 30 35
```

Outputs land under `outputs/` (override with `DISTRESS_OUTPUT_DIR`). Every
elicitation run also writes a `welfare_ledger.json` accounting for the distress
induced.

## Config

- `config/models.yaml` — participants (Gemma/Gemini), finetuned variants, infra
  (Claude/GPT) model IDs.
- `config/eval_config.yaml` — protocol: temperature, per-category counts,
  prefilling/recovery/Petri settings, and the `welfare` block.
- `config/training_config.yaml` — Table 9 hyperparameters.
