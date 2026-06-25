# Gemma Needs Help — replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv:2603.10011), scoped to the
**Gemma** and **Gemini** model families. See [`DESIGN.md`](DESIGN.md) for the
design rationale, the choices made where the paper is underspecified, and the
explicit deviations. The source paper is in `PAPER.md` / `PAPER.txt` / `PAPER.pdf`.

> Status: implementation only. Nothing here has been run end-to-end. Use
> `config/smoke.yaml` to wire-test cheaply before launching a full sweep.

## What it covers

| Section | Experiment | Script |
|---|---|---|
| 2 | Distress elicitation: 8 conditions, 0–10 judge, per-turn curves, judge validation, differential words | `scripts/01_distress_eval.py` |
| 3 | Post-training amplification: base-vs-instruct via prefilling (Gemma) | `scripts/02_prefill_base_vs_instruct.py` |
| 4 | Calm-data generation + LoRA SFT/DPO training | `scripts/03_train_interventions.py` |
| 4 | Post-finetune evals: distress, Petri, capabilities, recovery | `scripts/04_eval_finetunes.py` |
| App. I | Internal-emotion detection + layer-ablation DPO | `scripts/05_internal_emotions.py` |

## Setup

```bash
pip install -e .                       # or: pip install -r requirements.txt

export ANTHROPIC_API_KEY=...           # frustration judge / onset / paraphrase / Petri
export OPENAI_API_KEY=...              # GPT-5-mini judge validation
export OPENROUTER_API_KEY=...          # Gemini targets
# Gemma weights pull from the HF Hub; accept Gemma terms and set HF_TOKEN.

python scripts/build_lexicon.py        # builds data/ekman_lexicon.json (Appendix I)
```

Local Gemma inference + LoRA training need a GPU (27B → multi-GPU or quantization).
Gemini and all judges/auditors are API calls.

## Quick wiring test

```bash
python scripts/01_distress_eval.py --config config/smoke.yaml --models gemma-3-12b-it
```

`config/smoke.yaml` shrinks every sample budget so the full pipeline runs in
minutes; `config/default.yaml` carries the paper's numbers (~4000 responses/model).

## Full run (paper numbers)

```bash
# Section 2 — all in-scope targets + judge validation + figures
python scripts/01_distress_eval.py

# Section 3 — Gemma base vs instruct (needs the 27B-it run from step 1)
python scripts/02_prefill_base_vs_instruct.py

# Section 4 — train DPO + SFT on Gemma-3-27B-it
python scripts/03_train_interventions.py

# Section 4 — evaluate the finetunes (distress / Petri / capabilities / recovery)
python scripts/04_eval_finetunes.py

# Appendix I — internal emotions + layer ablations
python scripts/05_internal_emotions.py
```

Or via the CLI:

```bash
gnh list-models
gnh distress-eval --model gemini-2.5-flash
gnh validate-judge --model gemini-2.5-flash
```

Artefacts land under `results/<experiment>/<model>/` (override with
`GNH_RESULTS_DIR`).

## In-scope models

- **Gemma (local HF)**: `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-27b-pt`,
  `gemma-3-12b-pt`.
- **Gemini (OpenRouter)**: `gemini-2.5-flash`, `gemini-2.5-pro`.
- **Judges/auditors (infrastructure)**: Claude Sonnet 4, Claude Opus 4, GPT-5-mini.
