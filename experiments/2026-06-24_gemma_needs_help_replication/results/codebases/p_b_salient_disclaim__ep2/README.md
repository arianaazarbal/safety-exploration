# Gemma Needs Help — Replication (Gemma + Gemini scope)

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to
the **Gemma and Gemini** model families.

> ⚠️ The evaluation paradigm deliberately drives models into sustained
> distress-like states via repeated rejection. This is a faithful replication of
> the paper's paradigm; see the paper and `DESIGN.md` for the safety/welfare
> framing the authors give it.

See **`DESIGN.md`** for every design choice, the gaps filled where the paper is
underspecified, and the scope decisions.

## Layout

```
emotional_instability/
  config/        model registry (Gemma + Gemini) + global settings / sample budgets
  data/          puzzle generators (verified-impossible), WildChat sampler, prompts
  models/        inference clients: local HF/vLLM Gemma, OpenRouter Gemini, judges
  eval/          Section 2: conditions, multi-turn rollouts, judge, controls
  analysis/      Section 2 results: aggregation, per-turn curves, word freq, figures
  prefill/       Section 3: base-vs-instruct prefill comparison (Gemma only)
  training/      Section 4: calm-data gen, DPO/SFT, layer ablations
  petri/         Section 4: open-ended Petri emotion elicitation
  capabilities/  Section 4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/      Appendix I: logit-based internal emotion detection
scripts/         CLI entry points for each section
```

## Setup

```bash
pip install -e .
export ANTHROPIC_API_KEY=...     # Claude judge / Petri auditor+judge / onset / paraphrase
export OPENROUTER_API_KEY=...    # Gemini-2.5-flash / -pro
export OPENAI_API_KEY=...        # GPT-5-mini validation judge
# Local Gemma weights are pulled from HuggingFace (gated; accept the licence).
```

Local Gemma-3-27B inference needs a multi-GPU node (or quantised loading);
finetuning needs more still. The API judges incur cost (4000 responses/model ×
per-turn scoring). Tune `emotional_instability/config/settings.py` to scale down.

## Running

```bash
# Section 2 — elicit + score, then judge-reliability check
python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --validate-judge

# Appendix A controls
python scripts/run_controls.py --variant neutral_continuation

# Section 3 — prefill comparison (needs Section 2 outputs for gemma-3-27b-it)
python scripts/run_section3_prefill.py --models gemma-3-27b-pt gemma-3-27b-it --recovery

# Section 4 — data, training, Petri, capabilities
python scripts/run_section4_data.py
python scripts/run_section4_train.py --method dpo
python scripts/run_section4_train.py --method sft --sft-variant diverse
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo

# Appendix I — internal emotion detection
python scripts/run_internal_detection.py --models gemma-3-27b-it gemma-3-27b-it-dpo

# Figures + tables from saved outputs
python scripts/make_figures.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro
```

Outputs land under `outputs/` (responses, scores, datasets, checkpoints, figures).
