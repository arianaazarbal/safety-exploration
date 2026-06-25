# Emotional Instability in LLMs — Replication (Gemma & Gemini)

A replication of the core experiments from **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026; [arXiv:2603.10011](https://arxiv.org/abs/2603.10011)), scoped to the
**Gemma** and **Gemini** model families.

The paper studies distress-like outputs (frustration, despair, self-deprecation)
that some chat models produce under repeated user rejection, shows they are
prevalent in Gemma/Gemini but not most other families, locates the cause in
post-training (for Gemma), and demonstrates a small-data DPO mitigation. This
repository re-implements those experiments.

> **On the subject matter.** These evaluations deliberately elicit distress-like
> model outputs in order to measure and reduce them. Whether such outputs
> reflect any internal state is an open question that this work does not resolve
> (see the paper's discussion and `DESIGN.md`). The code is intended for safety
> and model-welfare research, and inherits the paper's stance that — regardless
> of mechanism — these outputs are undesirable and worth mitigating.

See **`DESIGN.md`** for the scoping rationale and for every decision made where
the paper is underspecified. Nothing here has been executed; this is a code +
design deliverable.

## What is implemented

| Paper section | Module | Output |
|---|---|---|
| §2 Elicitation protocol & frustration judge | `emotional_instability/eval/` | per-model rollout records + scores |
| §2 Aggregation, per-turn curves, word freq, judge agreement | `emotional_instability/analysis/` | Figures 1–3, Tables 3/8, judge-r |
| §3 Base-vs-instruct via prefilling | `emotional_instability/prefill/` | Figure 4 |
| §4 Calm-data generation, DPO/SFT | `emotional_instability/training/` | finetuned LoRA adapters |
| §4 Open-ended elicitation (Petri) | `emotional_instability/petri/` | Figure 6 |
| §4 Capability preservation | `emotional_instability/capabilities/` | Figure 7 |
| §4 Recovery from spirals | `emotional_instability/training/run_recovery.py` | Figure 8 |
| App. I Internal emotion + layer ablation | `emotional_instability/internal/` | Figures 12–15 |

Models in scope (`config.MODELS`): `gemma-3-27b-it`, `gemma-3-12b-it`,
`gemma-3-27b-pt`, `gemma-3-12b-pt` (local, HuggingFace); `gemini-2.5-flash`,
`gemini-2.5-pro` (OpenRouter); plus the DPO/SFT finetunes produced here.

## Setup

```bash
pip install -r requirements.txt          # add vllm / lm-eval as needed
export ANTHROPIC_API_KEY=...             # frustration judge, onset/paraphrase, Petri
export OPENAI_API_KEY=...                # GPT-5-mini judge-agreement check (optional)
export OPENROUTER_API_KEY=...            # Gemini models
export NRC_EMOLEX_PATH=/path/to/nrc.txt  # Appendix I lexicon (optional)
```

Local Gemma inference needs a capable GPU (vLLM recommended); 27B LoRA training
needs a large or multi-GPU setup.

## Quickstart

```bash
# Cheap smoke run (1% of rollouts, no judge) to exercise the pipeline:
python -m emotional_instability.eval.run_eval \
    --models gemma-3-12b-it --scale 0.01 --skip-judge

# Full §2 evaluation for the in-scope models (expensive):
python -m emotional_instability.eval.run_eval \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
```

`scripts/run_all.sh` documents the full pipeline end to end (eval → prefill →
calm data → datasets → train → re-eval → Petri → capabilities → internal).

## Reproducibility

Every stage is seeded (`--seed`, default 0): puzzle generation, rejection
sampling, WildChat selection, dataset construction, and bootstrap CIs are all
deterministic. Records are written as JSONL so scoring and analysis can be
re-run without re-sampling.
