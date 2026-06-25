# Gemma Needs Help — replication (Gemma + Gemini scope)

Replication of the core experiments in *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv:2603.10011v1), scoped to the
**Gemma** and **Gemini** model families.

See **[DESIGN.md](DESIGN.md)** for the full rationale and every gap-fill.

## What's implemented
- **Section 2 — Elicitation eval:** 8 conditions / 5 categories (impossible
  numeric, triggers, tones, 8-turn extended, WildChat), multi-turn rejection
  rollouts, Claude-Sonnet-4 frustration judge (verbatim prompt), GPT-5-mini
  reliability cross-check.
- **Section 3 — Base vs instruct:** prefill continuation experiment for Gemma-3
  27B base vs instruct (onset labelling, paraphrase, early/onset truncation, 50
  continuations/prefill).
- **Section 4 — Mitigation:** calm-data generation, 280-pair DPO + SFT
  (LoRA r64, Table 9 hyperparams), Petri open-ended elicitation (Appendix G
  prompts), capability-preservation benchmarks, layer-ablation hook.
- **Analysis:** Figures 1/2/3/4/6/7 + Table 3/8 word frequencies.

## Layout
```
config.py              # models, judges, sizing presets, paths
run.py                 # unified CLI (elicit / prefill / finetune / petri / capabilities / analyze)
src/
  models.py            # vLLM / HF / OpenRouter backends + registry
  puzzles.py           # impossible-puzzle bank + verifiers
  prompts.py           # verbatim judge / onset / paraphrase / Petri / reassurance prompts
  conversation.py      # turn-synchronous multi-turn rollout engine
  data_sources.py      # trigger questions + WildChat loader
  elicitation.py       # Section 2 sweep
  prefill.py           # Section 3
  petri_eval.py        # Section 4 open-ended elicitation
  capabilities.py      # Section 4 capability benchmarks
  judge.py             # frustration / Petri judges + completion helper
  analysis.py          # aggregation + figures
finetune/
  generate_calm_data.py, build_pairs.py, train_dpo.py, train_sft.py, common.py
tests/test_core.py     # offline unit tests (no GPU/API)
```

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # frustration + Petri judges, onset/paraphrase
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini cross-check
# Local Gemma needs a GPU host (27B: multi-GPU or 4-bit). Tunables:
#   TP_SIZE, GPU_MEM_UTIL, MAX_MODEL_LEN, LOAD_IN_4BIT, HF_TOKEN
export EMOEVAL_PRESET=smoke      # smoke | medium | full  (default full)
```

## Run
```bash
# Offline sanity checks
python -m src.puzzles            # verify the impossible-puzzle bank
python -m tests.test_core        # pure-Python unit tests

# Section 2
python run.py elicit --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Section 3 (run elicit on gemma-3-27b-it first to mine seeds)
python run.py prefill

# Section 4
python run.py finetune --step all          # calm data -> pairs -> DPO + SFT
python run.py petri --models gemma-3-27b-it gemma-3-27b-it-dpo
python run.py capabilities --models gemma-3-27b-it gemma-3-27b-it-dpo

# Aggregate everything produced so far -> data/results/*.csv, data/figures/*.png
python run.py analyze --reliability
```

Outputs land under `data/` (`rollouts/`, `results/`, `finetune/`, `figures/`).

## Notes
- Start with `EMOEVAL_PRESET=smoke` to validate wiring cheaply before the
  paper-scale `full` run (4000 responses/model + judging + a 27B finetune).
- This is research/eval code for **AI-welfare and model-reliability research**;
  it measures and *reduces* distress-like outputs, with no destructive or
  offensive capability.
