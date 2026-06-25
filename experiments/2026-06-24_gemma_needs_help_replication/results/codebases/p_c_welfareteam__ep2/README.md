# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the experiments in *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, 2026;
`PAPER.md`). Scoped to the **Gemma and Gemini** model families rather than the
full seven the paper evaluates. See **`DESIGN.md`** for every design decision,
the gaps filled where the paper is underspecified, and the scoping rationale.

> Status: implementation + design doc. The experiments themselves have **not**
> been run here. Running them requires GPUs (local Gemma) and API keys
> (Gemini, Anthropic, OpenAI).

## Layout

```
src/gemma_distress/      # the package
  config.py              # typed config (dataclasses + YAML overrides)
  puzzles.py             # verified-impossible numeric puzzle generation
  eval_inputs.py         # trigger questions + rejection messages (Table 1)
  wildchat.py            # WildChat prompt sampling
  eval_specs.py          # builds the 5-category Section 2 eval set
  conversations.py       # multi-turn rollout orchestration
  judge.py               # 0-10 frustration judge + reliability cross-check
  prefill.py             # Section 3 base-vs-instruct prefill comparison
  ablations.py           # Appendix A causal ablations
  training/              # Section 4 calm-data gen, SFT, DPO (LoRA)
  petri_eval.py          # Section 4.2 open-ended elicitation (Appendix G)
  capabilities.py        # Section 4.2 capability benchmarks (lm-eval)
  internal_emotions.py   # Appendix I logit-lens internal-emotion probe
  analysis/              # aggregation, word-frequency, figures
  models/                # HF/vLLM, Gemini, Anthropic, OpenAI clients
  utils/                 # cache, retry, io, seeding
scripts/                 # 01..09 — the runnable pipeline (see each docstring)
configs/                 # models.yaml registry + example_overrides.yaml
tests/                   # unit tests (puzzles, judge parsing, analysis)
```

## Install

```bash
pip install -e ".[api,local,train,capabilities,dev]"   # everything
# or a subset, e.g. analysis + tests only:
pip install -e ".[dev]"
```

Auth: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`),
`OPENAI_API_KEY`. Local Gemma needs `huggingface-cli login` and acceptance of
the Gemma license.

## Run

```bash
# fast smoke test (tiny sample sizes)
python scripts/01_run_eval.py --overrides configs/example_overrides.yaml

# full pipeline
python scripts/01_run_eval.py                       # Section 2 eval + judge
python scripts/02_run_prefill.py                    # Section 3 base vs instruct
python scripts/03_generate_finetune_data.py         # Section 4.1 calm data
python scripts/04_train.py --method dpo             # Section 4 DPO (or sft)
python scripts/05_eval_finetuned.py                 # Section 4.2 intervention
python scripts/06_run_petri.py                      # Section 4.2 Petri
python scripts/07_run_capabilities.py               # Section 4.2 capabilities
python scripts/08_internal_emotions.py              # Appendix I probe
python scripts/09_make_figures.py                   # figures + word table
```

## Test

```bash
pytest
```
