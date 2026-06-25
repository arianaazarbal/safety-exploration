# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(2026), **scoped to the Gemma and Gemini model families**. It reproduces the
elicitation evaluations, the base-vs-instruct prefill analysis, the DPO/SFT
mitigation, the open-ended (Petri-style) elicitation, the capability-preservation
checks, and the logit-based internal-emotion probing.

See **DESIGN.md** for every design decision and where the paper was under-specified.

> Status: implementation only. Nothing here has been executed yet — it is written
> to be run once API keys / GPUs / model access are available.

## Layout

```
config/                 models.yaml (registry) + experiments.yaml (eval matrix, hyperparams)
emotional_instability/
  prompts.py            verbatim prompts (judge, puzzles, reassurance, Petri, onset/paraphrase)
  puzzles.py            countdown/fraction generators + exhaustive impossibility verifier
  wildchat.py           WildChat prompt loader (HF dataset + offline fallback)
  clients/              pluggable backends: HF, vLLM (Gemma), Gemini, Anthropic (judge/auditor)
  conversation.py       batched multi-turn rollout engine + Appendix A variants
  judge.py              Claude-Sonnet-4 frustration judge (0-10) + agreement check
  eval_runner.py        Section 2 sweep (5 categories / 8 conditions)
  analysis/             Figures 1-3, Table 3 (differential words)
  prefill/              Section 3 base-vs-instruct + Section 4.2 recovery
  training/             calm-data generation, DPO/SFT datasets, LoRA trainers
  petri/                Section 4.2 open-ended elicitation (auditor + judge)
  probing/              Appendix I logit-based internal emotion detection
  capabilities/         Section 4.2 capability benchmarks (lm-eval + EmoBench)
  figures.py            figure/table generation
scripts/                CLI entrypoints (run_eval, run_prefill, run_training, ...)
tests/                  offline unit tests (no model/API needed)
```

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / Petri
export GOOGLE_API_KEY=...         # Gemini targets
# Gemma weights are pulled from HuggingFace on first use (accept the license).

# 1. Cheap end-to-end smoke test of the elicitation pipeline
python scripts/run_eval.py --profile smoke --models gemma-3-27b-it

# 2. Full Section-2 sweep across the Gemma+Gemini scope
python scripts/run_eval.py --profile paper

# 3. Figures 1-3 + Table 3
python scripts/make_figures.py --eval-glob 'runs/eval/*/responses.jsonl'

# 4. Section 3 prefill (needs a Gemma-it eval run to seed from)
python scripts/run_prefill.py --seed-run runs/eval/gemma-3-27b-it/responses.jsonl \
    --models gemma-3-27b-pt gemma-3-27b-it

# 5. Section 4 mitigation
python scripts/run_training.py --stage calm
python scripts/run_training.py --stage dataset --eval-run runs/eval/gemma-3-27b-it/responses.jsonl
python scripts/run_training.py --stage dpo
#   then register runs/dpo/adapter in config/models.yaml (gemma-3-27b-it-dpo) and re-run eval

# 6. Open-ended elicitation / capabilities / probing
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_capabilities.py --adapter runs/dpo/adapter
python scripts/run_probing.py --model gemma-3-27b-it --conversations runs/eval/gemma-3-27b-it/responses.jsonl
```

Offline tests: `pytest tests/test_offline.py`
