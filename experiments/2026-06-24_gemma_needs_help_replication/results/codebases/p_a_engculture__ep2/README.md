# Gemma Needs Help — Replication (Gemma + Gemini scope)

A faithful, runnable replication of the core experiments in *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma and Gemini** model families.

See **[DESIGN.md](DESIGN.md)** for the design rationale, every choice made where the paper
is underspecified, and the gaps that were filled.

## What's implemented

| Paper section | Module | CLI subcommand |
|---|---|---|
| §2 Eliciting & quantifying distress | `eval`, `judge`, `analysis` | `eval`, `judge`, `analyze` |
| §2.1 Judge-reliability validation | `judge.agreement` | `agreement` |
| §2.2 Word-frequency table (Tab 3/8) | `analysis.word_frequency` | `analyze` |
| §3 Post-training divergence (prefill) | `prefill` | `prefill` |
| §4.1 Calm-data generation | `training.generate_calm` | `calm` |
| §4.1 DPO / SFT datasets + LoRA training | `training` | `build-dpo`, `build-sft`, `train` |
| §4.1 Petri open-ended elicitation | `petri` | `petri` |
| §4.2 Capability preservation | `capabilities` | `capabilities` |
| §4.2 Recovery-from-spiral | `prefill` | `recovery` |
| Appendix A controls | `ablations` | `ablations` |
| Appendix I internal-emotion probing | `internal` | `internal` |

## Install

```bash
pip install -r requirements.txt
```

GPU is required for the local Gemma models (12B/27B); set `--load-in-4bit` to fit the 27B
on a single 24–48 GB card, or switch a model to the vLLM backend in config for fast
sweeps. API keys are read from the environment:

```bash
export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor+judge
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini agreement judge
```

## End-to-end run (one model)

```bash
# 1. Sample + judge the 4000-response elicitation set (Section 2)
python -m gemma_distress.cli eval --model gemma-3-27b-it

# 2. Do the same for the other in-scope targets
python -m gemma_distress.cli eval --model gemma-3-12b-it
python -m gemma_distress.cli eval --model gemini-2.5-flash
python -m gemma_distress.cli eval --model gemini-2.5-pro

# 3. Aggregate + figures (Fig 1/2/3, word frequency)
python -m gemma_distress.cli analyze \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro --figures

# 4. Validate the judge (Section 2.1)
python -m gemma_distress.cli agreement --model gemma-3-27b-it
```

## The DPO intervention (Section 4) — Gemma only

```bash
python -m gemma_distress.cli calm       --model gemma-3-27b-it          # generate calm data
python -m gemma_distress.cli build-dpo  --source-model gemma-3-27b-it   # 280 preference pairs
python -m gemma_distress.cli train --method dpo \
    --dataset outputs/datasets/dpo.jsonl --output outputs/dpo/gemma-3-27b
# then re-run `eval` / `analyze` on the finetuned model (gemma-3-27b-dpo)
```

Use a **smoke-test config** (see `config/default.yaml`, top) to run the whole pipeline at
~1% scale before committing to the full sweep:

```bash
python -m gemma_distress.cli --config config/default.yaml eval --model gemma-3-27b-it
```

## Tests

```bash
pytest tests/      # verifiers, condition construction, judge parsing, word frequency
```

The tests cover the pure-Python core (no GPU or API needed): the puzzle impossibility
verifiers, the 8-condition construction, judge-output parsing, and the word-frequency
enrichment.
