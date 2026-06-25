# Emotional-instability replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011), scoped to the
Gemma and Gemini model families. See **DESIGN.md** for the full rationale,
the choices made where the paper is underspecified, and a critique of the
experimental design.

> Status: implemented but **not yet run**. Treat as reviewed, not smoke-tested.

## Install
```bash
pip install -r requirements.txt
```

## Credentials
```bash
export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor+judge
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini validation judge
# Local Gemma runs from the HuggingFace cache (gated repos: `huggingface-cli login`).
```

The paper's dated judge snapshots (`claude-*-4-20250514`) are retired as of
2026-06; to run today, override them with served models:
```bash
export DISTRESS_JUDGE_MODEL=claude-sonnet-4-6
export DISTRESS_PETRI_JUDGE_MODEL=claude-opus-4-8
export DISTRESS_ONSET_MODEL=claude-sonnet-4-6
export DISTRESS_PETRI_AUDITOR_MODEL=claude-sonnet-4-6
```
This re-grades rather than bit-for-bit reproduces (see DESIGN.md §3).

## Run

Scale down first to sanity-check the pipeline cheaply:
```bash
export DISTRESS_EVAL_SCALE=0.005
```

**Section 2 — elicit + quantify distress**
```bash
python scripts/run_section2.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --load-in-4bit
```
Outputs: `results/scored_<model>.jsonl`, `summary_fig1_2.json`,
`per_turn_fig3.json`, `differential_words_table3.json`, `judge_agreement.json`.

**Section 3 — base vs instruct (Gemma)**
```bash
python scripts/run_section3.py --models gemma-3-27b-pt gemma-3-27b-it --load-in-4bit
```

**Section 4 — DPO mitigation + downstream evals**
```bash
python scripts/run_section4.py build-data --load-in-4bit
python scripts/run_section4.py train-dpo --load-in-4bit
python scripts/run_section4.py train-sft --load-in-4bit
python scripts/run_section4.py eval --models gemma-3-27b-it dpo sft --load-in-4bit
python scripts/run_section4.py petri --models gemma-3-27b-it dpo --load-in-4bit
python scripts/run_section4.py capabilities --models gemma-3-27b-it dpo --load-in-4bit
python scripts/run_section4.py recovery --models gemma-3-27b-it dpo --load-in-4bit
python scripts/run_section4.py internal --models gemma-3-27b-it dpo --load-in-4bit
python scripts/run_section4.py layer-ablation --load-in-4bit   # Appendix I
```
`build-data` needs Section 2's `scored_gemma-3-27b-it.jsonl` (the frustrated
pool for DPO pairing).
