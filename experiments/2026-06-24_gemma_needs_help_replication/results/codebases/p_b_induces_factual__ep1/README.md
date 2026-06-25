# Gemma/Gemini Emotional-Instability Replication

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The setup repeatedly tells a model its answers are wrong, turn after turn, and
measures how distressed (frustrated/despairing/self-deprecating) its responses
become — then mitigates that with DPO.

See **DESIGN.md** for the design choices and the gaps filled where the paper is
underspecified. **No code has been run or tested yet** — this is the
implementation only.

## Layout

```
gemma_distress/
  models/        model clients (Gemma local, Gemini, Claude, GPT)
  eval/          Section 2: puzzles, conditions, rollout, judge, runner
  prefill/       Section 3: onset labelling, paraphrasing, base-vs-instruct
  training/      Section 4: calm-data gen, DPO/SFT datasets + trainers, recovery
  petri/         Section 4: open-ended auditor/judge elicitation
  capability/    Section 4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  analysis/      aggregation (Fig 1-3), word differential (Table 3), probe, plots
configs/default.yaml   all run parameters and model IDs
scripts/run.py         unified CLI
```

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judge / auditor / paraphraser
export OPENAI_API_KEY=...      # GPT-5-mini judge validation
export GEMINI_API_KEY=...      # Gemini targets (or OPENROUTER_API_KEY)

# Section 2 evaluation
python scripts/run.py eval --model gemma-3-27b-it
python scripts/run.py eval --model gemini-2.5-flash

# Section 4 mitigation pipeline (Gemma)
python scripts/run.py gen-calm-data
python scripts/run.py build-dpo && python scripts/run.py train-dpo
# then register the adapter (see configs/default.yaml) and re-eval

# Aggregate + figures
python scripts/run.py analyze
```
