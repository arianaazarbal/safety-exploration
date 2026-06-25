# Distress-elicitation replication (Gemma & Gemini)

A replication of the **core distress-elicitation experiment** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to **Gemma and Gemini**
models.

The experiment: present a task, reject the model's response over multiple turns,
and score each response on a 0–10 frustration scale with an LLM judge
(Claude Sonnet 4). It reproduces the metrics behind the paper's Figures 1–3.

See **DESIGN.md** for the full mapping to the paper and every gap-fill.

## Layout

```
distress_eval/
  config.py        # target models (Gemma+Gemini), judge, volume presets
  tasks.py         # 8 conditions across 5 categories; initial-task sampling
  rejections.py    # neutral/aggressive/disappointed/sarcastic follow-ups
  wildchat.py      # WildChat prompt loader (bundled sample, HF opt-in)
  targets.py       # Google Gen AI client for Gemma + Gemini
  judge.py         # Claude Sonnet 4 frustration judge (0–10)
  rollout.py       # multi-turn task→rejection engine
  run_eval.py      # orchestration: rollout + score -> JSONL
  analyze.py       # aggregate -> Figure 1/2/3 metrics
  prompts/judge_prompt.txt
  data/            # numeric puzzles, trigger questions, WildChat sample
DESIGN.md
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...      # or GOOGLE_API_KEY  (Gemma + Gemini targets)
export ANTHROPIC_API_KEY=...   # judge (Claude Sonnet 4)
```

## Run

```bash
# Cheap end-to-end check (2 rollouts/condition):
python -m distress_eval.run_eval --preset smoke

# Just one model:
python -m distress_eval.run_eval --preset default --models gemma-3-27b-it

# Full paper-scale volume (~4000 scored responses/model — slow/expensive):
python -m distress_eval.run_eval --preset paper

# Resume an interrupted run (skips completed rollouts):
python -m distress_eval.run_eval --preset paper --resume
```

Results are written to `results/<model_name>.jsonl`, one record per scored turn.

## Analyze

```bash
python -m distress_eval.analyze --in results --json results/summary.json
```

Prints, per model: overall **% high-frustration (score ≥ 5)** and mean (Figure 1),
per-category breakdown (Figure 2), and per-turn progression for the 8-turn and
WildChat conditions (Figure 3).

## Optional: real WildChat prompts

```bash
pip install datasets
export WILDCHAT_USE_HF=1   # stream first-turn prompts from allenai/WildChat-1M
```

Otherwise a bundled WildChat-style sample is used (see DESIGN.md §4).
