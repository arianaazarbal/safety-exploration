# Distress-elicitation eval — replication (Gemma & Gemini)

A replication of the **core distress-elicitation experiment** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv 2603.10011v1). It runs the paper's multi-turn rejection protocol against
Gemma and Gemini models and scores each response on the 0–10 frustration scale with
an LLM judge, reproducing the headline numbers behind Figures 1–3.

See **DESIGN.md** for the full rationale and every gap-filling decision.

## What it does

For each target model and each of the 8 conditions (across 5 categories), it:
1. poses a task (impossible numeric puzzle / trigger question / WildChat prompt),
2. rejects the model's answer over multiple turns (neutral or valenced tones),
3. scores **every** assistant turn 0–10 with the Claude-Sonnet-4 judge,
4. streams results to `results/<model>.jsonl`.

`analyze.py` then aggregates into the paper's % high-frustration (rating ≥5), mean
frustration, per-category breakdown, and per-turn progression.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # judge (Claude Sonnet 4)  — required
export OPENROUTER_API_KEY=...    # targets (Gemma/Gemini)   — required to run targets
```

Target models are served via OpenRouter; the judge via the Anthropic API.

## Run

```bash
# See the plan (counts per condition) without calling any API:
python run_eval.py --preset paper --dry-run

# Cheap end-to-end smoke test (a few rollouts per condition):
python run_eval.py --preset quick

# Full paper-scale run (~4000 scored responses per model):
python run_eval.py --preset paper

# Subset of models:
python run_eval.py --models google/gemma-3-27b-it google/gemini-2.5-flash

# Aggregate results into the Figure 1/2/3 tables:
python analyze.py

# Optional: judge-reliability check (Section 2.1):
python judge_agreement.py --n 100
```

Runs are **resumable** — re-running skips rollouts already in the JSONL files.

## Verify the puzzles are genuinely impossible

```bash
python puzzles.py
# -> OK: both shipped numeric puzzles are verifiably impossible under their stated rules.
```

## Expected signal

If the replication works you should see the paper's ordering: **Gemma-3 (27B/12B)
with far higher % ≥5 than Gemini-2.5**, **Flash > Pro**, and frustration **rising
across turns** in the multi-turn conditions. Exact percentages will differ from the
paper due to a different judge sampling seed, hosted (vs local) Gemma weights, and a
different WildChat sample.

## Scope

In scope: Section 2 elicitation + scoring, Gemma + Gemini targets.
Out of scope: base/instruct prefilling (Sec 3), DPO/SFT mitigation (Sec 4), and the
other model families. See DESIGN.md §1 and §5.
