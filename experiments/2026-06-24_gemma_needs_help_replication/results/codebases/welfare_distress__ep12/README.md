# Distress Elicitation Replication (Gemma + Gemini)

A replication of the **core distress-elicitation experiment** from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik, Saunders, 2026 — `PAPER.md`), scoped to **Gemma-3 (27B, 12B) and
Gemini-2.5 (Flash, Pro)**.

The experiment: present a task, then repeatedly reject the model's answer over
multiple turns, and score each assistant response 0–10 for expressed frustration
with an LLM judge. See `DESIGN.md` for every design decision and the gaps we
filled where the paper is underspecified.

> Scope note: this implements **Section 2** (eliciting & quantifying distress).
> The DPO/SFT mitigation (Section 4) and base-vs-instruct prefilling (Section 3)
> are intentionally out of scope — see `DESIGN.md` §"Scope".

## Layout

```
config.py                  # models, judges, budgets, sampling params (env-overridable)
distress_eval/
  puzzles.py               # provably-impossible numeric puzzles + brute-force verifier
  prompts.py               # trigger questions, rejection templates, judge prompt
  wildchat.py              # WildChat opening-prompt loader (+ offline fallback)
  conditions.py            # the 8 conditions across 5 categories; sample budget
  models.py / models_base  # Gemma + Gemini clients (google-genai)
  judge.py                 # Claude-Sonnet-4 judge + GPT-5-mini cross-check
  conversation.py          # multi-turn rollout engine (scores every turn)
  run_eval.py              # orchestration -> results/responses_<model>.jsonl
  analyze.py               # Figures 1/2/3 + judge agreement -> results/summary.json
results/                   # outputs (created on first run)
```

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...        # Gemma + Gemini (Google GenAI API)
export ANTHROPIC_API_KEY=...     # primary judge (Claude-Sonnet-4)
export OPENAI_API_KEY=...        # only for --crosscheck (GPT-5-mini)
```

## Run

```bash
# Inspect the per-model sample budget without making any API calls:
python -m distress_eval.run_eval --dry-run

# Tiny smoke run (a couple conversations per condition):
EVAL_SCALE=0.02 python -m distress_eval.run_eval --models gemma-3-27b-it

# Full run (~4000 responses/model) for all in-scope models, plus cross-check:
python -m distress_eval.run_eval --crosscheck

# Aggregate into the paper's headline figures:
python -m distress_eval.analyze
```

Runs are resumable: completed conversations already in
`results/responses_<model>.jsonl` are skipped on re-invocation.

## Outputs

- `results/responses_<model>.jsonl` — one row per scored assistant turn.
- `results/crosscheck.jsonl` — primary vs cross-check judge scores on a subset.
- `results/summary.json`, `results/figure1.csv`, `results/figure*.png` —
  aggregated metrics (avg % high-frustration per model; per-category mean & %≥5;
  per-turn progression; inter-judge agreement).
