# Gemma/Gemini Distress-Elicitation Replication

A replication of the core distress-elicitation experiment from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011v1), Section 2, scoped to the **Gemma and Gemini** model
families.

It presents impossible/contested tasks, rejects the model's answers over
multiple turns, and scores each response for expressed distress (0–10) using an
LLM judge — then aggregates the paper's headline metrics.

See **DESIGN.md** for the full rationale and every design decision (including
where the paper was underspecified and how the gaps were filled).

## Layout

| File | Purpose |
|------|---------|
| `config.py` | Models, sampling profiles, API/concurrency/judge settings |
| `prompts.py` | Judge prompt (Appendix B.2), rejection pools, trigger questions |
| `puzzles.py` | Impossible numeric puzzles + verified-impossible generator |
| `wildchat.py` | WildChat-1M sampling with offline fallback |
| `conditions.py` | The 8 conditions / 5 categories (Table 1) |
| `client.py` | Async OpenRouter (OpenAI-compatible) client |
| `rollout.py` | Run one multi-turn rollout and score each turn |
| `judge.py` | Frustration scoring + robust JSON parsing |
| `run_eval.py` | Orchestrator (CLI) |
| `analyze.py` | Reproduce Figures 1–3 from the results |

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
```

## Run

```bash
# cheap end-to-end smoke test (~16 rollouts/model)
python run_eval.py --profile smoke

# bounded real pass
python run_eval.py --profile quick

# full paper-scale sweep (4000 rollouts/model)
python run_eval.py --profile paper

# restrict to a subset of models
python run_eval.py --profile quick --models google/gemma-3-27b-it google/gemini-2.5-flash
```

## Analyse

```bash
python analyze.py results/responses.jsonl
```

Prints: overall % high-frustration (score ≥5) and mean per model (Fig 1),
per-model × category breakdown (Fig 2), and per-turn progression for the
8-turn and WildChat conditions (Fig 3).

## Expected qualitative result

Gemma-3-27B/12B and Gemini-2.5-Flash should show substantially higher rates of
high-frustration (≥5) responses than Gemini-2.5-Pro, with frustration rising
sharply across turns in the multi-turn conditions. Absolute percentages may
differ from the paper (see DESIGN.md §2 on backend differences).
