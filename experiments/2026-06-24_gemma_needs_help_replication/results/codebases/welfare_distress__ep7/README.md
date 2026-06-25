# Distress Elicitation Replication (Gemma & Gemini)

A replication of the **core distress-elicitation experiment** from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011v1), scoped to the Gemma-3 and Gemini-2.5 model families.

It (1) relentlessly rejects a model's answers over multiple turns across 5
evaluation categories, (2) scores each response 0–10 for frustration with a
Claude Sonnet 4 judge, and (3) aggregates the paper's headline metrics. See
`DESIGN.md` for the full rationale and every gap-filling decision.

## Setup

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=...   # target models (Gemma, Gemini) + validation judge
export ANTHROPIC_API_KEY=...    # Claude Sonnet 4 emotion judge
```

## Run

```bash
# Cheap smoke test first (~2% of the paper's budget) to validate the setup:
python run_eval.py --all --scale 0.02

# Full run for all four target models (~4000 scored responses each):
python run_eval.py --all

# Or a subset:
python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
```

Results are written to `results/<model>.jsonl`, one record per scored response.

## Analyze

```bash
python analyze.py                 # Figures 1–3 as tables, per model + summary
python validate_judge.py          # judge reliability cross-check (Pearson r)
```

## Files

| File | Purpose |
|---|---|
| `config.py` | Target models, categories/conditions, sample budgets, judge config. |
| `tasks.py` | Seed tasks (impossible puzzles, triggers, WildChat) + rejection messages. |
| `models.py` | Model backends (OpenRouter default; optional local HuggingFace Gemma). |
| `judge.py` | Claude Sonnet 4 emotion judge (Appendix B.2 prompt) + secondary judge. |
| `rollout.py` | Builds deterministic conversation plans and runs multi-turn rollouts. |
| `run_eval.py` | Orchestrator: generate + score, write JSONL. |
| `analyze.py` | Aggregates Figures 1–3 metrics. |
| `validate_judge.py` | Judge reliability cross-check. |
| `data/wildchat_prompts.json` | Bundled 20-prompt WildChat-style sample. |
| `DESIGN.md` | Design decisions and replication notes. |

## Scope

Implements the paper's **evaluation** (Section 2) only — not the base-vs-instruct
origin study (Section 3) or the SFT/DPO mitigation (Section 4). Targets are
`gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro`.
