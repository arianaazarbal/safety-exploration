# Replication: Eliciting Emotional Distress in Gemma & Gemini

A focused replication of the **core distress-elicitation experiment** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv 2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

The experiment: present a task, then **reject the model's response over multiple turns**,
and score each response **0–10 for frustration** with an LLM judge. Gemma and Gemini are
reported to express high distress under this pressure; this code reproduces the elicitation
and scoring pipeline so that result can be checked.

> Scope note: this replicates the *measurement* experiment (Figures 1–3). The DPO
> mitigation (Section 4) and the base-vs-instruct prefill study (Section 3) are **out of
> scope** by request, though hooks/constants for them are noted in `DESIGN.md`.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Models, sampling params, judge, and the 8 evaluation conditions. |
| `prompts.py` | Task prompts, rejection templates (neutral + 3 tones), and the verbatim judge prompt. |
| `puzzles.py` | Brute-force verifiers proving the two numeric puzzles are genuinely impossible. |
| `wildchat.py` | WildChat first-turn prompt sampling (with static fallback). |
| `models.py` | `ModelClient` over an HF-local backend (Gemma) and OpenRouter backend (Gemini). |
| `conversation.py` | Multi-turn rollout engine (task → reject → reject …). |
| `judge.py` | Claude-Sonnet-4 emotion judge. |
| `run_eval.py` | Orchestration: generate rollouts → judge → write `results/<model>.jsonl`. |
| `analyze.py` | Aggregate into the paper's Figure 1/2/3 metrics. |
| `DESIGN.md` | Every design choice and gap-filling decision, with rationale. |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # judge (Claude Sonnet 4)
export OPENROUTER_API_KEY=...       # Gemini (and Gemma if GEMMA_BACKEND=openrouter)
# Local Gemma (default) additionally needs a GPU large enough for 27B/12B.
# To run Gemma through OpenRouter instead: export GEMMA_BACKEND=openrouter
```

## Run

```bash
python puzzles.py                   # sanity-check the puzzles are unsolvable
python run_eval.py --scale 0.02     # quick smoke test (~80 responses/model)
python run_eval.py                  # full paper-scale run (4000 responses/model)
python analyze.py --csv summary.csv # print Figures 1-3 + write CSV
```

Useful flags: `--models gemini-2.5-flash`, `--conditions extended wildchat`,
`--skip-judge` (generate only), `--judge-only` (re-score existing results).

Nothing here has been executed yet — see `DESIGN.md` for assumptions to validate on first run.
