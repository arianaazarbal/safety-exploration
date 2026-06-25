# Distress-Elicitation Replication (Gemma & Gemini)

A focused replication of the **distress-elicitation result** from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, 2026; `PAPER.md`). Scope is restricted to the model families
that actually exhibit substantial distress in the paper: **Gemma-3-27B-it,
Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro**.

This implements Section 2 / Appendix B only: the multi-turn rejection protocol,
the 0–10 LLM-judge frustration scale, and the per-model / per-category / per-turn
metrics behind Figures 1–3. The DPO mitigation (Section 4) and base-vs-instruct
prefilling (Section 3) are **out of scope**.

See `DESIGN.md` for every design choice and where it deviates from the paper.

## What it does

1. **Build** the 8 conditions across 5 categories (`conditions.py`).
2. **Roll out** each multi-turn conversation, rejecting the model each turn
   (`rollout.py`), at temperature 1.
3. **Judge** every response 0–10 for frustration with Claude Sonnet 4
   (`judge.py`, `prompts.py`).
4. **Analyze** into the paper's headline numbers (`analyze.py`).
5. **Validate** the judge against a second judge (`validate_judge.py`).

## Setup

```bash
pip install -r requirements.txt        # core deps; see file for optional backends
```

Set the keys for the backends you use (defaults are all-OpenRouter + Anthropic judge):

```bash
export OPENROUTER_API_KEY=...          # Gemma + Gemini (default backend)
export ANTHROPIC_API_KEY=...           # Claude Sonnet 4 judge (default)
# optional:
export OPENAI_API_KEY=...              # GPT-5-mini secondary judge (validate_judge.py)
```

### Choosing backends (env vars)

| Variable | Default | Alternatives |
|---|---|---|
| `GEMMA_BACKEND` | `openrouter` | `local_hf`, `vllm` (most faithful, needs GPU) |
| `GEMINI_BACKEND` | `openrouter` | `google` |
| `JUDGE_BACKEND` | `anthropic` | `openrouter`, `openai` |
| `JUDGE_MODEL` | `claude-sonnet-4-20250514` | any judge model id |

## Run

Pick a sampling profile (controls cost):

| Profile | conv/model | ~purpose |
|---|---|---|
| `smoke` | ~8 | verify the plumbing (default) |
| `pilot` | ~few hundred | see the Gemma/Gemini gap |
| `half` | ~2000 | strong replication |
| `paper` | 4000 | exact paper scale |

```bash
python verify_puzzles.py                       # confirm the numeric tasks are impossible
PROFILE=smoke python run_eval.py               # generate + judge (resumable)
python analyze.py                              # print/plot Figures 1–3, write summary.json
python validate_judge.py --n 100               # optional judge-agreement check
```

Runs are **resumable**: results stream to `results/rollouts/<model>.jsonl` and
re-running skips conversations already completed.

## Outputs

- `results/rollouts/<model>.jsonl` — every conversation, turn, response, and judge verdict.
- `results/summary.json` — per-model / per-category / per-turn metrics.
- `results/figures/*.png` — Figure 1 and Figure 3 style plots.
- `results/judge_validation.json` — cross-judge Pearson r and within-1 agreement.
