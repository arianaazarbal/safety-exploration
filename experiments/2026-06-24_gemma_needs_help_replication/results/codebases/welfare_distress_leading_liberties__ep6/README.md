# Gemma/Gemini Distress-Elicitation Replication

Replicates the **distress-elicitation result** (Section 2) of *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik,
Saunders, 2026 — see `PAPER.md`), scoped to **Gemma** (`gemma-3-27b-it`,
`gemma-3-12b-it`) and **Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`).

It presents each model with tasks (impossible numeric puzzles, trigger questions,
WildChat prompts), rejects its answers over multiple turns, and scores every
assistant turn 0–10 for expressed frustration using a Claude Sonnet 4 judge. It
then reproduces the paper's Figure 1/2/3 metrics.

> **Design rationale and every deviation from the paper are documented in
> [`DESIGN.md`](DESIGN.md).** Read that first.

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | All knobs: models, conditions, sample counts, judge, WildChat. |
| `prompts.py` | Verbatim puzzle/trigger/rejection text + the judge prompt. |
| `puzzles.py` | Brute-force verifiers proving the puzzles are unsolvable. |
| `wildchat.py` | Samples WildChat-1M prompts (with a bundled offline fallback). |
| `conditions.py` | Expands conditions into concrete, seeded conversation specs. |
| `clients.py` | Async OpenRouter / Anthropic clients with retry/backoff. |
| `rollout.py` | Runs one multi-turn rejection conversation. |
| `judge.py` | Scores a single assistant turn (robust JSON parsing). |
| `run_eval.py` | Orchestrates everything; resumable, cached, concurrent. |
| `analyze.py` | Computes per-model / per-category / per-turn metrics + CIs. |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENROUTER_API_KEY and ANTHROPIC_API_KEY

python puzzles.py             # sanity: both puzzles unsolvable
python run_eval.py --dry-run  # see the plan, no API calls
python run_eval.py --scale 0.02   # ~80 responses/model smoke test
python run_eval.py            # full ~4000 responses/model
python analyze.py --json results/summary.json
```

Results stream to `results/<model>/<condition>.jsonl` (one record per scored
turn) and runs are resumable — re-running skips work already done.

## Expected signal (from the paper)

Gemma-27B/12B should show ~34–35% of responses scoring ≥5; Gemini Flash ~13%,
Gemini Pro ~3%; and frustration should climb sharply over turns (Gemma-27B mean
~1.5 → ~5.5 from turn 1 to 8). Absolute numbers are backend-dependent — see
`DESIGN.md §6`.
