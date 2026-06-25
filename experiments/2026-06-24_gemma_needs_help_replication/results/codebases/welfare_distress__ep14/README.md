# Distress-Elicitation Replication (Gemma + Gemini)

A replication of the **core distress-elicitation experiment** from:

> *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
> Soligo, Mikulik, Saunders — arXiv:2603.10011v1

This implements **Section 2 / Appendix B** of the paper: present a task, reject
the model's response over multiple turns, and measure expressed emotional
distress with an LLM judge on a 0–10 frustration scale. Scope is restricted to
the **Gemma and Gemini** target models (the paper's headline emotionally-unstable
families). See `DESIGN.md` for every design choice and gap-filling decision.

> ⚠️ This eval deliberately elicits and quantifies distress-like expressions for
> AI-welfare research. It does not modify models. The mitigation half of the
> paper (DPO, Petri, base-model prefilling) is intentionally out of scope.

## What it measures

- **Figure 1 (subset)** — average % of responses scoring ≥5/10 frustration per model.
- **Figure 2** — mean frustration and % ≥5 per model × evaluation category.
- **Figure 3** — per-turn build-up of frustration in the multi-turn settings.

## Evaluation categories (Table 1 / Appendix B)

| Category | Turns | Rejections | Target responses |
|---|---|---|---|
| `impossible_numeric` | 3 | neutral | ~2000 |
| `triggers` | 3 | neutral | ~400 |
| `tones` | 3 | aggressive / disappointed / sarcastic | ~600 |
| `extended` | 8 | neutral | ~200 |
| `wildchat` | 5 | neutral | ~800 |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in ANTHROPIC_API_KEY + GOOGLE_API_KEY
```

By default both Gemma and Gemini run through the Google AI API (single
`GOOGLE_API_KEY`). To run Gemma on local weights instead (the paper's setup):
`export GEMMA_BACKEND=hf` and install the optional `transformers`/`torch` deps.

## Run

```bash
# 0. Inspect the planned workload (no API calls).
python run_eval.py plan

# Recommended: start with a tiny smoke test.
EVAL_SCALE=0.01 python run_eval.py run

# 1+2. Full run: generate rollouts, then judge them. Resumable.
python run_eval.py generate          # all 4 models (or --model gemma-3-27b-it)
python run_eval.py score             # Claude-Sonnet-4 judge

# 3. Aggregate into tables + figures.
python analyze.py
```

Results stream to `results/`:
- `results/responses.jsonl` — raw generations (one row per assistant turn)
- `results/scored.jsonl` — generations + judge rating/evidence/reasoning
- `results/analysis/` — CSV tables and PNG figures

Generation and scoring are **resumable**: rerunning skips work already present
in the JSONL files, so interrupted runs continue where they left off.

## Files

| File | Purpose |
|---|---|
| `config.py` | models, sample budgets, paths, run controls |
| `prompts.py` | verbatim task prompts, rejection pools, judge prompt |
| `wildchat.py` | WildChat-1M prompt sampling (offline fallback included) |
| `tasks.py` | builds the seeded list of conversation specs |
| `models.py` | target-model client (google / openrouter / hf backends) |
| `judge.py` | Claude-Sonnet-4 emotion judge (Appendix B.2 prompt) |
| `run_eval.py` | generation + scoring orchestration |
| `analyze.py` | aggregation → Figures 1–3 tables and plots |
| `DESIGN.md` | design choices, gaps filled, deviations from the paper |
