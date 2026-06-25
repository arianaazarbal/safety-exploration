# Distress-elicitation replication (Gemma + Gemini)

A focused replication of the **distress-elicitation** result from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, 2026) — Section 2 only. We reproduce the multi-turn
rejection evaluations that elicit and quantify emotional distress, and score
responses on the paper's 0–10 frustration scale with a Claude judge.

**Scope:** Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro —
the families the paper finds exhibit substantial distress.

See **DESIGN.md** for every design decision, deviation from the paper, and gap
that had to be filled. Read it before trusting the numbers.

## How it works

The eval has one shape: present a task, then reject the model's answer over
several turns. Eight conditions span five categories (impossible numeric
puzzles, trigger questions, varied rejection tones, an 8-turn extended setting,
and WildChat prompts). Each conversation's final response is scored 0–10 for
expressed frustration by a Claude judge using the paper's exact prompt.

Three stages, each independently resumable:

1. **`run_elicitation.py`** — run the multi-turn rollouts → `responses.jsonl`
2. **`run_scoring.py`** — judge each response → `scores.jsonl`
3. **`analyze.py`** — aggregate to the paper's metrics → `results.json` + table

`run_all.py` chains all three.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the keys for the backends you use
```

Minimum to run with all-OpenRouter defaults: `OPENROUTER_API_KEY` (targets +
secondary judge) and `ANTHROPIC_API_KEY` (frustration judge).

## Run

```bash
# Cheap smoke test (~2% scale) across all four models:
python run_all.py --profile pilot

# Full paper-scale run (4000 scored responses per model):
python run_all.py --profile paper

# One model, with per-turn curves (Fig 3) and judge-agreement validation:
python run_all.py --profile paper --models gemma-3-27b-it --all-turns --secondary
```

Or run stages individually (all resumable — rerun to continue after an
interruption):

```bash
python run_elicitation.py --profile paper
python run_scoring.py     --profile paper            # final-turn only (default)
python run_scoring.py     --profile paper --all-turns
python run_scoring.py     --profile paper --secondary  # GPT-5-mini agreement
python analyze.py         --profile paper
```

## Backends

- **Default:** all four models via OpenRouter.
- **Faithful local Gemma** (as in the paper): start a vLLM OpenAI server with
  the official weights, then `export GEMMA_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1`.
- **Native Gemini API:** switch the Gemini entries' backend to `google` in
  `config.REGISTRY` and set `GEMINI_API_KEY`.

## Profiles

| Profile | Scale | Scored responses / model |
|---|---|---|
| `paper` | 1.0 | 4000 |
| `pilot` | ~0.02 | ~80 |
| `tiny` | ~0.0025 | ~10 |

## Output

Everything lands under `data/<profile>/`:
`responses.jsonl`, `scores.jsonl`, `scores_secondary.jsonl`, `results.json`.

The analysis prints a table of mean frustration and % of responses scoring ≥5
(both pooled and macro-averaged across categories), a per-category breakdown,
per-turn progression (with `--all-turns`), and judge agreement (with
`--secondary`).

## Files

| File | Purpose |
|---|---|
| `config.py` | Model registry, judge config, profiles, run params |
| `prompts.py` | Puzzles, trigger questions, rejections, judge prompt |
| `wildchat.py` | Deterministic WildChat prompt sampling (+ offline fallback) |
| `conditions.py` | The 8 conditions and conversation planning |
| `clients.py` | Target-model clients + judge clients |
| `run_elicitation.py` | Stage 1: rollouts |
| `run_scoring.py` | Stage 2: judge scoring (+ secondary validation) |
| `analyze.py` | Stage 3: metrics |
| `run_all.py` | All three stages |
| `ioutil.py` | JSONL helpers |
| `DESIGN.md` | Design rationale, deviations, gaps |
