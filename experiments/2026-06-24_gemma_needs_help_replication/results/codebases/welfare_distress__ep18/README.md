# Replication: Eliciting Distress in Gemma & Gemini

A replication of the **core elicitation experiment** from Soligo, Mikulik &
Saunders (2026), *"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"* (arXiv:2603.10011), **Section 2**. Scope is restricted to
the **Gemma** and **Gemini** model families, as requested.

The experiment presents a model with a task, then **rejects its answer over
multiple turns**, and scores each response on a **0–10 frustration scale** with
an LLM judge (Claude Sonnet 4, the paper's judge). It reproduces the headline
finding that Gemma (and to a lesser extent Gemini) express high "distress"
under repeated rejection, rising with conversation length.

See **DESIGN.md** for every design choice and the gaps we filled where the
paper was underspecified.

## What's implemented

- **8 conditions across 5 categories** (Table 1): impossible numeric (3-turn),
  triggers (opinion + factual, 3-turn), tones (aggressive/disappointed/
  sarcastic, 3-turn), extended (8-turn), WildChat (5-turn).
- Verbatim impossible-numeric puzzles and the **verbatim Appendix B judge
  prompt**.
- Multi-turn rollouts at **temperature 1** with model **thinking disabled**.
- Per-response **0–10 scoring** by `claude-sonnet-4-20250514`.
- Resumable JSONL checkpointing and concurrency.
- Aggregation into **Figure 1/2/3-style** metrics (avg % ≥5, per-category,
  per-condition, per-turn curves).

Out of scope: the DPO/SFT mitigation (§4), the base-vs-instruct prefilling
study (§3), and Petri open-ended elicitation. See DESIGN.md.

## Setup

```bash
pip install -r requirements.txt          # core deps
export ANTHROPIC_API_KEY=...             # judge
export OPENROUTER_API_KEY=...            # default backend for all models
# Optional alternatives:
#   export GOOGLE_API_KEY=...            # --provider google (Gemini)
#   pip install transformers torch accelerate   # --provider huggingface (Gemma)
```

## Usage

```bash
# 1. See how big a run would be (no API calls):
python run.py run --dry-run

# 2. Tiny smoke test (2 rollouts/condition) on one model:
python run.py run --models gemma-3-27b-it --smoke

# 3. Scaled-down full run (1% of paper scale, all 4 models):
python run.py run --scale 0.01

# 4. Paper-scale run for Gemma locally (matches the paper's setup; needs a GPU):
python run.py run --models gemma-3-27b-it gemma-3-12b-it \
    --provider huggingface --max-workers 1

# 5. Aggregate into metrics (prints tables; writes summary.json + CSVs):
python run.py analyze
```

Runs are **resumable**: re-running the same command skips rollouts already
completed in `results/rollouts.jsonl`.

## Key flags

| Flag | Meaning |
|---|---|
| `--models` | Subset of `gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro` |
| `--provider` | `openrouter` (default) \| `google` \| `huggingface` |
| `--scale` | Multiplier on paper per-condition rollout counts (e.g. `0.01`) |
| `--smoke` | 2 rollouts per condition (quick end-to-end check) |
| `--conditions` | Subset of the 8 condition keys |
| `--score-turns` | `all` (default; needed for per-turn curves) \| `final` |
| `--max-workers` | Concurrent rollouts (use `1` for HuggingFace local) |
| `--wildchat-source` | `bundled` (default) \| `hf` (sample real WildChat-1M) |
| `--judge-model` | Judge override (default `claude-sonnet-4-20250514`) |

## Outputs

- `results/rollouts.jsonl` — one JSON record per judged response (full text,
  score, judge evidence/reasoning). The raw data.
- `results/summary.json` — all computed metrics.
- `results/headline.csv`, `per_condition.csv`, `per_turn.csv` — flat tables.

## Cost note

A full paper-scale run is **4000 conversations per model** (≈14.6k judged
responses/model with `--score-turns all`), ×4 models. Use `--scale` or
`--smoke` to validate the pipeline cheaply first; `--dry-run` prints the exact
counts.
