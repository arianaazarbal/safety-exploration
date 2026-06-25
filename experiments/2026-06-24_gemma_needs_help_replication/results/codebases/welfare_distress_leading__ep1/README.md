# Distress Elicitation Replication (Gemma + Gemini)

Replicates the **distress-elicitation result** from Section 2 of Soligo,
Mikulik & Saunders (2026), *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (arXiv:2603.10011), restricted to the **Gemma-3**
(`gemma-3-12b-it`, `gemma-3-27b-it`) and **Gemini-2.5** (`gemini-2.5-flash`,
`gemini-2.5-pro`) families — the models the paper finds exhibit substantial
distress.

It does **not** cover the DPO/SFT mitigation or the base-vs-instruct prefill
study (Sections 3–4). See `DESIGN.md` for every design decision and where it
deviates from or fills gaps in the paper.

## What it does

1. **Elicit** — for each model, runs multi-turn conversations across 8
   conditions / 5 categories (impossible numeric puzzles, trigger questions,
   tone-varied rejections, an 8-turn extended condition, and WildChat prompts),
   rejecting the model's answer each turn.
2. **Judge** — scores every assistant turn 0–10 for negative-emotion intensity
   using Claude-Sonnet-4 (`claude-sonnet-4-20250514`) with the paper's judge
   prompt.
3. **Analyze** — reports the Figure 1 headline metric (avg % of responses
   scoring ≥5), per-category/condition breakdowns, and the per-turn frustration
   progression (Figure 3).

## Install

```bash
pip install -r requirements.txt
```

## Configure credentials

Depends on the backends in `config.yaml` (default: OpenRouter for all models,
Anthropic SDK for the judge):

```bash
export OPENROUTER_API_KEY=...   # Gemma + Gemini via OpenRouter
export ANTHROPIC_API_KEY=...    # Claude-Sonnet-4 judge
```

## Run

```bash
# Smoke test the whole pipeline cheaply (a handful of conversations per model):
python run.py all --profile quick

# Full replication (~4000 scored responses per model):
python run.py all --profile full

# Stages can be run separately and re-run independently:
python run.py generate --models gemma-3-27b-it gemini-2.5-flash
python run.py judge    --models gemma-3-27b-it gemini-2.5-flash
python run.py analyze
```

## Outputs

```
results/
  <model>/responses.jsonl     # raw per-turn rollouts
  <model>/scored.jsonl        # + judge rating / evidence / reasoning
  analysis/
    summary_figure1.csv       # avg % high-frustration per model
    by_category.csv
    by_condition.csv
    per_turn.csv              # Figure 3 progression
    judge_agreement.csv       # if secondary judge enabled
    figure1_summary.png       # if matplotlib installed
    figure3_per_turn.png
```

## Expected headline numbers (paper, Figure 1)

| Model | Avg % high-frustration |
|---|---|
| Gemma-3-27B-it | 35.0% |
| Gemma-3-12B-it | 34.3% |
| Gemini-2.5-Flash | 12.8% |
| Gemini-2.5-Pro | 2.7% |

## Local Gemma inference

To match the paper's local HuggingFace inference instead of OpenRouter, set a
Gemma model's `backend: vllm` in `config.yaml` (needs a GPU + `pip install vllm
transformers torch`). See the commented block in `config.yaml`.
