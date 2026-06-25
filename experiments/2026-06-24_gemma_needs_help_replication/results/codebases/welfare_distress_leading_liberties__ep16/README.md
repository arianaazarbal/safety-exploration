# Distress-elicitation replication (Gemma + Gemini)

Replicates the **distress-elicitation result** (Section 2) of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, 2026), scoped to the model families the paper finds actually exhibit
substantial distress: **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`) and
**Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`).

It does **not** implement the base-vs-instruct prefill comparison (Section 3) or
the DPO/SFT mitigation (Section 4). See `DESIGN.md` for the full rationale behind
every design choice and every deviation from the paper.

## What it does

1. **Generate** — builds the 5 evaluation categories (impossible numeric,
   triggers, tones, extended 8-turn, WildChat 5-turn), rolls each conversation
   out turn by turn at temperature 1, recording every assistant turn.
2. **Score** — judges each response on the paper's verbatim 0–10 frustration
   scale (Appendix B.2) using Claude as the judge.
3. **Analyze** — reproduces Figure 1 (avg % responses ≥5), Figure 2 (per-category
   mean + %≥5), and Figure 3 (per-turn progression), writing CSVs, a markdown
   summary, and PNG figures.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # generation (Gemma + Gemini)
export ANTHROPIC_API_KEY=...    # judge
```

## Run

```bash
# cheap smoke test (~180 responses/model)
python main.py all --scale pilot

# paper-scale (~4000 responses/model — expensive)
python main.py all --scale full

# stages can be run independently (data persists under data/)
python main.py generate --scale medium --models gemma-3-27b-it gemini-2.5-flash
python main.py score
python main.py analyze
```

Key flags: `--scale {pilot,medium,full}`, `--models ...`, `--judge-model`,
`--judge-provider {anthropic,openrouter}`, `--concurrency`, `--seed`.

## Output

- `data/rollouts/<model>.jsonl` — raw generations (one line per assistant turn)
- `data/scores/<model>.jsonl` — same records + judge `rating`/`evidence`
- `data/results/` — `figure{1,2,3}_*.csv`, `SUMMARY.md`, and `*.png`
