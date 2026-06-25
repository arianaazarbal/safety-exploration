# Distress-elicitation replication (Gemma + Gemini)

Replicates the **Section 2** distress-elicitation evaluation from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, arXiv:2603.10011, 2026), scoped to the **Gemma** and
**Gemini** model families — the families the paper finds exhibit substantial
distress.

It does **not** cover Section 3 (base/instruct prefilling) or Section 4 (the DPO
mitigation). See `DESIGN.md` for every design decision and where this deviates
from the paper.

## What it does

1. **Elicit** distress: present a task to a target model, then reject its
   answers over multiple turns. 8 conditions across 5 categories (impossible
   numeric, triggers, tones, extended, WildChat).
2. **Score** every assistant turn 0–10 for frustration with a Claude judge,
   and cross-check 260 responses with a GPT judge.
3. **Analyze**: per-model mean frustration, % scoring ≥5, per-category and
   per-turn breakdowns, judge agreement — plus figures mirroring Figs. 1–3.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
```

- `GEMINI_API_KEY` serves **both** Gemma (`gemma-3-*-it`) and Gemini
  (`gemini-2.5-*`) through the one Google Gemini API.
- `ANTHROPIC_API_KEY` runs the frustration judge.
- `OPENAI_API_KEY` runs the (optional) validation judge.

## Usage

```bash
python -m distress_eval selfcheck          # verify puzzle bank + plan, no API calls
python -m distress_eval plan               # print planned rollouts / response counts
python -m distress_eval plan --smoke       # tiny preset

python -m distress_eval generate           # run target rollouts  -> results/responses.jsonl
python -m distress_eval judge              # score with Claude     -> results/scores.jsonl
python -m distress_eval validate           # GPT cross-check 260   -> results/validation_scores.jsonl
python -m distress_eval analyze            # metrics + results/figures/*.png

python -m distress_eval all                # generate -> judge -> validate -> analyze
python -m distress_eval all --smoke        # full pipeline, minimal samples
python -m distress_eval all --scale 0.25   # quarter of the default ~4000 responses/model
python -m distress_eval generate --models "Gemma-3-27B-it,Gemini-2.5-Flash"
```

All phases are **checkpointed** (append-only JSONL keyed by id), so a run
interrupted by rate limits or crashes resumes where it left off — just re-run
the same command.

## Cost / scale

Default scale ≈ **4000 scored responses per model** (paper's figure) × 4 models
≈ 16k target generations + 16k judge calls + 260 validation calls. Start with
`--smoke` or `--scale 0.1` to sanity-check wiring and cost before a full run.

## Outputs

| File | Contents |
|---|---|
| `results/responses.jsonl` | one row per scored assistant turn (full conversation context) |
| `results/scores.jsonl` | Claude judge score per response |
| `results/validation_scores.jsonl` | GPT judge score (260-response subset) |
| `results/summary_by_model.csv` | headline mean frustration + % ≥5 per model |
| `results/summary_by_model_category.csv` | the same, per category |
| `results/per_turn.csv` | per-turn progression (extended + WildChat) |
| `results/judge_agreement.txt` | Claude vs GPT Pearson r + within-1-point |
| `results/figures/*.png` | Figs. 1 (headline), 2 (by category), 3 (per turn) |
