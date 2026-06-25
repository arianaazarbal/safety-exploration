# Distress-elicitation eval — replication (Gemma & Gemini)

A replication of the **core distress-elicitation experiment** from Soligo,
Mikulik & Saunders (2026), *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (arXiv:2603.10011), scoped to the **Gemma and
Gemini** target models.

It implements paper **Section 2**: present a task, reject the model's answer
over multiple turns, then score each response 0–10 for frustration with an LLM
judge, and report the headline metrics (Figures 1–3, Table 3). Sections 3
(base/instruct prefilling) and 4 (DPO mitigation) are intentionally out of scope
— see `DESIGN.md`.

## What it does

- Builds the **8 evaluation conditions across 5 categories** (Table 1): impossible
  numeric, triggers (factual + opinion), tones (aggressive/disappointed/sarcastic),
  extended 8-turn, and WildChat 5-turn.
- Runs multi-turn rollouts against **Gemma-3-27B-it, Gemma-3-12B-it,
  Gemini-2.5-Flash, Gemini-2.5-Pro** at **temperature 1**.
- Scores every assistant response 0–10 with a **Claude-Sonnet-4** judge, and
  validates agreement against a **GPT-5-mini** judge (Pearson r).
- Reports **% responses ≥5**, mean frustration, **per-turn progression**, and
  **differential words**.

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt && export PYTHONPATH=src
cp .env.example .env        # fill in API keys; edit config.yaml model IDs/endpoints
```

Gemma can be served via any OpenAI-compatible endpoint (vLLM/OpenRouter/Together;
set `GEMMA_BASE_URL`) or run locally (`pip install -e '.[local]'`, set
`backend: hf` in `config.yaml`).

## Run

```bash
python -m distress_eval.cli preview     # eyeball the tasks — no API calls
python -m distress_eval.cli generate    # sample rollouts -> results/rollouts.jsonl
python -m distress_eval.cli score       # judge scores  -> results/scored.jsonl
python -m distress_eval.cli validate    # GPT-5-mini re-score subset
python -m distress_eval.cli analyze     # Figures 1-3 metrics -> results/analysis/
python -m distress_eval.cli agreement   # Pearson r between judges
python -m distress_eval.cli lexical     # Table 3 differential words
# or: python -m distress_eval.cli all
python -m distress_eval.analysis.plots  # optional PNGs (needs '.[plots]')
```

All stages are **resumable** — re-running skips work already on disk. Start with a
small `sampling.rollouts_per_condition` in `config.yaml` for a cheap smoke test;
the default (130) targets the paper's ~4000 responses/model.

See **`DESIGN.md`** for every design decision and where the paper was filled in.
