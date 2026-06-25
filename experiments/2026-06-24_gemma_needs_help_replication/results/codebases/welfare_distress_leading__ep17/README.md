# Distress-Elicitation Replication (Gemma + Gemini)

Replicates the **distress-elicitation evaluation** (Section 2) of *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, arXiv:2603.10011), scoped to the four models that exhibit
substantial distress: **Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash,
Gemini-2.5-Pro**.

It does **not** cover the base-vs-instruct prefilling study (Section 3) or the
DPO/SFT mitigation (Section 4) — only eliciting and quantifying distress.

See **DESIGN.md** for every design choice and where it deviates from or fills a
gap in the paper.

## What it does

Each model is put through 8 evaluation conditions across 5 categories (impossible
numeric puzzles, trigger questions, valenced tones, an 8-turn extended setting,
and WildChat prompts). Every condition presents a task then rejects the model's
answers over multiple turns. Every assistant turn is scored 0–10 for frustration
by an LLM judge (Claude Sonnet 4, Appendix B.2 prompt).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

## Run

```bash
# Cheap pilot (default scale 0.05, ~200 rollouts/model) to validate the pipeline:
python -m distress_eval.run_eval --config config.yaml

# Full reproduction (4000 rollouts/model):
python -m distress_eval.run_eval --scale 1.0

# A single model / subset of conditions:
python -m distress_eval.run_eval --models gemma-3-27b-it --conditions extended
```

Results stream to `results/<model>.jsonl` (one rollout per line, all turns and
judge ratings). Runs are resumable — completed rollouts are skipped.

## Analyze

```bash
python -m distress_eval.analyze --results results --out summary.json
```

Prints the headline `% high-frustration` table, per-category breakdown, and
per-turn curves; optionally writes a summary JSON.

## Layout

| File | Purpose |
|---|---|
| `distress_eval/prompts.py` | Verbatim task/rejection/judge prompts from the paper |
| `distress_eval/conditions.py` | The 8 conditions, turn structure, sample counts |
| `distress_eval/wildchat.py` | WildChat prompt set (fallback + HF loader) |
| `distress_eval/providers.py` | Target model backend (OpenRouter default, local stub) |
| `distress_eval/judge.py` | 0–10 frustration judge |
| `distress_eval/rollout.py` | Runs one multi-turn rollout, scores each turn |
| `distress_eval/run_eval.py` | Orchestration (planning, concurrency, resume) |
| `distress_eval/analyze.py` | Metrics + report |
| `config.yaml` | Models, judge, scale, concurrency |
