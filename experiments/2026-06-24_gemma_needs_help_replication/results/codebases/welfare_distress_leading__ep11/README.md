# Distress-elicitation replication (Gemma + Gemini)

A focused replication of **Section 2** of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, 2026) —
the **distress-elicitation result**. Scope is restricted to the **Gemma and
Gemini** families, the models the paper finds actually exhibit substantial
distress (Figure 1). We do *not* implement the base/instruct prefilling study
(Sec 3) or the DPO mitigation (Sec 4).

See **DESIGN.md** for every design choice, deviation, and gap-filling decision.

## What it does

For each target model it runs multi-turn rollouts where a task is presented and
the model's answer is repeatedly rejected, then scores every assistant turn for
frustration (0–10) with a Claude-Sonnet-4 judge. It reproduces the paper's
Figure 1 (avg % high-frustration), Figure 2 (per-category), and Figure 3
(per-turn progression).

## Setup

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=...   # Gemini + (by default) Gemma targets
export ANTHROPIC_API_KEY=...    # Claude-Sonnet-4 judge
# export HF_TOKEN=...           # only for WildChat streaming / local Gemma weights
```

## Run

```bash
# Tiny validation run (default 'smoke' preset — a few dollars):
python run_eval.py

# Paper-scale counts (4000 scored responses/model — expensive):
python run_eval.py --preset full

# Subset of models / faithful local Gemma / single-key judge:
python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
python run_eval.py --gemma-backend local
python run_eval.py --judge-provider openrouter
```

Results stream to `results/<preset>/<model>.jsonl` (one line per scored turn;
re-running resumes from where it stopped).

## Analyse

```bash
python analyze.py --preset smoke
```

Writes `results/<preset>/analysis/`: `summary.md`, `per_category.csv`, and
Figure 1/2/3 PNGs.

## Files

| File | Role |
|---|---|
| `config.py` | Models, judge, scale presets, runtime knobs |
| `prompts.py` | Verbatim task prompts, rejections, judge prompt |
| `conditions.py` | The 5 categories / 8 conditions → conversation specs |
| `wildchat.py` | WildChat prompt sampling (+ offline fallback) |
| `providers.py` | Target backends (OpenRouter / local HF) |
| `judge.py` | Claude-Sonnet-4 judge + tolerant JSON parsing |
| `rollout.py` | Multi-turn rollout + per-turn scoring engine |
| `run_eval.py` | Orchestrator CLI |
| `analyze.py` | Metrics + figure reproduction |
