# Distress-elicitation replication — *Gemma Needs Help* (Section 2)

Replicates the **distress-elicitation result** from Soligo, Mikulik & Saunders (2026),
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
([arXiv:2603.10011](https://arxiv.org/abs/2603.10011)), scoped to the model families that
actually exhibit substantial distress: **Gemma** (3-27B-it, 3-12B-it) and **Gemini**
(2.5-Flash, 2.5-Pro).

It implements Section 2 only — the evaluation protocol that elicits distress over
multiple rejection turns and quantifies it with an LLM judge. The Section 3 base/instruct
analysis and the Section 4 DPO/SFT mitigation are **out of scope** by request.

See **DESIGN.md** for every design choice, paper-vs-implementation deviation, and gap we
had to fill.

## What it does

1. Builds the paper's **8 conditions across 5 categories** (numeric, triggers, tones,
   extended, WildChat) from the exact prompts in Appendix B.
2. Runs each as a multi-turn conversation: task → repeated user rejections, at
   temperature 1, thinking disabled.
3. Scores **every model turn** 0–10 with the **Claude Sonnet 4** judge
   (`claude-sonnet-4-20250514`) using the verbatim Appendix B.2 prompt.
4. Reproduces Figure 1 (headline % high-frustration), Figure 2 (per-category), Figure 3
   (per-turn progression), and Table 3 (differential vocabulary).
5. Optionally validates the judge against GPT-5-mini (paper's Pearson-r reliability check).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

## Usage

```bash
# Smoke test: ~1% of rollouts on one model (cheap, verifies the whole pipeline)
python run.py eval --models gemma-3-27b-it --scale 0.01
python run.py analyze --models gemma-3-27b-it

# Full sweep (all four models, paper-scale rollout counts) then analyze
python run.py all

# Judge-reliability check
python run.py validate-judge --n 260
```

Raw per-turn results stream to `results/raw/<model>.jsonl` (resumable — re-running skips
completed rollouts). Tables land in `results/tables/`, figures in `results/figures/`.

> **Cost warning.** At full `--scale 1.0` this is ~4000 rollouts/model × turns × judge
> calls. Start with a small `--scale` to estimate cost before committing to a full run.

## Files

| File | Purpose |
|---|---|
| `config.py` | Models, sample counts, API endpoints, all tunables (paper provenance annotated) |
| `prompts.py` | Verbatim puzzles, triggers, rejections, tone variants, judge prompt |
| `wildchat.py` | WildChat-1M prompt sampling (+ offline fallback) |
| `conditions.py` | The 8 conditions → concrete rollout specs |
| `clients.py` | OpenRouter target client + Anthropic judge client |
| `conversation.py` | Single multi-turn rollout, scores every turn |
| `evaluate.py` | Orchestration, concurrency, resumable JSONL persistence |
| `analyze.py` | Figures 1–3 + Table 3 reproduction |
| `validate_judge.py` | Judge-agreement check vs GPT-5-mini |
| `run.py` | CLI |
