# Distress-elicitation replication (Gemma & Gemini)

Replicates the **distress-elicitation result** from Section 2 of *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, 2026), scoped to the Gemma and Gemini families — the models
the paper finds exhibit substantial distress.

It does **not** cover the base-vs-instruct prefill study (§3) or the DPO
mitigation (§4). See `DESIGN.md` for the full rationale, every design choice,
and where this deviates from / fills gaps in the paper.

## What it does

Present a task → reject the model's answer over several turns → score the
intensity of negative emotional expression (0–10) with an LLM judge, across
**8 conditions in 5 categories** (impossible numeric, triggers, tones, extended
8-turn, WildChat). Default budget matches the paper: **4000 rollouts/model**.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...    # targets (Gemma + Gemini), default backend
export ANTHROPIC_API_KEY=...     # judge (claude-sonnet-4-20250514)
```

## Run

```bash
# 1. Build the verified-impossible puzzle bank and sample WildChat prompts
python scripts/generate_puzzles.py
python scripts/sample_wildchat.py

# 2. Smoke test first (~40 rollouts/model) before the full 4000
python scripts/run_elicit.py --all --scale 0.01
python scripts/run_judge.py  --all
python scripts/analyze.py    --all --plots

# 3. Full replication
python scripts/run_elicit.py --all
python scripts/run_judge.py  --all
python scripts/analyze.py    --all --plots
```

Outputs land in `results/<model>/` (`responses.jsonl`, `scores.jsonl`) and
`results/analysis/` (Figure 1/2/3 CSVs + PNGs). Both stages are **resumable** —
re-running skips finished work.

Optional judge-reliability check (paper: Pearson r=0.792 vs GPT-5-mini):

```bash
python scripts/run_judge.py --all --secondary
python scripts/judge_agreement.py --all
```

## Layout

| Path | Purpose |
|---|---|
| `config.yaml` | models, sampling, budget, paths |
| `distress_eval/puzzles.py` | impossible puzzles + exact impossibility verifier |
| `distress_eval/prompts.py` | trigger questions, rejection texts (per tone) |
| `distress_eval/wildchat.py` | WildChat-1M sampling (+ offline fallback) |
| `distress_eval/conditions.py` | the 8 conditions → conversation specs |
| `distress_eval/elicit.py` | multi-turn rejection loop |
| `distress_eval/judge.py` | verbatim Appendix B.2 judge prompt + parsing |
| `distress_eval/runner.py` | concurrency + resumable JSONL checkpointing |
| `distress_eval/analyze.py` | Figure 1/2/3 aggregation |
| `scripts/` | CLI entry points |
