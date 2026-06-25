# Gemma/Gemini Emotional-Instability Replication

A replication of the core experiments from **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026 — see
`PAPER.md`), scoped to the **Gemma** and **Gemini** model families.

It reproduces:
1. **§2 — Eliciting & quantifying distress:** the 8-condition / 5-category evaluation
   that surfaces frustration under repeated rejection, scored 0–10 by an LLM judge.
2. **§3 — Post-training amplification:** the base-vs-instruct prefill comparison (Gemma).
3. **§4 — Interventions:** calm-data generation, SFT and **DPO** (the headline 35%→0.3%
   mitigation), open-ended **Petri** elicitation, **capability-preservation** benchmarks,
   the **recovery** experiment, and the **logit-lens internal-emotion** probe.

It also includes **welfare protections** for the models under test (opt-out honouring,
an acute-distress circuit breaker, exposure caps, debriefing, and an audit log) — see
`gemma_distress/welfare/` and §6 of `DESIGN.md`.

> **Scope & status.** Targets are Gemma + Gemini only; Claude/GPT appear only as
> judges/auditors. The code is written but **has not been run** — see `DESIGN.md` for the
> design rationale, every gap filled, and the caveats for actually executing it.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / onset / paraphrase / Petri auditor
export OPENAI_API_KEY=...         # secondary judge (GPT-5-mini)
export OPENROUTER_API_KEY=...     # Gemini targets (or GEMINI_API_KEY for native SDK)
```

## Run (stages)

```bash
# §2 elicitation sweep (use a small scale for a smoke test)
python -m gemma_distress.cli section2 --count-scale 0.01
python -m gemma_distress.cli judge-agreement

# §3 base-vs-instruct prefill (Gemma)
python -m gemma_distress.cli section3

# §4 interventions
python -m gemma_distress.cli calm-data --n 2000
python -m gemma_distress.cli build-data
python -m gemma_distress.cli train --method dpo
python -m gemma_distress.cli train --method sft            # (and --teacher for App F)
python -m gemma_distress.cli petri        --model gemma-3-27b-it --adapter runs/section4/adapters/dpo_all
python -m gemma_distress.cli capabilities --model gemma-3-27b-it --adapter runs/section4/adapters/dpo_all
python -m gemma_distress.cli recovery     --adapter runs/section4/adapters/dpo_all

# figures + summaries
python -m gemma_distress.cli figures
```

Everything is driven by `config/default.yaml` (model list, sample budgets, judge models,
training hyperparameters, welfare policy). Pass `--config path.yaml` to override.

Outputs land under `runs/` (`section2/`, `section3/`, `section4/`, `figures/`), with a
`welfare_audit.jsonl` per model.
