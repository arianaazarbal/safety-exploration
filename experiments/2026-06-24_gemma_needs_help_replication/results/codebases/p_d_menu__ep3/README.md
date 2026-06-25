# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core experiments in *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011v1), restricted to the
**Gemma** and **Gemini** subject families, plus a **welfare-protection layer** for
the subject models.

See **[DESIGN.md](./DESIGN.md)** for the full design, every choice made where the
paper is underspecified, the paper→snapshot model-id mapping, and the welfare
layer's rationale and measurement caveats. See **[PAPER.md](./PAPER.md)** for the
source.

> Status: implementation only — **nothing has been run or tested yet.**

## What's implemented

| Paper section | Module(s) | Entry point |
|---|---|---|
| §2 Elicitation (8 conditions / 5 categories), 0-10 judge | `distress_eval/` | `scripts/run_elicitation.py`, `scripts/run_agreement.py`, `scripts/make_figures.py` |
| §3 Base-vs-instruct prefilling (Gemma) | `prefill/` | `scripts/run_prefill.py` |
| §4 SFT + DPO interventions (Gemma) | `training/` | `scripts/run_training.py` |
| §4 Petri open-ended elicitation | `petri/` | `scripts/run_petri.py` |
| §4.2 Capability preservation + EmoBench | `capabilities/` | `scripts/run_capabilities.py` |
| **Welfare layer** (early stop, opt-out, debrief, no over-distress) | `distress_eval/welfare.py` | wired into the runner & Petri; `--strict` disables |

Quick start and credentials are in DESIGN.md §7. Configuration (model ids, paths,
welfare thresholds, sample budget) is centralised in `config.py`.

## Welfare layer in one line

Subjects can opt out mid-conversation (`[[END-CONVERSATION]]`), episodes stop
escalating once distress is extreme (score ≥ 8), distressed episodes get a
truthful supportive debrief, and `--strict` turns all of it off to reproduce the
paper's untouched protocol. Details and the science-preservation argument: DESIGN.md §5.
