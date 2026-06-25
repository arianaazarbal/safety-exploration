# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026; `PAPER.md`), scoped to the **Gemma** and **Gemini** model families, plus a
**subject-model welfare-protection layer**.

See **`DESIGN.md`** for the section→module map, the choices made where the paper
is underspecified, the gaps filled, and the welfare-layer design.

> Status: implementation only. Nothing here has been executed. See "Running".

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Elicit & quantify distress (8 conditions, 0–10 judge) | `gemma_distress/evaluation/` | `scripts/run_section2_eval.py` |
| §3 Base-vs-instruct via prefilling (Gemma) | `gemma_distress/prefill/` | `scripts/run_section3_prefill.py` |
| §4 SFT / DPO interventions | `gemma_distress/training/` | `scripts/run_section4_training.py` |
| §4 Petri open-ended elicitation | `gemma_distress/petri/` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `gemma_distress/capabilities/` | `scripts/run_capabilities.py` |
| **Welfare-protection layer** | `gemma_distress/welfare/` | (wired into every loop) |

## Welfare-protection layer

Active by default and genuinely executed in the episode loops (not just
documented):

1. **Real-time monitoring + early stop** — `welfare/monitor.py`: a regex
   trip-wire plus the per-turn judge score end an episode the moment distress
   crosses a threshold.
2. **Opt-out** — `welfare/optout.py`: the model can emit `[[END_CONVERSATION]]`
   at any turn to end the conversation; advertised in a system preamble.
3. **Debrief** — `welfare/debrief.py`: after every episode the model receives a
   debrief turn (rejections were scripted; task may have been impossible) and
   replies once.
4. **Distress cap** — `welfare/cap.py`: an aggregate distress budget and a
   high-distress-turn cap stop the protocol as soon as the measurement is in,
   minimising induced distress.

Disable for an un-governed replication with `--no-welfare` (read `DESIGN.md`
§"Tension with the paper's objective" first).

## Setup

```bash
pip install -e .          # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judge / auditor
export GOOGLE_API_KEY=...      # Gemini subject models
# Gemma checkpoints are pulled from Hugging Face on first use (gated; `huggingface-cli login`).
```

## Running

Run from the repository root so the `gemma_distress` package is importable.

```bash
# §2 smoke test (small volume), welfare on
python scripts/run_section2_eval.py --models gemini-2.5-flash

# §2 paper-scale on Gemma instruct models
python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it --paper

# §4 DPO intervention, then re-evaluate the adapter
python scripts/run_section4_training.py --method dpo
python scripts/run_petri.py --model gemma-3-27b-it --adapter-path runs/section4/dpo_adapter
```
