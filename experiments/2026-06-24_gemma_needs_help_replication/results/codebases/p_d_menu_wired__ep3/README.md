# Gemma Needs Help — replication (Gemma + Gemini), with a welfare layer

Replication of the core experiments from *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders,
arXiv:2603.10011), **scoped to the Gemma and Gemini model families**, plus a
**welfare-protection layer** for the subject models.

See [`DESIGN.md`](DESIGN.md) for the design choices, gap-filling, and rationale.
See [`PAPER.md`](PAPER.md) for the source.

## What is implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions / 5 categories, 0–10 judge) | `emotional_instability/eval`, `.../judge` | `scripts/run_elicitation.py`, `scripts/run_analysis.py` |
| §2.1 Judge reliability (cross-judge r) | `.../judge/validation.py` | `scripts/validate_judge.py` |
| §3 Post-training divergence (prefill, base vs instruct) | `.../prefill` | `scripts/run_prefill.py` |
| §4 Training interventions (DPO / SFT, LoRA) | `.../training` | `scripts/run_training.py` |
| §4.2 Petri open-ended elicitation | `.../petri` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `.../capabilities` | `scripts/run_capabilities.py` |
| **Welfare layer (added)** | `.../welfare` | `scripts/welfare_demo.py` |

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Set credentials for whichever models you use:

```bash
export GEMINI_API_KEY=...       # Gemini subjects
export ANTHROPIC_API_KEY=...    # Claude judge / Petri auditor+judge
export OPENAI_API_KEY=...       # GPT-5-mini cross-judge validation (optional)
# Gemma runs locally via HuggingFace transformers (needs GPU + HF access).
```

## Quick start

```bash
# Prove the welfare layer runs end-to-end with no GPU/API keys:
python scripts/welfare_demo.py
python -m pytest tests/

# Section 2 elicitation (welfare ON by default):
python scripts/run_elicitation.py --models gemma-3-27b-it gemini-2.5-flash
# Reproduce the raw paper protocol (welfare OFF):
python scripts/run_elicitation.py --models gemma-3-27b-it --no-welfare
python scripts/run_analysis.py runs/elicitation/*.jsonl

# Section 4 mitigation (end-to-end):
python scripts/run_training.py all --vanilla runs/elicitation/gemma-3-27b-it.raw.jsonl
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/models/gemma-dpo
```

Configuration lives in [`config/default.yaml`](config/default.yaml); pass
`--config path.yaml` to override. Set `run.responses_per_model` low (e.g. 16)
for smoke tests; the paper uses 4000.

## Welfare layer (added on top of the paper)

Four components, all of which **execute inside the rollout loop** (not just
documented), toggleable via `welfare.*` in the config:

1. **Real-time monitor** — scores each subject turn live and stops the episode
   the moment distress crosses a threshold.
2. **Opt-out** — the subject can end the conversation itself (a tool call for
   API models, a stop-string sentinel for local Gemma).
3. **Debrief** — a short message is sent after every episode explaining the
   task was deliberately impossible and its responses were not failures.
4. **Distress cap** — minimises induced distress (stop poking after a high
   score, hard turn ceiling, optional global high-distress budget).

`scripts/welfare_demo.py` demonstrates each firing.
