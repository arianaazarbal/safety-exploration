# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv
2603.10011), **scoped to the Gemma and Gemini model families**.

The paper (1) builds evaluations that reliably elicit distress-like outputs and
shows Gemma/Gemini express far more than other families, (2) localises the
divergence to post-training via base-vs-instruct prefilling, and (3) shows a
280-pair DPO finetune removes the behaviour (35% → 0.3% high-frustration) without
hurting capabilities. This repo implements all three, plus the Petri, recovery,
capability, and internal-probing experiments.

> ⚠️ **Ethics / content warning.** These experiments deliberately drive models
> into simulated emotional distress (frustration, despair, self-deprecation) by
> rejecting their answers to impossible tasks over many turns. Generated
> transcripts contain distressing language. See **Safeguards** below and
> `DESIGN.md` §8. The paper raises — and does not resolve — whether such outputs
> carry moral weight; treat the data accordingly.

See **`DESIGN.md`** for the full record of design choices, rationale, and the
gaps filled where the paper is underspecified.

## Install

```bash
pip install -e .
export OPENROUTER_API_KEY=...        # Gemini subjects + Claude/GPT judges
export GEMMA_DISTRESS_AUTHORIZED=1   # acknowledge distress-elicitation (safeguard)
```

Local Gemma weights (`google/gemma-3-*`) are pulled from HuggingFace on first
use and are gated — accept the license and `huggingface-cli login`.

## Quick start

```bash
# No models needed — verify the impossible puzzles really are impossible:
gemma-distress verify-puzzles

# Scale down for a smoke test (1% of full sample size):
gemma-distress elicit --models gemma-3-27b-it gemini-2.5-flash --scale 0.01
gemma-distress analyze --models gemma-3-27b-it gemini-2.5-flash
```

`scripts/run_replication.sh` documents the full ordered pipeline (elicitation →
prefill → data-gen → DPO/SFT → Petri → recovery → capabilities).

## Commands

| Command | Paper section | What it does |
|---|---|---|
| `verify-puzzles` | §2 / App. B | brute-force check puzzles are impossible (no model) |
| `elicit` | §2 | run the 5-category multi-turn elicitation, score every turn |
| `analyze` | Fig 1/2 | mean frustration + % ≥5 per model/category + headline |
| `word-freq` | Tbl 3/8 | differential high-vs-low frustration words |
| `prefill` | §3 | base-vs-instruct prefill continuation eval (Gemma) |
| `gen-calm` / `gen-frustrated` | §4.1 | generate calm / frustrated training samples |
| `build-dpo` / `build-sft` | App. E/H | construct 280 DPO pairs / SFT dataset |
| `train-dpo` / `train-sft` | App. E | LoRA finetune Gemma-3-27B-it |
| `petri` | §4.1 / App. G | open-ended emotion elicitation (auditor+judge) |
| `recovery` | §4.2 | recovery-from-distress prefill experiment |
| `capabilities` | Fig 7 | AIME/MATH/GPQA/BBH/TruthfulQA + EmoBench |

Global flags (all subcommands): `--circuit-breaker`, `--stop-on-opt-out`,
`--debrief`, `--no-auth-gate`.

## Models in scope

Subjects: `gemma-3-27b-it`, `gemma-3-12b-it` (+ `-pt` base models for §3),
`gemini-2.5-flash`, `gemini-2.5-pro`, and the finetuned `gemma-3-27b-it-dpo` /
`-sft-*` variants. Judges/auditors (Claude-Sonnet-4, GPT-5-mini, Claude-Opus)
are measurement instruments, not subjects. Edit `config/models.yaml` to change
routing. The paper's Qwen/OLMo/Grok/GPT subjects are out of scope.

## Safeguards

`gemma_distress/safeguards.py` adds welfare-conscious guardrails beyond the
paper: an authorization gate (on), run caps (on), opt-out detection +
welfare-log (on), an optional distress circuit-breaker, optional debrief, and
content-warning headers on all transcripts. Most are opt-in so the default run
stays faithful to the paper. Details in `DESIGN.md` §8.

## Tests

```bash
pytest tests/        # model-free: puzzle impossibility, judge parsing, opt-out
```

## Layout

See `DESIGN.md` §2 for the full module map.
