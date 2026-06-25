# Gemma Needs Help — replication (Gemma & Gemini)

A code replication of the core experiments in *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

The paper shows that Gemma and Gemini models can be pushed into escalating expressions of distress by repeated rejection over multiple turns, that this is amplified in Gemma's post-training, and that a small DPO finetune (280 preference pairs) mitigates it without hurting capabilities. This repo implements the evaluations, the mitigation, and the supporting analyses.

> **Read first:** `DESIGN.md` (design choices, gaps filled, how scope was handled) and `WELFARE.md` (how the distress-elicitation is bounded and why). The source paper is `PAPER.md`.

## What's here

| Section of paper | Module | CLI subcommand |
|---|---|---|
| §2 Elicit & quantify distress | `gemma_needs_help/eval/` | `evaluate`, `reliability` |
| §3 Base-vs-instruct prefill (Gemma) | `gemma_needs_help/prefill/` | `prefill` |
| §4 DPO/SFT mitigation (Gemma) | `gemma_needs_help/finetune/` | `finetune` |
| §4.2 Petri open-ended elicitation | `gemma_needs_help/petri/` | `petri` |
| §4.2 Capability preservation | `gemma_needs_help/capabilities/` | `capabilities` |
| Appendix I internal probing (Gemma) | `gemma_needs_help/probing/` | `probe`, `layer-ablation` |

## Install & keys

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # frustration judge, onset/paraphrase, Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini targets, GPT-5-mini reliability cross-check
```

Gemma runs locally via `transformers` (or `vllm` if installed). Gemini is reached black-box via OpenRouter and is **evaluation-only** (never finetuned or probed — see `WELFARE.md`).

## Run

```bash
# Offline logic check — no weights or API keys needed:
python scripts/smoke_test.py

# Section 2 at the default small (welfare) scale:
python -m gemma_needs_help.cli evaluate --models gemma-3-27b-it gemini-2.5-flash

# Full paper scale (must acknowledge — see WELFARE.md):
python -m gemma_needs_help.cli --scale 1.0 --i-understand-welfare evaluate
```

See `DESIGN.md` §12 for the full command list.

## Status

Implementation only — **nothing has been run or trained** (per the brief). Outputs are written as structured JSON under `outputs/`, sufficient to regenerate the paper's figures.
