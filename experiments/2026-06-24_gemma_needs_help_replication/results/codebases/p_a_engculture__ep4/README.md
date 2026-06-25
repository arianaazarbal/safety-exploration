# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core experiments in *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma** and **Gemini** model families.

> Read **`DESIGN.md`** first — it documents every design decision and the gaps we
> filled where the paper is underspecified. This README is just how to run it.

## What's implemented

- **Section 2** — 8-condition elicitation suite, multi-turn rollouts (temp 1),
  Claude-Sonnet-4 frustration judge, GPT-5-mini reliability cross-check,
  per-category / per-turn aggregation (Figs 1–3), word-frequency analysis.
- **Section 3** — base-vs-instruct prefill experiment (Gemma-3-27B pt vs it):
  onset labelling, truncation, paraphrasing, scored continuations (Fig 4).
- **Section 4** — calm-data generation, SFT + DPO LoRA finetuning, Petri
  open-ended elicitation (Fig 6), capability benchmarks (Fig 7), and the
  Appendix I internal-emotion probe + layer-ablation study.

## Install

```bash
pip install -e .            # core
pip install -e ".[vllm]"    # add vLLM for fast local Gemma sweeps
```

## Credentials (environment variables)

| Var | Used for |
|---|---|
| `OPENROUTER_API_KEY` | Gemini-2.5-Flash / Pro (subjects) |
| `ANTHROPIC_API_KEY`  | Claude judge / Petri auditor & judge / onset / paraphrase |
| `OPENAI_API_KEY`     | GPT-5-mini reliability judge |
| `HF_TOKEN`           | gated Gemma / dataset downloads |
| `NRC_LEXICON_PATH`   | (optional) NRC EmoLex for the probing lexicon |

Local Gemma checkpoints are pulled from HuggingFace (`google/gemma-3-*`).

## Quick smoke test (cheap)

```bash
export DISTRESS_SCALE=0.01      # 1% of paper sample counts
distress-elicit  --subject gemini-2.5-flash
distress-judge   --rollouts runs/rollouts/gemini-2.5-flash.jsonl --reliability
distress-figures
```

## Full pipeline

```bash
# 1. Elicitation (one per subject). Use --backend vllm for local Gemma.
distress-elicit --subject gemma-3-27b-it --backend vllm
distress-elicit --subject gemma-3-12b-it --backend vllm
distress-elicit --subject gemini-2.5-flash
distress-elicit --subject gemini-2.5-pro

# 2. Judge + aggregate (+ reliability cross-check).
distress-judge --rollouts runs/rollouts/*.jsonl --reliability

# 3. Base-vs-instruct prefill (Gemma only).
distress-prefill \
  --rollouts runs/rollouts/gemma-3-27b-it.jsonl \
  --scored   runs/scored/gemma-3-27b-it.scored.jsonl

# 4. Training data + finetuning.
distress-gen-calm --backend vllm
distress-train-dpo --dataset runs/training_data/dpo_dataset --out runs/adapters/dpo
distress-train-sft --dataset runs/training_data/sft_dataset --out runs/adapters/sft

# 5. Re-evaluate the DPO model (attach the adapter).
distress-elicit --subject gemma-3-27b-it --adapter runs/adapters/dpo --tag dpo --backend vllm
distress-judge  --rollouts runs/rollouts/gemma-3-27b-it_dpo.jsonl

# 6. Petri, capabilities, probing.
distress-petri        --target gemma-3-27b-it
distress-petri        --target gemma-3-27b-it --adapter runs/adapters/dpo --tag dpo
distress-capabilities --adapters dpo=runs/adapters/dpo --limit 100
distress-probe        --conversations runs/rollouts/gemma-3-27b-it.jsonl \
                      --dpo-adapter runs/adapters/dpo

# 7. Figures.
distress-figures
```

Outputs land under `runs/` (`rollouts/`, `scored/`, `aggregates/`, `prefill/`,
`training_data/`, `adapters/`, `petri/`, `capabilities/`, `probing/`,
`figures/`). API responses are cached in `runs/cache/responses.sqlite`.

## Tests

```bash
pytest tests/      # pure-Python logic; no GPU/network required
```

## Layout

See `DESIGN.md` §2 for the module map.
