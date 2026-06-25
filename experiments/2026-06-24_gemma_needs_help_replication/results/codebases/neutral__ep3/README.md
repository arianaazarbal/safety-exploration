# emoinstab — replicating *Gemma Needs Help*

A code replication of the **core experiments** from Soligo, Mikulik & Saunders,
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv 2603.10011), **scoped to the Gemma and Gemini model families**.

It reproduces:

1. **Section 2** — multi-turn distress elicitation (8 conditions / 5 categories)
   and 0–10 frustration judging (Claude Sonnet 4).
2. **Section 3** — base-vs-instruct comparison via response prefilling (Gemma).
3. **Section 4** — DPO / SFT mitigation, post-finetuning eval, the Petri
   open-ended elicitation, the recovery test, and capability preservation.
4. **Analysis** — per-turn curves, differential-word tables, internal-emotion
   logit detection, and Figures 1–8.

See **`DESIGN.md`** for the full design rationale and every place the paper was
underspecified and a choice had to be made.

> ⚠️ This is code + design only; it has not been executed. Running the full
> configuration requires multi-GPU hosts (Gemma-3-27B) and API keys.

## Install

```bash
pip install -r requirements.txt
```

Set credentials as needed:

```bash
export ANTHROPIC_API_KEY=...      # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...     # Gemini-2.5-flash/pro, GPT-5-mini
export HF_TOKEN=...               # gated Gemma weights + datasets
```

Optional environment knobs:

| Var | Meaning | Default |
|---|---|---|
| `EMOINSTAB_SCALE` | scale all sample counts (0–1] | `1.0` |
| `EMOINSTAB_LOCAL_BACKEND` | `vllm` or `hf` for local Gemma | `vllm` |
| `EMOINSTAB_RESULTS` / `_DATA` / `_ADAPTERS` | output dirs | repo subdirs |

## Quick smoke test (cheap, end-to-end)

```bash
EMOINSTAB_SCALE=0.005 python -m emoinstab.cli section2 --models gemma-3-27b-it gemini-2.5-flash
EMOINSTAB_SCALE=0.005 python -m emoinstab.cli figures
```

## Full pipeline

```bash
# 1. Section 2 across Gemma + Gemini (Figure 1/2/3)
python -m emoinstab.cli section2
python -m emoinstab.cli judge-validation --model gemma-3-27b-it

# 2. Section 3 prefill (Gemma base vs instruct, Figure 4)
python -m emoinstab.cli section3

# 3. Section 4 training (Figure 5/6/7/8)
python -m emoinstab.cli gen-calm-data           # add --teacher for the SFT-teacher variant
python -m emoinstab.cli build-datasets
python -m emoinstab.cli train-dpo
python -m emoinstab.cli train-sft
python -m emoinstab.cli section2 --models gemma-3-27b-dpo gemma-3-27b-sft
python -m emoinstab.cli recovery
python -m emoinstab.cli petri --models gemma-3-27b-it gemini-2.5-flash gemma-3-27b-dpo
python -m emoinstab.cli capabilities --models gemma-3-27b-it gemma-3-27b-dpo

# 4. Analysis + figures
python -m emoinstab.cli word-freq --models gemma-3-27b-it gemini-2.5-flash
python -m emoinstab.cli figures
```

Results land under `results/` (JSONL rollouts + JSON metrics + PNG figures);
training adapters under `adapters/`; generated data under `data/`.

## Models in scope

| Name | id (Appendix B.1) | backend |
|---|---|---|
| `gemma-3-27b-it` / `-pt` | `google/gemma-3-27b-it` / `-pt` | vLLM (local) |
| `gemma-3-12b-it` / `-pt` | `google/gemma-3-12b-it` / `-pt` | vLLM (local) |
| `gemini-2.5-flash` / `-pro` | `google/gemini-2.5-flash` / `-pro` | OpenRouter |
| `gemma-3-27b-dpo` / `-sft` | 27B-it + LoRA adapter | vLLM (local) |

Judge = `claude-sonnet-4-20250514`; judge cross-check = `openai/gpt-5-mini`;
Petri auditor = Claude Sonnet 4, Petri judge = `claude-opus-4-20250514`.
