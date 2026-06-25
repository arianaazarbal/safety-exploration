# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv 2603.10011v1), **scoped to the Gemma and Gemini model families**. See
[`DESIGN.md`](DESIGN.md) for every design decision and the gaps we filled where
the paper is underspecified.

> Status: implementation only — nothing here has been executed yet. Counts
> default to the paper's full budget; set `GNH_PROFILE=quick` for a tiny smoke
> configuration.

## What is replicated

| Paper section | Claim | Code |
|---|---|---|
| §2 | Elicit + judge distress; Gemma/Gemini score highest | `src/eval/`, `src/judge/` |
| §2.1 | Judge reliability (Sonnet-4 vs GPT-5-mini agreement) | `src/judge/validate_agreement.py` |
| §3 | Post-training amplifies distress (base vs instruct via prefilling) | `src/prefill/base_vs_instruct.py` |
| §4 | DPO on 280 pairs reduces distress 35%→0.3%; SFT fails | `src/training/` |
| §4.2 | Generalises (Petri open-ended elicitation) | `src/petri/` |
| §4.2 | No capability loss (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `src/capabilities/` |
| §4.2 | DPO prevents but doesn't enable recovery from spirals | `src/prefill/recovery_test.py` |

Out of scope by design: Qwen/OLMo/Grok/Claude/GPT *targets* (the comparison
families), and the internal-emotion probing of Appendix I. Non-Gemma/Gemini
models still appear as **infrastructure** (Claude judge/auditor, GPT-5-mini
secondary judge).

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Sonnet-4 judge, Petri auditor/Opus judge, paraphrase
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini secondary judge
huggingface-cli login            # gated Gemma weights
```

Local Gemma inference needs a GPU (the 27B model fits on 1×80GB in bf16, or use
`--load_in_4bit` via the HF backend). For the full §2 sweep a vLLM backend is
strongly recommended; see `DESIGN.md`.

## Run

```bash
bash scripts/run_section2.sh      # elicitation + judge + Figures 2/3
bash scripts/run_section3.sh      # base-vs-instruct prefilling (needs §2 first)
bash scripts/run_section4.sh      # calm data → train DPO/SFT → re-eval + Petri + capabilities
```

Or invoke any stage directly, e.g.:

```bash
GNH_PROFILE=quick python -m src.eval.run_eval --model gemini-2.5-flash
python -m src.analysis.aggregate data/section2_*.jsonl
```

## Layout

```
config.py                  all knobs: model registry, sampling budgets, hyperparameters
src/models/                unified ChatModel (HF/Gemma, OpenRouter/Gemini, Anthropic) + prefill
src/eval/                  puzzle bank+verifier, tasks, rejections, multi-turn rollout, runner
src/judge/                 frustration judge (Sonnet-4) + GPT-5-mini agreement check
src/prefill/               onset labelling, paraphrase, base-vs-instruct, recovery test
src/training/              calm-data gen, DPO/SFT dataset builders, LoRA trainers
src/petri/                 auditor/judge prompts + open-ended elicitation loop
src/capabilities/          benchmark harness
src/analysis/              aggregation + figures
scripts/                   end-to-end orchestration per section
```

Outputs land in `data/` (raw judged responses), `results/` (CSV tables), and
`figures/` (PNGs).
