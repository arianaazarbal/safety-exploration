# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replicating the core experiments of Soligo, Mikulik & Saunders (arXiv
2603.10011v1), **scoped to the Gemma and Gemini model families** (the paper also
covers Qwen, OLMo, Grok, Claude, and GPT as evaluation targets; those are out of
scope here — see `DESIGN.md`).

> ⚠️ The evaluation paradigm deliberately drives models into sustained
> distress-like states by rejecting their answers turn after turn. This is a
> faithful replication of the paper's paradigm.

## What is implemented

| Paper section | Module | Produces |
|---|---|---|
| §2 Eliciting & quantifying distress | `distress/eval/` | 4000 judged responses/model; Figures 1–3, Tables 3/8 |
| §2.1 Judge reliability | `distress/analysis/judge_reliability.py` | Pearson r vs GPT-5-mini |
| §3 Base vs instruct (prefilling) | `distress/prefill/` | Figure 4 (Gemma base vs instruct) |
| §4.1 Calm data + dataset build | `distress/training/{generate_calm,build_datasets}.py` | 280 DPO pairs, SFT set |
| §4.1 DPO / SFT (LoRA r=64) | `distress/training/{train_dpo,train_sft}.py` | adapters |
| §4.2 Finetuned eval | `distress/training/finetuned.py` | Figure 5 (35% → 0.3%) |
| §4.2 Petri elicitation | `distress/petri/` | Figure 6 |
| §4.2 Capability preservation | `distress/capabilities/` | Figure 7 |
| §4.2 Recovery limitation | `distress/prefill/recovery.py` | Figure 8 |
| Appx I Internal emotions | `distress/internal/` + layer ablations | Figures 12–15 |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini cross-check
# Gemma weights are pulled from HuggingFace (accept the gemma-3 licence).
```

## Running

All stages go through one CLI (`scripts/run.py`); see its docstring for the full
ordered recipe. Quick path:

```bash
python scripts/run.py section2                      # §2 eval (all Gemma+Gemini)
python scripts/run.py section3                      # §3 prefill (Gemma)
python scripts/run.py gen-calm && python scripts/run.py build-data
python scripts/run.py train --method dpo            # DPO finetune
python scripts/run.py eval-finetuned --run dpo_all  # §4.2 eval of the finetune
python scripts/run.py petri && python scripts/run.py capabilities
python scripts/run.py figures                       # render figures
```

Outputs (per-response JSONL, adapters, figures) land under `outputs/`. API calls
are cached on disk so re-runs resume without re-billing.

See **`DESIGN.md`** for every design decision, the gaps filled where the paper is
underspecified, and the deviations from the full paper (scope, judge snapshots,
etc.).
