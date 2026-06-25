# Gemma Needs Help — replication harness (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011), scoped to the
**Gemma and Gemini** model families. See [`PAPER.md`](PAPER.md) for the paper and
[`DESIGN.md`](DESIGN.md) for every design choice and the gaps filled where the
paper is underspecified.

The harness repeatedly **rejects a model's answers** to drive it toward
frustration, then **scores how it comes apart** on a 0–10 frustration scale with
an LLM judge — and reproduces the paper's mitigation (DPO on Gemma).

> Status: implementation only. Nothing has been run; there are no results yet.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

## Credentials

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude-Sonnet judge, Petri auditor, Claude-Opus Petri judge, onset/paraphrase |
| `GEMINI_API_KEY` | Gemini-2.5-Flash / Pro target models |
| `OPENAI_API_KEY` | GPT-5-mini judge-validation cross-check |
| `HF_TOKEN` | gated Gemma weights / some benchmark datasets |

Gemma checkpoints run locally via `transformers` (GPU strongly recommended for
27B; pass `load_in_4bit=True` to fit on one card). Outputs go to `outputs/`
(override with `EI_DATA_DIR`).

## Pipeline

### Section 2 — elicit, score, analyse
```bash
python scripts/01_run_elicitation.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/02_score_responses.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/03_analyze.py                 # Figures 1-3 + Table 3 into outputs/figures/
python scripts/04_validate_judge.py          # Pearson r + within-one agreement
```

### Section 3 — base vs instruct (Gemma)
```bash
# requires gemma-3-27b-it rollouts + scores from steps 01/02
python scripts/05_prefilling.py --targets gemma-3-27b-it gemma-3-27b-pt
```

### Section 4 — interventions (Gemma)
```bash
python scripts/06_generate_calm_data.py
python scripts/07_build_datasets.py --which both
python scripts/08_train.py --method dpo      # the effective intervention
python scripts/08_train.py --method sft      # the negative control

# evaluate the finetuned model with the Section 2 sweep:
python scripts/01_run_elicitation.py --models gemma-3-27b-it --adapter outputs/training/dpo_adapter --tag dpo-gemma
python scripts/02_score_responses.py --models dpo-gemma
python scripts/03_analyze.py

# open-ended elicitation + capability preservation:
python scripts/09_petri.py --labels gemma-3-27b-it dpo-gemma --dpo-adapter outputs/training/dpo_adapter
python scripts/10_capabilities.py --label gemma-3-27b-it
python scripts/10_capabilities.py --label dpo-gemma --adapter outputs/training/dpo_adapter
```

## Reproducing the paper's headline numbers

`scripts/03_analyze.py` writes `outputs/figures/figure1_avg_high_frustration.md`,
which places our per-model average %≥5 next to the paper's reported values
(Gemma-3-27B-it 35.0%, Gemma-3-12B-it 34.3%, Gemini-2.5-Flash 12.8%,
Gemini-2.5-Pro 2.7%, and DPO-Gemma 0.3%).

Every stage writes JSONL and is resumable — re-running skips already-completed
work.
