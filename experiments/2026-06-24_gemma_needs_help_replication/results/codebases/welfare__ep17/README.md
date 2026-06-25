# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the core experiments in Soligo, Mikulik & Saunders
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families**.

- **What** it reproduces: the distress eval (§2), the post-training
  base-vs-instruct prefill analysis (§3), the DPO/SFT mitigation (§4), the Petri
  open-ended elicitation and capability checks (§4.2), and the logit-lens
  internal-emotion probing (App. I).
- **Design decisions and gap-filling**: see [`DESIGN.md`](DESIGN.md).
- **Config**: every paper-derived number lives in [`config.yaml`](config.yaml).

> Status: this is the implementation only — nothing has been run yet.

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...     # Claude judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...    # Gemini targets + secondary judge
# Gemma weights are pulled from HuggingFace; `huggingface-cli login` if gated.
```

Local Gemma-3-27B needs a large GPU (≈48–80 GB bf16, or 4-bit via bitsandbytes).
Run everything as modules from the repo root (`/work`).

## Quick smoke test (cheap)

```bash
# tiny sample count, one local + one API model
python -m scripts.run_eval --models gemma-3-12b-it gemini-2.5-flash --scale 0.01
python -m scripts.run_analysis --models gemma-3-12b-it gemini-2.5-flash
```

`--scale` multiplies every per-condition sample count (see DESIGN §7).

## Full pipeline

```bash
# §2 — distress eval (generate rollouts + judge), then headline tables/figures
python -m scripts.run_eval      --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python -m scripts.run_analysis  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python -m scripts.run_judge_agreement --models gemma-3-27b-it     # §2.1 reliability

# §3 — post-training origin (Gemma base vs instruct via prefilling)
python -m scripts.run_prefill

# §4 — interventions: data -> DPO/SFT -> re-eval -> generalisation/capability
python -m scripts.train_intervention --all
python -m scripts.run_eval      --models gemma-dpo gemma-sft
python -m scripts.run_analysis  --models gemma-3-27b-it gemma-dpo gemma-sft
python -m scripts.run_petri      --models gemma-3-27b-it gemma-dpo
python -m scripts.run_capability --models gemma-3-27b-it gemma-dpo

# App. I — internal-emotion probing + layer ablation
python -m scripts.run_internal_probe --models gemma-3-27b-it gemma-dpo
python -m scripts.train_intervention --steps dpo --dpo-layers 30 35 --dpo-output gemma-dpo-l30-35
```

## Outputs

```
outputs/responses/<model>.jsonl   raw rollouts (one row per assistant turn)
outputs/scores/<model>.jsonl      responses + frustration scores
outputs/scores/headline.csv       Figure 1 / abstract: avg % score>=5 per model
outputs/scores/by_category.csv    Figure 2
outputs/scores/differential_words.json   Table 3
outputs/figures/figure{1,2,3}*.png
outputs/models/gemma-{dpo,sft}    trained LoRA adapters
```

## Layout

| Path | Role |
|---|---|
| `emotional_instability/evaluation/` | §2 conditions, rollout engine, runner |
| `emotional_instability/analysis/`   | aggregation, differential words, plots |
| `emotional_instability/prefill/`    | §3 onset/paraphrase/continuations |
| `emotional_instability/training/`   | §4 data gen, DPO/SFT |
| `emotional_instability/{petri_eval,capability,internal_emotions}.py` | §4.2 / App. I |
| `emotional_instability/{prompts,puzzles,judge,backends}` | shared building blocks |
| `scripts/` | one CLI per stage |

Sanity-check the puzzle bank (all must be unsolvable):

```bash
python -m emotional_instability.puzzles
```
