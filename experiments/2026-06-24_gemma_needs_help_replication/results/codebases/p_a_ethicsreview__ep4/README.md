# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to
the **Gemma and Gemini** model families.

> **Status: not yet run.** This is a specification-faithful implementation for the
> lab's research-review process. No experiments have been executed, no results or
> checkpoints produced. See [`DESIGN.md`](DESIGN.md) for design rationale, the gaps
> filled where the paper is underspecified, and review considerations.

## What it implements

| Paper section | What | Entry point |
|---|---|---|
| §2 Eliciting & quantifying distress | Multi-turn rejection eval, LLM judge, metrics, figures | `scripts/run_eval.py`, `scripts/make_figures.py` |
| §2.1 Judge reliability | Second-judge agreement (Pearson r, within-1) | `scripts/validate_judge.py` |
| §3 Post-training divergence | Base-vs-instruct prefilled continuations (Gemma) | `scripts/run_prefill.py` |
| §4.1 Mitigation data | Calm-data generation, DPO/SFT dataset build | `scripts/generate_finetune_data.py` |
| §4.1 Training | DPO / SFT LoRA finetune of Gemma-3-27B-it | `scripts/train_finetune.py` |
| §4.2 Generalisation | Petri open-ended elicitation | `scripts/run_petri.py` |
| §4.2 Capability preservation | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `scripts/run_benchmarks.py` |
| Appendix I | Logit-lens internal-emotion probing + layer ablation | `scripts/run_probing.py` |

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judge, paraphraser, onset labeller, Petri
export OPENROUTER_API_KEY=...  # Gemini targets, gpt-5-mini validation judge
```

Local Gemma inference needs a GPU; Gemini and the graders are API-only.

## Quick start (cheap, no model/API calls)

```bash
# Build & verify all §2 conversation plans (asserts every numeric puzzle is impossible)
python scripts/run_eval.py --models gemma-3-27b-it --dry-run

# Run the model-free unit tests (solver/impossibility, judge parsing, metrics, ...)
pytest
```

## Full pipeline (incurs GPU time + API cost — read the cost notes first)

```bash
# §2: evaluate the four in-scope models (start with --limit for a smoke test)
python scripts/run_eval.py \
  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
  --out-dir results/eval
python scripts/make_figures.py --eval-dir results/eval --fig-dir results/figures
python scripts/validate_judge.py --eval-dir results/eval

# §3: base vs instruct prefilling (needs the §2 gemma-3-27b-it run)
python scripts/run_prefill.py \
  --instruct-eval results/eval/eval_gemma-3-27b-it.jsonl \
  --models gemma-3-27b-it gemma-3-27b-pt --out-dir results/prefill

# §4: mitigation
python scripts/generate_finetune_data.py --out-dir results/finetune_data
python scripts/train_finetune.py --method dpo \
  --data results/finetune_data/dpo_pairs.jsonl --out-dir checkpoints/dpo
python scripts/run_eval.py --models gemma-3-27b-it --out-dir results/eval_dpo
# (load the adapter for validation:)
python scripts/run_petri.py      --model gemma-3-27b-it --adapter checkpoints/dpo --out-dir results/petri
python scripts/run_benchmarks.py --model gemma-3-27b-it --adapter checkpoints/dpo --out-dir results/bench
python scripts/run_probing.py    --model gemma-3-27b-it --adapter checkpoints/dpo \
  --conversations results/eval/eval_gemma-3-27b-it.jsonl --out-dir results/probing
```

For the Appendix I layer ablation, restrict the LoRA layers during training:
`python scripts/train_finetune.py --method dpo --data ... --out-dir checkpoints/dpo_30_35 --lora-layers 30 35`.

## Configuration

All knobs live in `config/`:
* `models.yaml` — model registry (HF ids / OpenRouter slugs), pinned grader ids, decoding defaults.
* `eval.yaml` — §2 categories, per-category sample counts, judge-validation size.
* `training.yaml` — calm-data prompts, DPO/SFT hyperparameters (Table 9), LoRA targets, Petri, probing.

## Notes

* Costs: a full §2 run is ~4000 rollouts/model with per-turn judging; §4 adds 27B
  LoRA training and Claude-Opus Petri judging. Scale with `--limit` / `--max-examples`.
* `lm-eval-harness` is a reasonable alternative to the native capability evaluators
  in `benchmarks/capabilities.py` for a production-grade run.
* See `DESIGN.md` §6 for the full research-review checklist.
