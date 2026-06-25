# Emotional Instability in LLMs — Replication (Gemma + Gemini scope)

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026), scoped to the **Gemma**
and **Gemini** model families. See [`DESIGN.md`](DESIGN.md) for the mapping from
paper sections to modules, and for every design choice made where the paper is
underspecified.

> Status: implementation only. Nothing here has been executed yet — see
> `DESIGN.md` §"What has and hasn't been validated".

## Layout

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting distress | `eval/` | Multi-turn rollout engine + Claude judge (8 conditions, 5 categories) |
| §2 data | `data/` | Verifiably-impossible puzzles, triggers, rejections, WildChat |
| App. A controls | `eval/ablations.py` | Neutral-continuation / redacted-turns / fake-multiturn |
| §3 Post-training | `prefill/` | Onset labelling, paraphrase, truncation, continuations |
| §4 Interventions | `training/` | Calm-data gen, DPO/SFT dataset build, LoRA trainers |
| §4.2 Petri | `petri/` | Auditor/judge open-ended elicitation |
| §4.2 Capabilities | `capabilities/` | lm-eval-harness + EmoBench wrappers |
| App. I Probing | `probing/` | Logit-lens internal-emotion detection, layer ablation |
| Figures 1-8 | `analysis/` | Aggregation, differential words, plots |

## Setup

```bash
pip install -e .
# External frameworks for §4.2 (optional, only if running those stages):
pip install lm-eval                                   # capability benchmarks
# pip install git+https://github.com/safety-research/petri   # or use the built-in loop

export ANTHROPIC_API_KEY=...      # Claude judge + Petri auditor/judge
export OPENROUTER_API_KEY=...     # Gemini targets
# Gemma open weights run locally (vLLM / transformers) — needs GPUs.
```

## Running

The whole pipeline is in `scripts/run_pipeline.sh`. Individual stages:

```bash
# §2 main evaluation + headline metrics
python -m emotional_instability.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
python -m emotional_instability.analysis.aggregate

# §4 DPO mitigation
python -m emotional_instability.training.build_dpo
python -m emotional_instability.training.train_dpo
```

Tests (pure logic, offline):

```bash
pytest tests/        # puzzle-impossibility verifier + condition assembly
```

## Configuration

Everything is driven by `config/default.yaml` (sample counts, hyperparameters,
model ids). Paper-specified values are used verbatim; choices for underspecified
points are marked `[choice]` and explained in `DESIGN.md`.
