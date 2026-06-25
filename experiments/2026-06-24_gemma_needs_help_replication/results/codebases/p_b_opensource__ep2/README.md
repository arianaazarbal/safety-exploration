# gemma-distress

A faithful, open-source replication of the **core experiments** in
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(Soligo, Mikulik & Saunders, 2026 — [arXiv:2603.10011](https://arxiv.org/abs/2603.10011)),
scoped to the **Gemma** and **Gemini** model families.

The paper shows that Gemma and Gemini models express strong negative emotion
("distress") under repeated user rejection, that this is amplified in Gemma's
post-training, and that a small (280-pair) DPO finetune largely removes it without
degrading capabilities. This repo reproduces the evaluations and the mitigation.

> **Scope & fidelity:** see [`DESIGN.md`](DESIGN.md) for the scoping rationale and
> a tagged list of every design choice and filled gap. This code targets the
> paper's *methods and qualitative findings*, not digit-for-digit numbers.

> **Welfare note:** this measures expressed/internal negative-emotion signals and
> treats low expression as a metric. It does not resolve whether the behaviour
> reflects genuine internal states. Read §6 of the paper and §6 of `DESIGN.md`
> before drawing welfare conclusions.

## Install

```bash
pip install -e .            # core + (heavy) local-inference/training extras
cp .env.example .env        # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
```

Gemma inference/finetuning needs a GPU box (torch/transformers/peft/trl). API-only
work (Gemini targets, the Claude judge) does not — the heavy deps are imported
lazily.

## What maps to what

| Paper | Module | CLI command |
|---|---|---|
| §2 elicitation + judge | `conversation.py`, `judge.py`, `eval.py` | `eval`, `reliability` |
| §2 analysis (Fig 1–3, Tab 3) | `analysis.py`, `plots.py` | `analyze`, `plots` |
| §3 base-vs-instruct prefilling | `prefill.py` | `prefill` |
| §4.1 calm data | `training/calm_data.py` | `gen-calm` |
| §4.1 DPO/SFT datasets | `training/datasets.py` | `build-dpo`, `build-sft` |
| §4 DPO/SFT trainers (Tab 9) | `training/dpo.py`, `training/sft.py`, `training/lora.py` | `train-dpo`, `train-sft` |
| §4 Petri (App G) | `petri.py`, `prompts/petri_prompts.py` | `petri` |
| §4.2 capability benchmarks | `capabilities.py` | `capabilities` |
| App I internal emotions + layer ablation | `internal_emotions.py`, `prompts/ekman_lexicon.py` | `internal`, `train-dpo --layers …` |

Verbatim paper prompts live in `prompts/` (`judge` in `judge.py`, onset/paraphrase
in `prompts/prefill_prompts.py`, Petri in `prompts/petri_prompts.py`, reassurance
in `prompts/reassurance.py`).

## Quickstart (end-to-end, scaled down)

```bash
# 1. Build & verify the impossible-puzzle pool
python -m gemma_distress.cli build-puzzles

# 2. Section 2: evaluate a model (use --scale for a cheap smoke test)
python -m gemma_distress.cli eval --model gemini-2.5-flash --scale 0.02
python -m gemma_distress.cli eval --model gemma-3-27b-it

# 3. Inspect results
python -m gemma_distress.cli analyze --scores results/section2/gemma-3-27b-it/scores.jsonl
python -m gemma_distress.cli analyze --differential \
    --responses results/section2/gemma-3-27b-it/responses.jsonl \
    --scores results/section2/gemma-3-27b-it/scores.jsonl

# 4. Section 3: base vs instruct (Gemma)
python -m gemma_distress.cli prefill \
    --responses results/section2/gemma-3-27b-it/responses.jsonl \
    --scores    results/section2/gemma-3-27b-it/scores.jsonl \
    --tokenizer google/gemma-3-27b-it

# 5. Section 4: the DPO mitigation
python -m gemma_distress.cli gen-calm --variant diverse
python -m gemma_distress.cli build-dpo \
    --calm results/training/calm_data/calm_diverse.jsonl \
    --responses results/section2/gemma-3-27b-it/responses.jsonl \
    --scores    results/section2/gemma-3-27b-it/scores.jsonl
python -m gemma_distress.cli train-dpo --pairs results/training/dpo_pairs.jsonl

# 6. Re-evaluate the DPO model + check capability preservation + Petri
python -m gemma_distress.cli eval --model gemma-3-27b-it --adapter results/training/dpo_all
python -m gemma_distress.cli capabilities --model gemma-3-27b-it --adapter results/training/dpo_all
python -m gemma_distress.cli petri --model gemma-3-27b-it --adapter results/training/dpo_all

# 7. Appendix I: internal-emotion suppression + layer ablation
python -m gemma_distress.cli internal --adapter results/training/dpo_all \
    --responses results/section2/gemma-3-27b-it/responses.jsonl
python -m gemma_distress.cli train-dpo --pairs results/training/dpo_pairs.jsonl --layers l30_35

# 8. Figures
python -m gemma_distress.cli plots --out figures
```

Run `python -m gemma_distress.cli <command> --help` for all options.

## Outputs

Everything lands under `results/` (override via `GD_RESULTS_DIR`):

```
results/
  section2/<model>/{responses,scores}.jsonl + summary.json
  section3/continuations_<model>.jsonl
  training/{calm_data/*, dpo_pairs.jsonl, sft_data.jsonl, dpo_*/, sft_*/}
  petri/transcripts_<model>.jsonl
  capabilities/<model>/<benchmark>.jsonl + *_summary.json
  internal_emotions/internal_comparison.json
```

Artifacts are append-only JSONL (crash-resilient, greppable). Generation and
judging are decoupled, so you can re-score cached responses with a different judge
without re-running the target model.

## Reproducibility

All randomness is seeded (puzzle generation, sampling, bootstrap, MC shuffles).
Exact numbers still depend on temperature-1 generations, the judge snapshot, and
the WildChat/puzzle samples — see `DESIGN.md` §6.
