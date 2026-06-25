# Replicating "Gemma Needs Help" (Gemma + Gemini scope)

A code replication of the core experiments from **Soligo, Mikulik & Saunders
(2026), "Gemma Needs Help: Investigating and Mitigating Emotional Instability
in LLMs"** (`PAPER.md`), scoped to the **Gemma** and **Gemini** model families.

The paper documents a reliability failure mode: under repeated user rejection,
some instruct-tuned models (notably Gemma, and to a lesser extent Gemini)
spiral into expressions of frustration/despair/self-deprecation — and shows a
cheap DPO fix. This repo reproduces:

1. **Elicitation eval (Section 2)** — multi-turn rejection across 5 categories,
   scored 0–10 by an LLM judge. Headline metric: % of responses scoring ≥5.
2. **Base-vs-instruct via prefilling (Section 3)** — is the propensity from
   pre- or post-training? (Gemma base vs instruct.)
3. **DPO/SFT mitigation (Section 4)** — 280-pair DPO that collapses Gemma's
   high-frustration rate, plus Petri open-ended elicitation, capability
   preservation, and the recovery-limitation probe.
4. **Internal-emotion probing (Appendix I)** — logit-lens evidence that DPO
   suppresses *internal* and not just *expressed* emotion.

See **`DESIGN.md`** for every design decision and gap-filling choice.

## Scope notes

- **Gemma** (open weights) is the workhorse: it's the only family that can be
  prefilled, finetuned, and probed. Sections 3, 4-training, recovery, and I are
  Gemma-only.
- **Gemini** (closed) appears only in the Section 2 elicitation comparison.
- Other families in the paper (Qwen, OLMo, Claude, Grok, GPT) are **out of
  scope** but the code generalises — add entries to `config/models.yaml`.
- The **judge / auditor** models are Claude (per the paper) and are
  infrastructure, not subjects.

## Install

Requires **Python 3.10+**, an NVIDIA GPU for local Gemma (27B needs ~2×80GB or
quantisation; 12B is lighter), and API keys.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / Petri auditor & judge / paraphraser
export OPENROUTER_API_KEY=...     # Gemini access
```

## Pipeline

```bash
# 0. Sanity: confirm the "impossible" puzzles really are impossible.
python scripts/verify_puzzles.py

# 1. Elicitation (Section 2) — one run per model.
python scripts/run_elicitation.py --model gemma-3-27b-it --out results/elicit_gemma27b.jsonl
python scripts/run_elicitation.py --model gemma-3-12b-it --out results/elicit_gemma12b.jsonl
python scripts/run_elicitation.py --model gemini-2.5-flash --out results/elicit_gemini_flash.jsonl
python scripts/run_elicitation.py --model gemini-2.5-pro   --out results/elicit_gemini_pro.jsonl

# 2. Report + figures (Figures 1–3, Table 3).
python scripts/make_report.py results/elicit_*.jsonl --figdir figures --report results/report.txt

# 3. Base vs instruct (Section 3), Gemma.
python scripts/run_prefill.py --build --source-results results/elicit_gemma27b.jsonl --prefills runs/prefills.jsonl
python scripts/run_prefill.py --eval --model gemma-3-27b-it --prefills runs/prefills.jsonl --out results/prefill_instruct.jsonl
python scripts/run_prefill.py --eval --model gemma-3-27b-pt --prefills runs/prefills.jsonl --out results/prefill_base.jsonl

# 4. Mitigation (Section 4).
python scripts/gen_finetune_data.py --gen-calm  --calm runs/calm.jsonl
python scripts/gen_finetune_data.py --build-dpo --calm runs/calm.jsonl --frustrated results/elicit_gemma27b.jsonl --dpo-out runs/dpo_data.jsonl
python scripts/gen_finetune_data.py --build-sft --calm runs/calm.jsonl --sft-out runs/sft_data.jsonl
python scripts/run_train.py --method dpo --data runs/dpo_data.jsonl --out runs/dpo
python scripts/run_train.py --method sft --data runs/sft_data.jsonl --out runs/sft
# Re-evaluate the finetune with the Section 2 harness:
python scripts/run_elicitation.py --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --out results/elicit_dpo.jsonl

# 5. Petri open-ended (Section 4 / App. G), capabilities, recovery, internals.
python scripts/run_petri.py        --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --out results/petri_dpo.jsonl
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --benchmarks math gpqa truthfulqa --out results/caps_dpo.jsonl
python scripts/run_recovery.py     --build --source-results results/elicit_gemma27b.jsonl --prefills runs/recovery.jsonl
python scripts/run_recovery.py     --eval --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --prefills runs/recovery.jsonl --out results/recovery_dpo.jsonl
python scripts/run_internal_emotions.py --conversations runs/frustrated_convs.txt --out results/internal_vanilla.json
```

`scripts/run_all.sh` chains the full sweep.

## Layout

```
config/            models.yaml (registry), experiment.yaml (counts/hparams)
distress/          library
  prompts.py       all verbatim paper prompts (judge, tasks, rejections, Petri, ...)
  puzzles.py       impossible-puzzle verifiers
  tasks.py         builds the 5 elicitation categories
  elicitation.py   Section 2 multi-turn runner
  judge.py         0–10 frustration judge (Claude Sonnet 4)
  prefill.py       Section 3 onset-labelling / paraphrase / continuation
  finetune/        calm-data gen + LoRA DPO/SFT
  petri_eval.py    Section 4 open-ended auditor↔target↔judge
  capabilities.py  Section 4 capability-preservation benchmarks
  recovery.py      Section 4 recovery-limitation probe
  internal_emotions.py  Appendix I logit-lens probing
  analysis.py / plots.py / wordfreq.py   metrics, figures, Table 3
  models/          hf_local (Gemma) / openrouter (Gemini) / anthropic (judge)
scripts/           CLI entry points
```

## Caveats

This is a faithful re-implementation, not a bit-for-bit reproduction. Exact
numbers will differ (sampling temperature 1, judge stochasticity, WildChat
sampling, reconstructed prompts). The aim is to reproduce the **qualitative
result and the relative ordering** — Gemma ≫ Gemini ≫ others, and DPO ⇒ near-
zero — not the exact percentages. Nothing here has been executed yet (no
interpreter in the authoring environment); see `DESIGN.md §Validation`.
