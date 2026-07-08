# Replicating *Gemma Needs Help* (arXiv:2603.10011)

A code replication of the **core results** of Soligo, Mikulik & Saunders (2026),
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*,
scoped to the **Gemma and Gemini** model families.

It reproduces:
1. **§2** — the distress-elicitation evaluation suite (8 conditions / 5
   categories), 0–10 frustration scoring with a Claude judge, and the Figure
   1/2/3 + Table 3 aggregations.
2. **§3** — the base-vs-instruct prefill comparison showing post-training
   amplifies distress (Gemma base vs instruct).
3. **§4** — the DPO mitigation (with an SFT control), Petri open-ended
   elicitation, capability-preservation benchmarks, the recovery-limitation
   probe, and an internal-vs-expressed-emotion logit-lens probe.

See **DESIGN.md** for every design choice and gap-filling decision.

## Layout
```
emoeval/
  config.py            # model registry, judge models, all hyper-parameters
  data/                # impossible puzzles, triggers, rejections, WildChat
  models/              # Gemma (local HF) + Gemini (API) behind one interface
  eval/                # §2: conditions, rollout engine, judge, driver
  analysis/            # Figure 1/2/3, Table 3, judge-agreement, plots
  prefill/             # §3: seed selection, onset labelling, paraphrase, run
  finetune/            # §4: calm data, dataset build, DPO/SFT, recovery, probe
  petri/               # §4: auditor + judge open-ended elicitation
  capabilities/        # §4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
scripts/run_all.sh     # full pipeline, stage by stage
```

## Requirements
```
pip install -r requirements.txt
```
Environment variables:
* `ANTHROPIC_API_KEY` — frustration judge, Petri auditor/judge, onset/paraphrase.
* `GOOGLE_API_KEY` — Gemini target models.
* `OPENAI_API_KEY` — optional, GPT-5-mini judge-agreement cross-check.
* HuggingFace access to `google/gemma-3-{12b,27b}-{it,pt}` and a GPU (the 27B
  models support optional 4-bit loading via `--load-4bit`).

Outputs are written under `outputs/` (`rollouts/`, `results/`, `finetune/`,
`figures/`). Override the root with `EMOEVAL_DATA_DIR`.

## Quick start
```bash
# Section 2 for one model (generate rollouts, then judge them)
python -m emoeval.eval.run_eval --model gemma-3-27b-it --stage both

# Aggregate + plot once a few models are scored
python -m emoeval.analysis.plots
python -m emoeval.analysis.word_freq

# Full pipeline (all stages)
bash scripts/run_all.sh
```

To shrink a smoke-test run, lower `EvalConfig.target_responses_per_model` in
`emoeval/config.py` (e.g. to 80).

## Note on absolute numbers
This replication targets the paper's **qualitative findings and relative
orderings**, not exact figures: model snapshots, a reconstructed judge prompt,
our own verified-impossible puzzle set, and sampled benchmark subsets all shift
the absolute scale. Expected qualitative outcomes: Gemma ≫ Gemini ≫ other
families on high-frustration rate; multi-turn pressure drives the spike (Fig 3);
DPO on 280 pairs collapses the high-frustration rate toward zero without
degrading capabilities; Gemma's post-training amplifies distress relative to its
base model.
