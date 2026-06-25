# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replication of **Soligo, Mikulik & Saunders (2026), *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"*** (arXiv:2603.10011),
scoped to the **Gemma and Gemini** model families.

The paper documents a reliability failure mode: under repeated user rejection,
Gemma and Gemini models produce escalating expressions of distress (frustration,
self-deprecation, breakdown) — and shows a 280-pair DPO fine-tune that removes the
behaviour without hurting capability. This repo reproduces the core experiments.

> **Design rationale and every gap-filling decision are documented in
> [`DESIGN.md`](DESIGN.md).** Read it first.

## Experiments

| Script | Reproduces | What it does |
|---|---|---|
| `experiments/exp1_elicitation.py` | Fig 1/2/3 | Multi-turn rejection rollouts (8 conditions / 5 categories), 0–10 frustration scoring. |
| `experiments/judge_validation.py` | App. B | Claude-vs-GPT-5-mini judge agreement (Pearson r, %-within-1). |
| `experiments/exp2_prefill.py` | Fig 4 | Base-vs-instruct Gemma via prefilled continuations (post-training amplifies distress). |
| `experiments/exp3a_generate_calm.py` | §4.1 | Generate calm response data with reassuring prompt additions. |
| `experiments/exp3b_build_datasets.py` | App. H | Build the 280 DPO pairs + SFT dataset. |
| `experiments/exp3c_train.py` | App. E | LoRA DPO / SFT of Gemma-3-27B-it. |
| `experiments/exp3d_evaluate.py` | Fig 5 | Re-evaluate vanilla vs DPO vs SFT (35% → 0.3%). |
| `experiments/exp4_petri.py` | Fig 6 | Open-ended Petri elicitation (generalisation). |
| `experiments/exp5_capabilities.py` | Fig 7 | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench parity. |
| `experiments/exp6_probing.py` | App. I | Layer-subset DPO ablation + logit-lens internal-emotion probe. |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...   # frustration judge, Petri auditor+judge, paraphrase
export GOOGLE_API_KEY=...       # Gemini models
export OPENAI_API_KEY=...       # optional: secondary validation judge (gpt-5-mini)
```

Local Gemma generation, prefilling, and training require a GPU and HuggingFace
access to the gated `google/gemma-3-27b-it`, `-12b-it`, and `-27b-pt` weights.

## Running

```bash
# Cheap end-to-end smoke test (tiny sample counts) — default profile:
EI_PROFILE=smoke python run_all.py

# Full replication (paper's sample budgets; expensive):
EI_PROFILE=full python run_all.py

# A single stage / subset:
python run_all.py --only exp1 exp3d

# Or run scripts directly, e.g. just Gemini elicitation:
EI_PROFILE=smoke python experiments/exp1_elicitation.py --models gemini-2.5-flash
```

`EI_PROFILE` (default `smoke`) scales all sample budgets — see `ei/config.py`.

## Outputs

Results stream to `results/` as JSONL (raw rollouts) and JSON (summaries).
Figures are regenerated from summaries via `ei/analysis/plots.py`
(`figure1_table`, `figure2`, `figure3`, `figure5`, `figure6`).

## Layout

See `DESIGN.md §3`. In short: `ei/` is the library (model clients, eval engine,
training, petri, probing, analysis); `experiments/` are thin runnable drivers;
`config.py` holds every paper-specified constant.
