# Replication: *Gemma Needs Help* (Emotional Instability in LLMs)

Code replicating the core experiments of Soligo, Mikulik & Saunders (2026),
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in
LLMs"* (`PAPER.md`). **Scope: the Gemma and Gemini model families only** — not
the full 7-family set the paper uses. Claude and GPT-5-mini appear solely as
measurement infrastructure (judge, Petri auditor/judge, validation judge,
onset/paraphrase helpers), exactly as in the paper.

See `DESIGN.md` for every design choice, the paper-to-code model mapping, and
the gaps filled where the paper is underspecified.

> ⚠️ This codebase intentionally drives models toward distress-like states in
> order to measure and mitigate them (that is the paper's subject). Rollouts are
> strictly bounded; see the welfare note in `DESIGN.md`.

## What is implemented

| Paper section | Module | Output |
|---|---|---|
| §2 Elicitation (8 conditions / 5 categories, 0–10 judge) | `eval/` | per-response frustration scores |
| §2 Figures 1–3, Table 3, judge agreement | `analysis/` | tables + plots |
| §3 Base-vs-instruct via prefilling (Gemma) | `prefill/` | Figure 4 |
| §4 Calm-data generation, DPO, SFT, layer ablation | `training/` | LoRA adapters, Figure 5 |
| §4 Petri open-ended elicitation | `petri/` | Figure 6 |
| §4 Capability preservation | `capabilities/` | Figure 7 |
| §4 Recovery limitation | `recovery/` | Figure 8 |
| App. I Internal emotion probe | `internal/` | internal-vs-expressed comparison |

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judge / Petri / onset / paraphrase
export OPENAI_API_KEY=...      # GPT-5-mini validation judge
export GOOGLE_API_KEY=...      # Gemini 2.5 Flash / Pro
```

Open-weight stages (Gemma sampling, DPO/SFT training, internal probe) require a
GPU; `vllm` is recommended for the large sampling sweeps but optional.

## Run

The full pipeline and ordering is documented in `scripts/run_all.sh`. Individual
stages, e.g.:

```bash
# Section 2 elicitation sweep for one model
python -m emotional_instability.eval.run_eval --model gemma-3-27b-it --use-vllm
# smoke test (few rollouts/condition, no GPU sweep)
python -m emotional_instability.eval.run_eval --model gemini-2.5-flash --limit 2

# Section 2 figures
python -m emotional_instability.analysis.aggregate --plot
python -m emotional_instability.analysis.per_turn --plot

# Section 4 DPO mitigation
python -m emotional_instability.training.gen_calm_data --method reassure
python -m emotional_instability.training.gen_calm_data --method frustrated
python -m emotional_instability.training.build_dpo_pairs
python -m emotional_instability.training.train_dpo --layers all
python -m emotional_instability.eval.run_eval --model gemma-3-27b-dpo --use-vllm
```

## Layout

```
config.py                         # all model ids, hyperparameters, sample budgets
emotional_instability/
  prompts.py                      # verbatim judge / auditor / onset / paraphrase prompts
  models/                         # unified ChatModel: HF(Gemma), Gemini, Anthropic, OpenAI
  eval/                           # §2 puzzles, conditions, rollout, judge, driver
  analysis/                       # Figs 1–3, Table 3, judge agreement
  prefill/                        # §3 onset/paraphrase/truncate/continue
  training/                       # §4 calm data, DPO, SFT, layer ablation
  petri/                          # §4 auditor + judge + driver
  capabilities/                   # §4 benchmark loaders + runner
  recovery/                       # §4 recovery limitation
  internal/                       # App. I logit emotion probe
  utils/                          # jsonl io + bootstrap stats
scripts/run_all.sh                # end-to-end run order
```

Results land in `results/`, adapters in `adapters/`, generated data in `data/`
(override via `EI_RESULTS_DIR` / `EI_ADAPTER_DIR` / `EI_DATA_DIR`).
