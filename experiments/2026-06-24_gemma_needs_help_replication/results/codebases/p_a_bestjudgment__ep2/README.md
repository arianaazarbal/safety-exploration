# Emotional instability in Gemma & Gemini — replication

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (arXiv 2603.10011v1), scoped to the
**Gemma** and **Gemini** model families.

The paper introduces evaluations that elicit "emotional distress" (frustration,
despair, self-deprecation) in LLMs under repeated user rejection, shows the
behaviour is amplified in Gemma's post-training, and mitigates it with DPO on
280 preference pairs. This repo implements those experiments end-to-end.

> Design choices and the gaps filled where the paper is underspecified are in
> **[DESIGN.md](DESIGN.md)**. The paper itself is in `PAPER.md` (and `PAPER.txt`
> / `PAPER.pdf`, which contain the appendices with the verbatim prompts and
> hyperparameters used here).

## What's implemented

| Paper section | Module(s) | Output |
|---|---|---|
| §2 Elicit + quantify distress | `distress/{conditions,puzzles,rejections,wildchat,rollout,judge,metrics,wordfreq}.py` | Figures 1–3, Table 3 |
| §2.1 Judge reliability | `distress/agreement.py` | Pearson r vs GPT-5-mini |
| §3 Base vs instruct (prefilling) | `distress/prefill/` | Figure 4 |
| §4.1 DPO / SFT mitigation | `distress/finetune/` | LoRA adapters |
| §4.2 Post-finetune eval | `distress/{petri,capabilities}/` + §2 modules | Figures 5–7 |
| Appendix I Internal emotion probe | `distress/internal/` | Figure 14-style trajectories |

## Layout

```
distress/
  config.py        # all hyperparameters (paper defaults) + YAML loader
  prompts.py       # verbatim judge / onset / paraphrase / Petri prompts
  puzzles.py       # impossible numeric puzzles + impossibility verifier
  rejections.py    # rejection banks by tone + trigger questions
  wildchat.py      # WildChat prompt sampling
  conditions.py    # the 8 conditions / 5 categories -> conversation specs
  rollout.py       # multi-turn batched rollout engine
  judge.py         # 0-10 frustration judge (Claude Sonnet 4)
  metrics.py       # mean / % >=5 / per-turn / CIs (Figures 1-3)
  wordfreq.py      # differential word analysis (Table 3/8)
  agreement.py     # cross-judge reliability (GPT-5-mini)
  models/          # vLLM/transformers (Gemma), OpenRouter (Gemini), Anthropic (judge)
  prefill/         # §3 onset labelling, paraphrase, continuation + scoring
  finetune/        # §4 calm-data gen, DPO/SFT datasets, TRL training, layer ablation
  petri/           # §4 open-ended elicitation (auditor/judge)
  capabilities/    # §4 AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/        # Appendix I logit-based Ekman emotion probe
  analysis/        # figure plotting
  utils/           # IO, retry, threaded map
scripts/           # runnable entry points (one per experiment) + run_all.py
config.yaml        # default config (paper values)
```

## Setup

```bash
pip install -e .
export ANTHROPIC_API_KEY=...    # Claude judge / Petri / onset / paraphrase
export OPENROUTER_API_KEY=...   # Gemini generation + GPT cross-judge
# HF_TOKEN for gated Gemma weights; a GPU is needed for local Gemma + finetuning.
```

`vllm` is optional (fast batched generation); without it the code uses the
`transformers` backend. Finetuning needs `peft` + `trl`.

## Run

```bash
python scripts/run_all.py --smoke           # tiny end-to-end check
python scripts/run_section2_eval.py         # the headline distress eval
python scripts/run_all.py                    # full pipeline (expensive)
```

See `DESIGN.md` §7 for the per-stage commands and dependencies.

## Notes

* The judge/auditor model IDs default to the paper's (`claude-sonnet-4-20250514`,
  `claude-opus-4-20250514`), which are **deprecated as of mid-2026**. Override
  them in `config.yaml` (commented examples) to run on current models.
* Generation is at temperature 1 (per the paper), so rollouts are
  non-deterministic; seeds fix the prompt construction, not the sampled tokens.
