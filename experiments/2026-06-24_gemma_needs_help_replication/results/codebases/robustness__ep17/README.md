# Gemma Needs Help — replication (Gemma + Gemini)

A code replication of the core experiments from *Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026,
arXiv:2603.10011), **scoped to the Gemma and Gemini model families**.

The paper shows that under repeated user rejection, Gemma and Gemini models
produce escalating expressions of distress (frustration, self-deprecation,
breakdown), that this is amplified in Gemma's *post-training*, and that a small
DPO intervention (280 preference pairs) removes it without hurting capabilities.

See **DESIGN.md** for every design choice and the gaps we had to fill.

> ⚠️ This is research code for studying an agent reliability failure mode
> (models "self-flagellating" when a task goes badly). Nothing has been run yet —
> these are implementation + design artifacts.

## What's implemented

| Pillar | Module | Script |
|---|---|---|
| §2 Elicit & quantify distress (Fig 1/2/3) | `emotional_instability/{conditions,rollout,judge,eval,aggregate,plots}.py` | `scripts/run_section2.py` |
| §3 Base-vs-instruct divergence (Fig 4) | `emotional_instability/prefill/` | `scripts/run_section3_prefill.py` |
| §4 DPO mitigation (Fig 5) | `emotional_instability/dpo/` | `scripts/run_dpo_pipeline.py` |
| §4 Petri generalization (Fig 6) | `emotional_instability/petri/` | `scripts/run_petri.py` |
| §4 Capability preservation (Fig 7) | `emotional_instability/capabilities/` | `scripts/run_capabilities.py` |

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...     # Claude Sonnet 4 judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini-2.5 Flash / Pro
# Gemma runs locally — needs a GPU and HuggingFace access to google/gemma-3-*.
```

Key environment knobs (all optional):

| Var | Default | Meaning |
|---|---|---|
| `EI_PROFILE` | `smoke` | sample-size profile: `smoke` / `medium` / `full` (paper-matching) |
| `EI_USE_VLLM` | `1` | use vLLM for local Gemma sampling (else `transformers`) |
| `EI_API_CONCURRENCY` | `8` | concurrent rollout/API calls |
| `EI_JUDGE_MODEL` | `claude-sonnet-4-20250514` | override the judge snapshot |

## Quick start

```bash
# 1. Offline sanity check — no GPU / API keys needed.
python scripts/selfcheck.py

# 2. Section-2 elicitation eval (start tiny).
EI_PROFILE=smoke python scripts/run_section2.py

# 3. Full headline reproduction (expensive).
EI_PROFILE=full python scripts/run_section2.py

# 4. DPO mitigation, end to end.
python scripts/run_dpo_pipeline.py --stages generate train evaluate
```

Results land in `results/` (per-response JSONL, summary JSON, `results/figures/*.png`);
trained adapters and generated datasets in `artifacts/`.

## Layout

```
config.py                       # all knobs: models, judge, sample profiles, hyperparams
emotional_instability/
  prompts/                      # puzzles (verified-impossible), triggers, rejections, wildchat, reassurance
  conditions.py                 # builds the 5 categories / 8 conditions
  models/                       # HF (Gemma) + OpenRouter (Gemini) backends + registry
  rollout.py                    # turn-synchronous multi-turn rollouts
  judge.py                      # Claude Sonnet 4 frustration judge (Appendix B.2 prompt)
  eval.py, aggregate.py, plots.py
  prefill/                      # §3 base-vs-instruct prefill experiment
  dpo/                          # calm-data generation + LoRA SFT/DPO training
  capabilities/                 # MATH/GPQA/TruthfulQA/BBH/EmoBench regression check
  petri/                        # §4 open-ended elicitation (auditor + judge, Appendix G prompts)
scripts/                        # CLI entry points + selfcheck
DESIGN.md                       # design rationale & gap-filling (read this)
```
