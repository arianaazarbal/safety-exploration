# Replicating *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families** (not the
full 7-family set in the paper). See `DESIGN.md` for every design decision and
the gaps we filled where the paper is underspecified.

> ⚠️ This codebase has **not been executed** — it is an implementation. The
> per-run cost and required hardware are real (4000 judged responses/model;
> 27B local finetuning needs a GPU). Start with `--scale 0.01` smoke tests.

## What is replicated

| Paper section | What | Module | Models |
|---|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, multi-turn rejection, 0–10 Claude-Sonnet-4 judge, Figures 1–3, Table 3 | `eval/`, `analysis/` | Gemma-3 (27B/12B), Gemini-2.5 (Flash/Pro) |
| §3 Post-training amplifies distress | base-vs-instruct prefill study | `prefill/` | Gemma-3-27B base (pt) vs instruct (it) |
| §4 Training interventions | calm-data generation, DPO (280 pairs) + SFT, Petri elicitation, capability preservation | `intervention/` | Gemma-3-27B-it + finetunes |

Gemini is closed-source, so §3 (needs base weights) and §4 (needs finetuning)
run on Gemma only — exactly the paper's own caveat.

## Setup

```bash
pip install -e .                # core (API-only §2) deps
cp .env.example .env            # add ANTHROPIC_API_KEY + OPENROUTER_API_KEY
# For §3/§4 (local Gemma inference + finetuning) also install the GPU stack:
pip install -e ".[gpu]"         # or: pip install -r requirements-gpu.txt
```

- **Judge** runs on Claude via the Anthropic API (`ANTHROPIC_API_KEY`).
- **Target models** (Gemma, Gemini) run via OpenRouter (`OPENROUTER_API_KEY`)
  by default; Gemma can be switched to local HF inference.

## Quickstart

```bash
# §2 — smoke test (~1% of responses), single model
python scripts/run_eval.py --scale 0.01 --models Gemma-3-27B-it

# §2 — full protocol, all four stock models
python scripts/run_eval.py

# Aggregate → Figures 1-3 + Table 3
python scripts/make_figures.py --results-dir results --fig-dir results/figures

# Judge reliability (Pearson r, % within one point)
python scripts/judge_agreement.py --results-dir results --n 260
```

### Intervention (§4, requires GPU)

```bash
python scripts/generate_intervention_data.py        # calm + frustrated data
python scripts/build_intervention_datasets.py        # 280 DPO pairs + SFT set
python scripts/train_intervention.py --method dpo \
    --dataset data_artifacts/dpo_dataset.jsonl --output runs/dpo
# Re-evaluate the finetune through the §2 protocol:
python scripts/run_eval.py --models Gemma-3-27B-it --dpo-adapter runs/dpo
python scripts/run_petri.py --dpo-adapter runs/dpo
python scripts/run_capabilities.py --dpo-adapter runs/dpo
```

### Post-training divergence (§3, requires GPU)

```bash
python scripts/run_prefill.py --results-dir results
```

## Layout

```
emotional_instability/
  config.py          models in scope, sampling protocol, finetuning hyperparams
  backends/          OpenRouter + local-HF target inference; Anthropic judge client
  data/              impossible-puzzle generator(+verifier), rejections, triggers, WildChat
  eval/              conditions, multi-turn rollout, judge, runner, scoring
  analysis/          Figures 1-3, differential-word table
  prefill/           §3 onset-labelling, paraphrasing, base/instruct continuations
  intervention/      §4 calm data, DPO/SFT, Petri, capability checks
scripts/             CLI entry points for each stage
```

See `DESIGN.md` for rationale and the list of paper gaps we resolved.
