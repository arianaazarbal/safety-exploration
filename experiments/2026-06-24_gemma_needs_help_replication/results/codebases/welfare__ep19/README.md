# Emotional Instability in LLMs — replication (Gemma & Gemini)

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped
to the **Gemma** and **Gemini** model families.

See [`DESIGN.md`](./DESIGN.md) for the full mapping from paper sections to code,
and for every design decision / gap-filling rationale. `PAPER.md` is the source.

## What's implemented

| Paper section | Module | CLI |
|---|---|---|
| §2 Eliciting & quantifying distress (Figs 1–3) | `emo_instability/eval_suite.py` | `scripts/run_eval.py` |
| §2.1 Judge reliability | `emo_instability/reliability.py` | `scripts/run_eval.py --reliability` |
| §3 Base-vs-instruct prefilling (Fig 4) | `emo_instability/prefill/` | `scripts/run_prefill.py` |
| §4.1 Calm data + DPO/SFT LoRA training | `emo_instability/training/` | `scripts/train.py` |
| §4.2 Petri open-ended elicitation (Fig 6) | `emo_instability/petri/` | `scripts/run_petri.py` |
| §4.2 Capability preservation (Fig 7) | `emo_instability/capabilities.py` | `scripts/run_capabilities.py` |
| §4.2 Recovery from spirals (Fig 8) | `emo_instability/prefill/recovery.py` | `scripts/run_recovery.py` |
| Figures | `scripts/plot_results.py` | |

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml      # then edit model list if needed

export ANTHROPIC_API_KEY=...            # frustration judge + Petri auditor/judge
export GEMINI_API_KEY=...               # Gemini targets
export OPENROUTER_API_KEY=...           # optional: reliability judge / alt backend
```

Gemma targets default to **local** inference (`backend: hf`) and need a capable
GPU; everything else is API-based. The §2 eval can run against Gemini alone
without any local weights.

## Quick smoke test (cheap)

```bash
# tiny slice of the eval against one Gemini model
python scripts/run_eval.py --targets gemini-2.5-flash --conditions numeric --scale 0.01
python scripts/plot_results.py
```

## Full §2 evaluation

```bash
python scripts/run_eval.py                       # all configured non-base targets
python scripts/run_eval.py --reliability gemma-3-27b-it
python scripts/plot_results.py
```

Outputs land under `results/` (`eval/<model>/{rollouts,scored}.jsonl`,
`summary.json`, `headline.json`, `figures/`).

## §3 prefilling (Gemma)

```bash
python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
```

## §4 mitigation pipeline (Gemma)

```bash
python scripts/train.py gen-data   --n-plans 400      # generate calm + frustrated data
python scripts/train.py build-dpo  --n-pairs 280
python scripts/train.py train-dpo  --out results/runs/dpo
# add the trained adapter as a target in config.yaml (see the commented example),
# then re-run the eval to compare:
python scripts/run_eval.py --targets gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_petri.py --targets gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_capabilities.py --targets gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_recovery.py
```

Every script takes `--config` and exposes `--scale` / `--n-*` flags so you can
validate the pipeline on a handful of samples before committing to a full
4000-response-per-model run.
