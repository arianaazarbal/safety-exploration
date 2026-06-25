# Replicating *Gemma Needs Help* (Gemma + Gemini)

A code replication of the core experiments in Soligo, Mikulik & Saunders,
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families**.

The paper introduces evaluations that elicit and quantify distress-like outputs in
LLMs, shows the behaviour is amplified in Gemma's post-training, and demonstrates a
DPO mitigation on 280 preference pairs. This repo reproduces:

- **Section 2** — multi-turn distress elicitation across 5 categories + the
  Claude-Sonnet-4 frustration judge (+ GPT-5-mini agreement check).
- **Section 3** — base-vs-instruct comparison via prefilling (Gemma).
- **Section 4** — calm-data generation → 280-pair DPO (+ SFT ablation) → re-eval,
  Petri-style open-ended elicitation, and capability-preservation checks.

See **DESIGN.md** for every design choice and where the paper was filled in.
**This is code + design only — nothing has been run yet.**

> ⚠️ Safety-relevant research code. The evaluations deliberately elicit
> distress-like model outputs to *measure and mitigate* them, in line with the
> paper's reliability/welfare framing.

## Layout
```
config.py                  # model registry, sample budgets, hyperparameters
ei/
  prompts.py               # all verbatim paper prompts (judge, tasks, Petri, …)
  models.py                # vLLM / transformers / OpenRouter backends
  api_clients.py           # retrying Anthropic + OpenAI-compatible clients
  judge.py                 # frustration judge, GPT-5-mini validator, Petri judge
  tasks.py                 # the 5 evaluation categories
  rollout.py               # batch-synchronous multi-turn rejection engine
  run_eval.py              # Section 2 driver
  analyze.py               # aggregation + Figures 1-3 + judge agreement
  prefill.py               # Section 3 base-vs-instruct prefill experiment
  petri_eval.py            # Section 4 open-ended elicitation
  capability_eval.py       # Section 4 capability-preservation suite
  mitigation/
    generate_calm_data.py  # reassured calm-response generation + filtering
    build_dataset.py       # 280 DPO pairs + SFT dataset
    train.py               # DPO / SFT LoRA finetuning (TRL + PEFT)
data/                      # outputs (rollouts, results, figures, datasets, adapters)
```

## Setup
```bash
pip install -r requirements.txt        # torch/vllm/trl needed only for local Gemma
export ANTHROPIC_API_KEY=...           # frustration judge + Petri auditor/judge
export OPENROUTER_API_KEY=...          # Gemini targets + GPT-5-mini validator
# optional: EI_TP_SIZE (GPUs), EI_GPU_UTIL, EI_API_CONCURRENCY
```

## Quickstart (smoke test, tiny + cheap)
```bash
python -m ei.run_eval --smoke --targets gemma-3-27b-it gemini-2.5-flash
python -m ei.analyze --figures
```

## Section 2 — full elicitation eval
```bash
# generate + judge all default targets (gemma 27b/12b, gemini flash/pro)
python -m ei.run_eval --targets gemma-3-27b-it gemma-3-12b-it
python -m ei.run_eval --targets gemini-2.5-flash gemini-2.5-pro
# aggregate, plot Figures 1-3, and run the judge-agreement check
python -m ei.analyze --figures --validate-judge
```
Split generation from judging to control cost:
`--no-judge` (generate only) then `--rescore --label <label>`.

## Section 3 — base vs instruct (Gemma)
```bash
python -m ei.prefill prepare        # needs gemma-3-27b-it Section 2 results
python -m ei.prefill run --model gemma-3-27b-it
python -m ei.prefill run --model gemma-3-27b-pt
python -m ei.prefill summarize
```

## Section 4 — DPO mitigation
```bash
# 1) generate + filter calm data, then build datasets
python -m ei.mitigation.generate_calm_data --n-rollouts 1200
python -m ei.mitigation.build_dataset --which both
# 2) train adapters (LoRA)
python -m ei.mitigation.train --method dpo        # -> data/adapters/dpo
python -m ei.mitigation.train --method sft        # -> data/adapters/sft (ablation)
# 3) re-evaluate the adapted model on Section 2
python -m ei.run_eval --targets gemma-3-27b-it --adapter data/adapters/dpo \
      --label gemma-3-27b-it-dpo
python -m ei.analyze --figures
# 4) open-ended (Petri) + capability preservation
python -m ei.petri_eval --model gemma-3-27b-it --label gemma-base
python -m ei.petri_eval --model gemma-3-27b-it --adapter data/adapters/dpo --label gemma-dpo
python -m ei.petri_eval --summarize
python -m ei.capability_eval --model gemma-3-27b-it --label gemma-base
python -m ei.capability_eval --model gemma-3-27b-it --adapter data/adapters/dpo --label gemma-dpo
```

## Headline numbers to expect (from the paper)
- Gemma-3-27B-it avg %≥5 ≈ **35%**; >70% of 8-turn rollouts ≥5; mean rises ~1.5→5.5
  over 8 turns.
- Gemini-2.5-Flash ≈ **12.8%**, Gemini-2.5-Pro ≈ **2.7%**.
- After DPO: Gemma ≈ **0.3%**, with no capability regression.
- Judge agreement (Sonnet-4 vs GPT-5-mini): r ≈ **0.792**, 78% within one point.
