# Replication: *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

A code replication of the core experiments from *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs*
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families**.

The paper's three core claims, and where each is reproduced here:

| Claim | Paper | This repo |
|---|---|---|
| Distress can be reliably elicited in Gemma & Gemini (not other families) | §2 | `emotional_instability/eval` + `scripts/run_section2_eval.py` |
| The propensity is amplified in Gemma's **post-training** | §3 | `emotional_instability/prefill` + `scripts/run_prefill.py` (Gemma base vs instruct) |
| A small **DPO** fine-tune (280 pairs) removes it without hurting capabilities | §4 | `emotional_instability/dpo`, `petri`, `capabilities` |

See **DESIGN.md** for every design decision and the gaps we filled where the
paper is underspecified.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
```

Local Gemma inference needs a GPU box (vLLM); the 27B model wants ~2x A100/H100
or 4-bit loading. Gemini runs through OpenRouter (API only). The Claude judge,
Petri auditor/judge, onset labelling and paraphrasing use the Anthropic API.

## Quick smoke test (cheap)

```bash
python scripts/check_puzzles.py                      # confirm puzzles are impossible
EVAL_SCALE=0.02 python scripts/run_section2_eval.py --models gemma-3-27b-it
python scripts/run_analysis.py
```

## Full pipeline

```bash
# Section 2 -- elicitation across Gemma + Gemini
python scripts/run_section2_eval.py --models \
    gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/validate_judge.py            # judge reliability vs GPT-5-mini
python scripts/run_analysis.py              # Figures 1/2/3 tables

# Section 3 -- base vs instruct (Gemma only; Gemini has no public base model)
python scripts/run_prefill.py

# Section 4 -- the DPO mitigation
python scripts/generate_calm_data.py        # calm + frustrated pools
python scripts/build_datasets.py            # 280 DPO pairs + 1150 SFT samples
python scripts/train.py dpo                 # LoRA rank-64, 1 epoch, lr 5e-5
python scripts/train.py sft                 # SFT baseline (ineffective, per paper)
python scripts/run_section2_eval.py --models dpo-gemma-3-27b \
    --lora results/section4/adapters/dpo    # re-run the eval on the fine-tune
python scripts/run_petri.py --models gemma-3-27b-it dpo-gemma-3-27b gemini-2.5-flash
python scripts/run_capabilities.py --models gemma-3-27b-it dpo-gemma-3-27b \
    --lora results/section4/adapters/dpo
python scripts/run_analysis.py
```

## Layout

```
config.py                     all experiment knobs (models, samples, hparams)
emotional_instability/
  puzzles.py  prompts.py  wildchat.py     eval stimuli
  conversation.py                          multi-turn rollout engine
  judge.py                                 Claude frustration judge (+ GPT-5-mini)
  models/                                  Gemma (vLLM) + Gemini (OpenRouter)
  eval/        Section 2 elicitation
  prefill/     Section 3 base-vs-instruct
  dpo/         Section 4 calm-data gen + SFT/DPO training
  petri/       Section 4.2 open-ended elicitation
  capabilities/Section 4.2 capability benchmarks
  analysis/    aggregation -> Figures 1/2/3 tables
scripts/                       CLI entry points (one per experiment stage)
```

> Status: code + design only. Nothing in here has been executed; see DESIGN.md
> §"What is and isn't validated".
