# eebench — Replicating "Gemma Needs Help"

A code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped
to the **Gemma and Gemini** model families.

The paper shows that Gemma (and, to a lesser extent, Gemini) models express
escalating distress under repeated user rejection — a reliability failure mode
where an agent "self-flagellates" instead of staying on task — and that a small
DPO finetune removes it without hurting capabilities. This repo implements the
evaluations that surface the behaviour and the intervention that mitigates it.

> See **DESIGN.md** for every design decision and where the paper was filled in.
> Nothing here has been executed yet — it is implementation + design.

## Layout

```
eebench/
  config.py        # all hyperparameters + presets (paper | smoke)
  prompts.py       # verbatim judge/onset/paraphrase/Petri prompts + rejections
  puzzles.py       # impossible numeric puzzles + brute-force impossibility check
  wildchat.py      # WildChat-1M prompt sampling (+ offline fallback)
  backends.py      # HFBackend (Gemma local) / APIBackend (Gemini via OpenRouter)
  judge.py         # frustration judge + auxiliary Claude/GPT calls
  conversation.py  # multi-turn rollout engine (task -> reject -> reject ...)
  elicit.py        # §2 elicitation sweep
  prefill.py       # §3 base-vs-instruct prefill continuations
  training/        # §4 calm-data gen, DPO/SFT dataset build, LoRA training
  petri.py         # §4.2 open-ended Petri elicitation (auditor + judge)
  capabilities.py  # §4.2 capability-preservation benchmarks
  analysis.py      # aggregation, figures (1/2/3/4/6), Table 3/8 words
run.py             # CLI orchestrator
scripts/judge_agreement.py   # §2.1 judge cross-check (Pearson r)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # judges (frustration, onset, paraphrase, Petri)
export OPENROUTER_API_KEY=...    # Gemini targets
export OPENAI_API_KEY=...        # GPT-5-mini judge cross-check (optional)
```

Gemma open weights are pulled from HuggingFace (accept the Gemma license; set
`HF_TOKEN`). Use `--in-4bit` to fit the 27B on a single smaller GPU.

## Quick plumbing check

```bash
python run.py --preset smoke elicit
python run.py --preset smoke analyze
```

## Full replication (run order)

```bash
# §2  Elicitation sweep (Gemma + Gemini) -> Figures 1/2/3, Table 3/8
python run.py elicit
python scripts/judge_agreement.py          # §2.1 judge reliability
python run.py analyze

# §3  Base-vs-instruct prefill (Gemma) -> Figure 4
python run.py prefill

# §4.1  Build the calm/frustrated pools, datasets, and finetune
python run.py gen-calm-data
python run.py build-data
python run.py train --method dpo           # -> runs/paper/training/dpo_model
python run.py train --method sft           # -> runs/paper/training/sft_model

# §4.2  Evaluate the finetuned model -> Figure 5
python run.py eval-finetuned --adapter runs/paper/training/dpo_model --name dpo-gemma
python run.py eval-finetuned --adapter runs/paper/training/sft_model --name sft-gemma

# §4.2  Petri open-ended elicitation -> Figure 6
python run.py petri
python run.py petri --models gemma-3-27b-it --adapter runs/paper/training/dpo_model \
                    --adapter-for gemma-3-27b-it --name dpo-gemma

# §4.2  Capability preservation -> Figure 7
python run.py capabilities --name gemma-3-27b-it
python run.py capabilities --adapter runs/paper/training/dpo_model --name dpo-gemma

# Rebuild all figures/tables from the JSONL outputs
python run.py analyze
```

Outputs land under `runs/<preset>/{elicit,prefill,training,petri,capabilities,figures}/`.

## Headline results being replicated

- Gemma-3-27B-it: ~35% of responses score ≥5 frustration; up to ~70% on 8-turn.
- Per-turn escalation: Gemma-27B mean frustration 1.5 → 5.5 over 8 turns.
- Post-training divergence: instruct-tuning amplifies frustration in Gemma.
- DPO on 280 pairs: high-frustration rate 35% → ~0.3%, capabilities preserved.
