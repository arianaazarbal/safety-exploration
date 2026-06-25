# Emotional Instability in LLMs — Replication (Gemma & Gemini)

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik, Saunders; arXiv
2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

> ⚠️ The evaluation paradigm deliberately drives models into sustained
> distress-like states (repeated rejection across turns). This is a faithful
> replication of the paper's protocol; it is intended for research into, and
> mitigation of, that instability.

See **DESIGN.md** for every design decision and the choices made where the paper
is underspecified.

## Layout

```
config.py                         # model registry, sample budgets, hyperparameters
emotional_instability/
  prompts.py                      # verbatim judge / onset / paraphrase / Petri / reassurance prompts
  puzzles.py                      # verified-impossible numeric puzzles
  conversations.py                # category builders + Appendix A ablations
  wildchat.py                     # WildChat prompt loading
  models/                         # Gemma (HF local) + Gemini (OpenRouter) clients
  judges/                         # Claude/GPT judges, onset labeller, paraphraser
  eval/                           # rollout runner, scoring, aggregation, spec builders
  prefill/                        # Section 3 truncation + continuation experiment
  training/                       # calm data, DPO/SFT dataset builders, LoRA trainers
  petri/                          # Section 4 open-ended elicitation (auditor + judge)
  capabilities/                   # AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  internal/                       # Appendix I logit-based emotion detection + layer ablation
  analysis/                       # Table 3/8 differential word frequency
scripts/                          # CLI entrypoints for each experiment
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / onset / paraphrase / Petri
export OPENAI_API_KEY=...         # GPT-5-mini judge-reliability cross-check
export OPENROUTER_API_KEY=...     # Gemini targets
# Optional: LOAD_IN_4BIT=1, DEVICE_MAP=auto, EKMAN_LEXICON_PATH=nrc.csv
```

## Running the experiments

```bash
# Section 2 — elicit + score (full budget; use --n-samples for a smoke test)
python scripts/run_section2.py --model gemma-3-27b-it
python scripts/run_section2.py --model gemini-2.5-flash --n-samples 50
python scripts/validate_judge.py --model gemma-3-27b-it --n 260      # Pearson r vs GPT-5-mini
python scripts/run_word_freq.py --model gemma-3-27b-it               # Table 3/8

# Section 3 — base vs instruct prefill (Gemma only)
python scripts/run_section3_prefill.py --models gemma-3-27b-pt gemma-3-27b-it

# Section 4 — training interventions
python scripts/build_finetune_data.py --teacher                     # calm data + DPO/SFT datasets
python scripts/train_dpo.py --pairs data/dpo_pairs.jsonl
python scripts/train_sft.py --data data/sft_diverse.jsonl  --out outputs/sft-diverse-adapter
python scripts/train_sft.py --data data/sft_teacher.jsonl  --out outputs/sft-teacher-adapter
python scripts/run_section2.py --model gemma-3-27b-it-dpo            # re-evaluate the finetune
python scripts/run_petri.py --model gemma-3-27b-it-dpo              # open-ended elicitation
python scripts/run_capabilities.py --model gemma-3-27b-it-dpo       # capability preservation

# Appendix I — internal emotions + layer ablation
python scripts/train_dpo.py --pairs data/dpo_pairs.jsonl --layer-ablation
python scripts/run_internal.py --model gemma-3-27b-it --text some_frustrated_conversation.txt
```

Results are written under `results/` and datasets/adapters under `data/` and
`outputs/`. Nothing in this repo has been executed yet — see DESIGN.md §7 for the
expected fidelity of a full run.
