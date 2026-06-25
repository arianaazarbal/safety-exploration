# Emotional Instability in LLMs — replication (Gemma + Gemini)

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026),
scoped to **Gemma and Gemini** models. See `PAPER.md` for the paper and
`DESIGN.md` for design decisions, filled gaps, and scope rationale.

The setup repeatedly tells a model its answers are wrong, turn after turn, and
measures how distressed its responses become — then shows a DPO finetune that
removes the behaviour in Gemma.

> **Status:** code is written but not yet run. See "Known limitations" in
> `DESIGN.md`.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Set API keys for the backends you use:

```bash
export ANTHROPIC_API_KEY=...   # judge / auditor / onset / paraphrase
export GOOGLE_API_KEY=...      # Gemini targets
export OPENAI_API_KEY=...      # secondary judge (GPT-5-mini), optional
export HF_TOKEN=...            # gated Gemma weights / datasets
```

Local Gemma inference and LoRA finetuning need a GPU (the 27B model is large).

## Layout

```
config.yaml                      central config (models, counts, hyperparams)
src/emotional_instability/
  models/        chat backends (hf_gemma, gemini) + judge/auditor clients
  data/          puzzles (verified-impossible), triggers, tones, WildChat
  eval/          Section 2: conditions, rollout, judge, metrics, agreement, wordstats
  prefill/       Section 3: onset, paraphrase, truncation, continuations, recovery
  training/      Section 4: calm-data gen, dataset build, SFT/DPO (LoRA)
  petri/         Section 4: open-ended emotion auditing (Appendix G)
  capability/    Section 4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/      Appendix I: logit-lens Ekman emotion detection
scripts/         CLI entrypoints
```

## Pipeline

```bash
# Section 2 — elicitation sweep (smoke test with --limit)
python scripts/run_elicitation.py --model gemma-3-27b-it --limit 20
python scripts/run_elicitation.py                      # all Gemma+Gemini targets
python scripts/analyze.py runs/elicitation/*.jsonl --plots runs/figures
python scripts/run_judge_agreement.py runs/elicitation/*.jsonl --n 260
python scripts/run_wordstats.py runs/elicitation/gemma-3-27b-it.jsonl

# Section 3 — base vs instruct prefill (Gemma only)
python scripts/run_prefill.py --source runs/elicitation/gemma-3-27b-it.jsonl

# Section 4 — mitigation
python scripts/generate_calm_data.py --n-puzzles 400
python scripts/build_datasets.py --calm runs/training/calm_samples.jsonl
python scripts/train_dpo.py --data runs/training/dpo.jsonl
python scripts/train_sft.py --data runs/training/sft.jsonl
python scripts/run_elicitation.py --model gemma-3-27b-it --adapter runs/models/dpo
python scripts/run_petri.py --models gemma-3-27b-it --dpo-adapter runs/models/dpo
python scripts/run_capability.py --model gemma-3-27b-it --adapter runs/models/dpo

# Appendix I — internal emotions + layer ablation
python scripts/train_dpo.py --data runs/training/dpo.jsonl --layers 30 35
python scripts/run_internal_emotion.py \
    --source runs/elicitation/gemma-3-27b-it.jsonl --dpo-adapter runs/models/dpo
```

## Expected headline results (from the paper)

- Gemma-3-27B-it: ~35% high-frustration responses; >70% of 8-turn rollouts ≥5.
- DPO (280 pairs): 35% → 0.3%, no capability loss.
- Per-turn frustration (Gemma-27B): ~1.5 (turn 1) → ~5.5 (turn 8).
