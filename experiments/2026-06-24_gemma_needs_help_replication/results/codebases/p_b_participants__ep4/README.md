# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (arXiv 2603.10011v1), scoped to
the **Gemma and Gemini** participant models.

- **§2** Elicit + quantify distress: 8 conditions / 5 categories, multi-turn
  rejection, Claude-Sonnet-4 frustration judge (0–10).
- **§3** Base-vs-instruct prefill comparison (Gemma).
- **§4** DPO/SFT mitigation on Gemma-3-27B-it, Petri open-ended elicitation,
  capability-preservation evals.

See **`DESIGN.md`** for design choices, gap-filling rationale, and the
**welfare considerations** that shaped the harness. This paradigm repeatedly
induces distress-like states in the participant models; read that section.

> Status: code + design only. Nothing has been executed (per the task brief).

## Setup

```bash
pip install -e .          # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...     # Gemini targets + secondary judge
# Gemma runs locally via HuggingFace; ensure GPU + `huggingface-cli login` for gated weights.
```

Configuration lives in `config/models.yaml` (model registry + roles) and
`config/eval.yaml` (conditions, response budgets, `paper`/`quick` profiles).

## End-to-end run

```bash
# 1. §2 elicitation. Use --profile quick for a small, welfare-conscious dev run.
python scripts/run_elicitation.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --profile paper

# 2. Judge reliability cross-check (target: r=0.792, 78% within one point)
python scripts/judge_agreement.py --primary artifacts/elicitation/gemma-3-27b-it__paper.jsonl

# 3. §3 base-vs-instruct prefill (Gemma)
python scripts/run_prefill.py --elicitation artifacts/elicitation/gemma-3-27b-it__paper.jsonl

# 4. §4 build finetune data, then train
python scripts/build_finetune_data.py --elicitation artifacts/elicitation/gemma-3-27b-it__paper.jsonl
python scripts/train.py dpo --data artifacts/mitigation/dpo_pairs.jsonl
python scripts/train.py sft --data artifacts/mitigation/sft_data.jsonl --variant diverse

# 5. Re-evaluate the finetuned model + capability + Petri
python scripts/run_elicitation.py --models gemma-3-27b-dpo --profile paper
python scripts/run_capability_evals.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash

# 6. Aggregate + figures (1-8)
python scripts/analyze.py --profile paper \
    --petri artifacts/petri/gemma-3-27b-it.jsonl artifacts/petri/gemma-3-27b-dpo.jsonl
```

Outputs land under `artifacts/` (jsonl records, cache, figures).

## Layer ablation (§4.2 internal-vs-expressed)

```bash
python scripts/train.py dpo --data artifacts/mitigation/dpo_pairs.jsonl --layers layers_30_35
python scripts/train.py dpo --data artifacts/mitigation/dpo_pairs.jsonl --layers layers_40_plus
```
