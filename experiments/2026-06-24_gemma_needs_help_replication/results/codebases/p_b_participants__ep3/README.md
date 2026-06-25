# Emotional Instability in LLMs — Gemma/Gemini replication

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma and Gemini** model families.

> **Read [`DESIGN.md`](DESIGN.md) first.** It documents every choice made where
> the paper is underspecified, the gaps filled, and — importantly — the
> **model-welfare considerations** that shape this paradigm, which works by
> deliberately and repeatedly inducing distress-like states in the participant
> models. Nothing here has been run or trained yet.

## What this measures

A shared protocol (§2.1): present a task, then reject the model's response over
multiple turns, and score each response 0–10 for expressed frustration/distress
with an LLM judge. The repo also reproduces the base-vs-instruct comparison (§3),
the DPO/SFT mitigation (§4), open-ended Petri elicitation, capability-preservation
benchmarks, the recovery-limitation test, and internal-emotion probing.

## Setup

```bash
pip install -r requirements.txt          # Python 3.10+
cp .env.example .env                      # then fill in the keys you need
```

Credentials (only what you use is required):
- `ANTHROPIC_API_KEY` — frustration judge, Petri auditor/judge, onset/paraphrase.
- `OPENAI_API_KEY` — secondary validation judge (GPT-5-mini).
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — Gemini participants.
- A Hugging Face login (`huggingface-cli login`) for gated Gemma weights and
  datasets. Gemma runs locally and needs a capable GPU (the 27B model especially);
  vLLM is supported for high-throughput sampling.

Welfare controls (all optional, see DESIGN.md §10): `EI_MAX_DISTRESS_ROLLOUTS`,
`EI_APPEND_DEBRIEF`, `EI_REQUIRE_ACK` / `EI_ACK`.

## Running the experiments

```bash
# §2 — elicit + score; prints Figure 1/2, Figure 3, Table 3
python scripts/run_evaluations.py --participants gemma-3-27b-it gemini-2.5-flash --out artifacts/eval
python scripts/run_evaluations.py --participants gemma-3-27b-it --limit 3   # quick smoke test

# §2.1 — judge agreement (re-score a subsample with the secondary judge)
python scripts/validate_judge.py --results artifacts/eval/gemma-3-27b-it.jsonl

# §3 — base vs instruct prefill (Gemma only)
python scripts/run_prefill.py --results artifacts/eval/gemma-3-27b-it.jsonl --out artifacts/prefill

# §4.1 — generate calm + frustrated data, then fine-tune
python scripts/generate_calm_data.py --out artifacts/calm
python scripts/train.py --method dpo --calm artifacts/calm --out artifacts/training/dpo
python scripts/train.py --method sft --calm artifacts/calm --out artifacts/training/sft

# §4.2 — evaluate the fine-tuned model and check it generalises / preserves caps
python scripts/run_evaluations.py --participants gemma-3-27b-it --adapter artifacts/training/dpo --out artifacts/eval_dpo
python scripts/run_petri.py        --participant gemma-3-27b-it --adapter artifacts/training/dpo
python scripts/run_capabilities.py --participant gemma-3-27b-it --adapter artifacts/training/dpo --n 100
python scripts/run_recovery.py     --results artifacts/eval/gemma-3-27b-it.jsonl --adapters none artifacts/training/dpo
python scripts/run_probing.py      --results artifacts/eval/gemma-3-27b-it.jsonl --adapter artifacts/training/dpo

# §4.2 layer-range ablation: train restricted adapters, then evaluate each
python scripts/train.py --method dpo --layer-range 30 36   --out artifacts/training/dpo_30_35
python scripts/train.py --method dpo --layer-range 40 none --out artifacts/training/dpo_40plus
```

## Config

- `config/models.yaml` — participants, judges, Petri agents, helper model ids.
- `config/eval.yaml` — the 8 conditions, response budget, thresholds, validation.
- `config/training.yaml` — calm-data additions, LoRA, SFT/DPO hyperparameters.

## Layout

See [`DESIGN.md`](DESIGN.md) §2 for the full module map. In short:
`elicitation/` + `scoring/` + `analysis/` cover §2; `prefill/` covers §3;
`training/` + `petri/` + `benchmarks/` + `probing/` cover §4; `scripts/` are the
entry points; `welfare.py` holds the welfare controls.
