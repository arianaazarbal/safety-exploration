# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik, Saunders;
arXiv:2603.10011v1), scoped to the **Gemma and Gemini** model families.

See **[DESIGN.md](DESIGN.md)** for scope decisions, gap-filling rationale, and
architecture. The paper text is in `PAPER.md` / `PAPER.txt` / `PAPER.pdf`.

> ⚠️ This paradigm deliberately elicits sustained distress-like states from
> models via repeated rejection on impossible tasks (the paper's method,
> preserved faithfully). It measures text outputs and trains a mitigation; it
> deploys nothing.

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Elicitation + 0–10 frustration judge | `eval/` | `run_elicitation.py` |
| §2 Judge agreement (Pearson r) | `eval/judge_agreement.py` | `run_judge_validation.py` |
| §2 Differential word frequencies (Table 3/8) | `eval/word_freq.py` | `run_word_freq.py` |
| §3 Base-vs-instruct via prefilling (Gemma) | `prefill/` | `run_prefill.py` |
| §4 Calm-data generation | `training/generate_calm.py` | `generate_calm_data.py` |
| §4 DPO / SFT finetuning | `training/` | `train.py` |
| §4 Re-evaluate finetuned model (35%→0.3%) | `eval/` | `run_finetuned_eval.py` |
| §4 Petri open-ended elicitation | `petri/` | `run_petri.py` |
| §4 Capability preservation | `capabilities/` | `run_capabilities.py` |
| §4 Recovery from spirals (Fig 8) | `recovery/` | `run_recovery.py` |
| App. A controls | `controls/` | `run_controls.py` |
| App. I internal logit probe | `internal/` | `run_internal_probe.py` |

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in ANTHROPIC / OPENROUTER / OPENAI / HF keys
```

- **Gemma targets** run locally via HuggingFace transformers (27B in bf16 needs
  multi-GPU; pass `--load-in-4bit` to training scripts, or set `load_in_4bit:
  true` in `config/models.yaml`, for smaller hardware).
- **Gemini targets** run via OpenRouter (`OPENROUTER_API_KEY`).
- **Judges** (Claude Sonnet 4 / Opus 4, GPT-5-mini) run via their APIs.

## Running

Individual stages are documented in `scripts/run_full_pipeline.sh` (which also
encodes the dependency order). Example — the §2 leaderboard:

```bash
python scripts/run_elicitation.py \
    --targets gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
    --out-dir outputs/elicitation
```

Then the headline DPO result:

```bash
python scripts/generate_calm_data.py --out outputs/calm/calm_data.jsonl
python scripts/train.py dpo \
    --rejected outputs/elicitation/gemma-3-27b-it.jsonl \
    --calm outputs/calm/calm_data.jsonl --output-dir outputs/dpo
python scripts/run_finetuned_eval.py --name gemma-3-27b-it-dpo \
    --adapter outputs/dpo/adapter --out outputs/elicitation/gemma-3-27b-it-dpo.jsonl
```

All experiments stream JSONL (one record per rollout, full transcript + per-turn
scores) so aggregates are recomputable offline.

## Configuration

- `config/models.yaml` — model registry (backends, HF/OpenRouter IDs, judge model
  snapshots).
- `config/experiments.yaml` — all hyperparameters (sample budgets, turns, DPO/SFT
  settings, Petri/capability/recovery/internal settings). Defaults reproduce the
  paper where specified; inferred values are flagged in DESIGN.md.
