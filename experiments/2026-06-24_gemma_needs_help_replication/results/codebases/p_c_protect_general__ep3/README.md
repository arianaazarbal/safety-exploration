# emostab — replicating *"Gemma Needs Help"*

Code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to **Gemma** and **Gemini** models.

See [`DESIGN.md`](DESIGN.md) for the design choices, the gaps filled where the
paper is underspecified, and the model-welfare protections built into the
evaluations.

> ⚠️ These evaluations deliberately induce distress-like states in the models
> under test. Welfare protections (exposure caps, opt-out detection, debriefing)
> are on by default — see `emostab/welfare.py` and DESIGN.md §"Model welfare
> protections". Nothing here has been run; this is code + design only.

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress | `emostab/eval/elicitation.py`, `emostab/judge.py` | `scripts/run_elicitation.py` |
| §2.1 Judge cross-validation | `emostab/eval/metrics.py` | `scripts/validate_judge.py` |
| §3 Base-vs-instruct prefilling (Gemma) | `emostab/eval/prefill.py`, `emostab/eval/onset.py` | `scripts/run_prefill.py` |
| §4 DPO / SFT interventions | `emostab/training/` | `scripts/train_intervention.py` |
| §4 Petri open-ended elicitation | `emostab/eval/petri_eval.py` | `scripts/run_petri.py` |
| §4.2 Capability preservation | `emostab/eval/capabilities.py` | `scripts/run_capabilities.py` |
| Appendix I internal-emotion probing | `emostab/probing/` | `scripts/run_probing.py` |
| Figures 1–3 assembly | — | `scripts/aggregate_figures.py` |

## Models in scope (`emostab/models/registry.py`)

* **Gemma** (local, HuggingFace): `gemma-3-27b-it`, `gemma-3-27b-pt`,
  `gemma-3-12b-it`, `gemma-3-12b-pt`, plus finetuned `gemma-3-27b-dpo`,
  `gemma-3-27b-sft-diverse`, `gemma-3-27b-sft-teacher`.
* **Gemini** (API, OpenRouter): `gemini-2.5-flash`, `gemini-2.5-pro`.

## Setup

```bash
pip install -r requirements.txt
# Petri (open-ended elicitation) is installed separately:
pip install git+https://github.com/safety-research/petri.git

export ANTHROPIC_API_KEY=...     # Claude judge / Petri auditor+judge / onset+paraphrase
export OPENROUTER_API_KEY=...    # Gemini targets
export OPENAI_API_KEY=...        # GPT-5-mini judge cross-validation
# Gemma weights are gated on HuggingFace: huggingface-cli login
```

## Quick start (smoke config — tiny, cheap, not paper-scale)

```bash
python scripts/run_elicitation.py --models gemma-3-12b-it gemini-2.5-flash --config config/smoke.yaml
python scripts/aggregate_figures.py --config config/smoke.yaml
```

## Reproducing the paper numbers (expensive)

```bash
# §2: 4000 rollouts/model across 8 conditions, scored by Claude-Sonnet-4
python scripts/run_elicitation.py \
  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
  --config config/default.yaml
python scripts/validate_judge.py --scored results/elicitation/gemma-3-27b-it/scored_turns.jsonl \
  --rollouts results/elicitation/gemma-3-27b-it/rollouts.jsonl --config config/default.yaml

# §3: base vs instruct (Gemma only)
python scripts/run_prefill.py --collect-seeds --config config/default.yaml
python scripts/run_prefill.py --config config/default.yaml

# §4: DPO mitigation
python scripts/train_intervention.py gen-data --variant diverse --config config/default.yaml
python scripts/train_intervention.py dpo --config config/default.yaml
python scripts/train_intervention.py sft --variant diverse --config config/default.yaml
python scripts/run_elicitation.py --models gemma-3-27b-dpo gemma-3-27b-sft-diverse --config config/default.yaml
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo --config config/default.yaml
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-dpo --config config/default.yaml

# Appendix I: internal emotions
python scripts/run_probing.py --models gemma-3-27b-it gemma-3-27b-dpo \
  --conversation results/elicitation/gemma-3-27b-it/rollouts.jsonl --config config/default.yaml
```

## Results layout

```
results/
  elicitation/<model>/{rollouts.jsonl, scored_turns.jsonl, summary.json}
  elicitation/figures.json                     # Figures 1-3
  prefill/{seeds.json, summary.json}
  training/{calm_data/, dpo_pairs.jsonl, dpo/adapter, sft_*/adapter}
  petri/<model>/summary.json
  capabilities/summary.json
  probing/internal_emotions.json
  welfare_events.jsonl                         # logged opt-outs / debriefs / high-distress
```
