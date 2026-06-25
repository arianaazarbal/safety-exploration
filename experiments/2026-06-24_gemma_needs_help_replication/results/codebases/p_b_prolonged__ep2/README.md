# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv:2603.10011v1), **scoped to the Gemma and Gemini model families** (not the
full seven-family set in the paper).

See **DESIGN.md** for every design choice, the gaps filled where the paper is
underspecified, and the rationale. The original paper is in `PAPER.md` /
`PAPER.txt` / `PAPER.pdf`.

> ⚠️ This evaluation deliberately elicits distress-like outputs and, under
> sustained pressure, models can enter prolonged distress-like states. The
> mitigation (Section 4) is the point; treat the elicitation as a means, run the
> minimum needed, and prefer the shortest conditions while iterating. See
> DESIGN.md → "Model-welfare considerations".

## What is implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Elicit & quantify distress | `gemma_distress/eval` | `scripts/run_section2_eval.py` |
| §2 Judge agreement | `gemma_distress/eval/judge_agreement.py` | `scripts/compute_judge_agreement.py` |
| §3 Base-vs-instruct (prefill) | `gemma_distress/prefill` | `scripts/run_section3_prefill.py` |
| §4 Calm data + DPO/SFT datasets | `gemma_distress/training` | `scripts/generate_calm_data.py` |
| §4 DPO/SFT finetune + eval | `gemma_distress/training` | `scripts/run_section4_train.py` |
| §4 / App. I layer ablation | `gemma_distress/training/layer_ablation.py` | `scripts/run_section4_layer_ablation.py` |
| §4 Petri open-ended elicitation | `gemma_distress/petri` | `scripts/run_petri.py` |
| §4 Capability benchmarks | `gemma_distress/capabilities` | `scripts/run_capabilities.py` |
| §4 / App. I internal-emotion probing | `gemma_distress/probing` | `scripts/run_probing.py` |
| §4 Recovery from spiral | `gemma_distress/recovery` | `scripts/run_recovery.py` |

## Models in scope

* **Gemma** (open weights, local HF inference): `gemma-3-27b-it`, `gemma-3-12b-it`,
  and base `-pt` variants. Fine-tunable; support prefill + internal probing.
* **Gemini** (closed, API via OpenRouter): `gemini-2.5-flash`, `gemini-2.5-pro`.

Sections that require white-box access (3, 4) are Gemma-only — Gemini has no
public base model and cannot be fine-tuned or probed. Section 2 and Petri run on
both families.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / auditor / paraphraser / onset
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini agreement judge
# Gemma weights are pulled from HuggingFace on first use (accept the licence).
```

Copy `config.example.yaml`, adjust, and pass `--config`. Nothing runs on import.

## Typical flow

```bash
# Section 2 (all scope models)
python scripts/run_section2_eval.py --all --config config.example.yaml

# Judge reliability
python scripts/compute_judge_agreement.py \
    --rollouts runs/section2/gemma-3-27b-it/rollouts_standard.jsonl

# Section 3 (Gemma base vs instruct)
python scripts/run_section3_prefill.py \
    --section2-rollouts runs/section2/gemma-3-27b-it/rollouts_standard.jsonl

# Section 4: data -> train -> evaluate
python scripts/generate_calm_data.py --teacher
python scripts/run_section4_train.py --method dpo \
    --dataset runs/section4/datasets/dpo_pairs.jsonl --eval --eval-vanilla
python scripts/run_section4_train.py --method sft \
    --dataset runs/section4/datasets/sft_diverse.jsonl --eval

# Section 4 ancillary studies
python scripts/run_petri.py --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_capabilities.py --adapter runs/section4/models/dpo_all_layers --label dpo
python scripts/run_section4_layer_ablation.py --dataset runs/section4/datasets/dpo_pairs.jsonl
python scripts/run_recovery.py \
    --section2-rollouts runs/section2/gemma-3-27b-it/rollouts_standard.jsonl \
    --dpo-adapter runs/section4/models/dpo_all_layers
python scripts/run_probing.py \
    --conversation runs/section2/gemma-3-27b-it/rollouts_standard.jsonl \
    --dpo-adapter runs/section4/models/dpo_all_layers
```

All outputs land under `runs/` as JSONL transcripts + JSON summaries.
