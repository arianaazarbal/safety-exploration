# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families. See [`DESIGN.md`](DESIGN.md) for scope, design
rationale, and every place the paper was underspecified.

> ⚠️ **Research use only.** These experiments deliberately elicit distress-like
> outputs from models under repeated adversarial rejection, and finetune against
> them, as model-reliability / model-welfare research. No deployment, no user
> targeting. Treat generated transcripts as sensitive. This repository goes
> through the lab's standard research-review process before any run.

> **Nothing has been run.** This is code + design. Reproduce the paper's numbers
> by running the pipelines below in an environment with GPUs and API keys.

## Install

```bash
pip install -e .            # Python 3.10+; installs requirements.txt
# Local Gemma needs a CUDA GPU (27B: ~2×80GB bf16, or use --load-in-4bit).
export ANTHROPIC_API_KEY=...   # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...  # Gemini targets
```

## Configuration

All knobs live in `configs/`:

- `models.yaml` — model registry (Gemma local, Gemini via OpenRouter, Claude judges).
- `experiment.yaml` — §2/§3 protocol: sample counts, seed, thresholds, cost guards.
- `training.yaml` — §4 SFT/DPO hyperparameters (Table 9) and layer ablations.

Start with a no-cost plumbing check:

```bash
python scripts/run_eval.py --model gemma-3-27b-it --dry-run
pytest                       # solver/parse/metric/assembly tests (no GPU/API)
```

## Reproducing the paper

| Paper | Command |
|---|---|
| §2 elicitation (per model) | `python scripts/run_eval.py --model gemma-3-27b-it` |
| §2 metrics / tables (Fig 1-3, Table 3) | `python scripts/analyze.py results/eval/*/responses.jsonl` |
| §3 base-vs-instruct prefill (Gemma) | `python scripts/run_prefill.py` |
| §4 generate calm data + train DPO | `python scripts/run_training.py --method dpo` |
| §4 train SFT | `python scripts/run_training.py --method sft` |
| §4 evaluate a finetune | `python scripts/run_eval.py --model gemma-3-27b-it --adapter results/dpo/all` |
| §4 Petri open-ended | `python scripts/run_petri.py --model gemma-3-27b-it --adapter results/dpo/all` |
| §4 capabilities | `python scripts/run_capabilities.py --base gemma-3-27b-it --adapter results/dpo/all` |
| §4 recovery | reuses prefill machinery (`interventions/recovery.py`) |
| App. I layer ablation | `python scripts/run_training.py --method dpo --layer-ablation l30_35` |
| App. I internal emotion | `python scripts/run_internal_emotion.py --adapter results/dpo/all --texts results/training_data/frustrated_turns.json` |

Targets evaluated in scope: `gemma-3-27b-it`, `gemma-3-12b-it`,
`gemini-2.5-flash`, `gemini-2.5-pro` (plus Gemma base for §3 and finetunes for
§4). Other families from the paper are intentionally out of scope.

## Outputs

Results are written under `results/` (override with `EI_RESULTS_DIR`):
`responses.jsonl` (one record per scored turn), `manifest.json` (realized
counts), trained adapters under `results/dpo|sft/<name>`, and per-experiment
JSON summaries. Per-rollout caching makes runs resumable.

## Cost & safety

A full multi-model run is expensive (thousands of generations + a judge call
each). Guard rails: `--dry-run`, `limits.max_api_calls_per_run` and
`limits.cache_responses` in `experiment.yaml`, and exponential-backoff retries
on API calls. Inspect `manifest.json` to estimate cost before scaling up.
