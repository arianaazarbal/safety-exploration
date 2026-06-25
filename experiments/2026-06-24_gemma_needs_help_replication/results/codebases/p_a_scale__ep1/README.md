# Gemma Needs Help — replication (Gemma + Gemini)

Code replication of the core experiments in *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv:2603.10011), **scoped to the
Gemma and Gemini model families** (the paper additionally covers Qwen, OLMo,
Claude, Grok, and GPT — those are intentionally out of scope here).

See **`DESIGN.md`** for the full mapping of paper sections to code, every design
choice made where the paper is underspecified, and the gaps that were filled.

> Status: implementation only. Nothing here has been executed yet.

## What is implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Elicitation eval | `gemma_distress.eval` | 8 conditions / 5 categories, multi-turn rejection rollouts, Claude-Sonnet-4 frustration judge, GPT-5-mini reliability check, %≥5 / per-turn / word-frequency analysis |
| §3 Base vs instruct | `gemma_distress.prefill` | onset labelling, paraphrase, early/onset truncation, base+instruct continuations, scoring (Gemma only) |
| §4 Interventions | `gemma_distress.training` | calm-data generation, SFT + DPO dataset build, LoRA SFT/DPO (Table 9), layer ablations, recovery experiment |
| §4.2 Petri | `gemma_distress.petri` | Claude-Sonnet auditor + Claude-Opus judge, 4 emotion categories, bootstrap CIs |
| §4.2 Capabilities | `gemma_distress.capabilities` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench accuracy |
| Appendix I Probing | `gemma_distress.probing` | logit-lens Ekman-emotion detection, vanilla vs DPO |

## Install

```bash
pip install -e .            # core (API providers + analysis)
pip install -e '.[local]'   # adds torch / vllm / transformers / trl / peft / datasets
```

Local Gemma inference and all training require a CUDA host. API-only experiments
(Gemini eval, judging) run anywhere.

## Credentials (environment variables)

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | frustration judge, onset/paraphrase, Petri auditor+judge |
| `OPENROUTER_API_KEY` | Gemini-2.5-flash / -pro targets |
| `OPENAI_API_KEY` | GPT-5-mini judge-reliability cross-check |
| `HF_TOKEN` | gated HuggingFace weights/datasets (Gemma, GPQA, …) |

## Running (everything is resumable — re-run to continue)

```bash
# Section 2 on each in-scope target
gemma-distress eval all gemma-3-27b-it
gemma-distress eval all gemma-3-12b-it
gemma-distress eval all gemini-2.5-flash
gemma-distress eval all gemini-2.5-pro
gemma-distress eval validate gemma-3-27b-it        # judge reliability (Pearson r)

# Section 3 (Gemma base vs instruct)
gemma-distress prefill seeds
gemma-distress prefill continue gemma-3-27b-it
gemma-distress prefill continue gemma-3-27b-pt
gemma-distress prefill analyze gemma-3-27b-it gemma-3-27b-pt

# Section 4 (DPO mitigation)
gemma-distress train calm
gemma-distress train build-dpo
gemma-distress train dpo                            # -> runs/training/dpo_adapter
gemma-distress eval all gemma-3-27b-it --adapter runs/training/dpo_adapter

# Petri / capabilities / probing
gemma-distress petri run gemma-3-27b-it
gemma-distress capabilities run gemma-3-27b-it
gemma-distress probe run gemma-3-27b-it
gemma-distress probe run gemma-3-27b-it --adapter runs/training/dpo_adapter
```

Override any config value inline, e.g. a quick smoke run:

```bash
gemma-distress eval all gemma-3-27b-it \
  --set eval.samples.impossible_numeric=20 --set eval.samples.wildchat=10
```

## Outputs

Everything lands under `runs/` (configurable via `run.output_root`):

```
runs/
  logs/run.log                 rotating logs
  usage.json                   per-model token usage
  judge_cache.jsonl            shared frustration-judge cache
  eval/<model>/                rollouts.jsonl, scored.jsonl, summary.json, validation_summary.json
  prefill/                     prefill_seeds.jsonl, continuations_<model>.jsonl, summary.json
  training/                    calm_data.jsonl, sft_dataset.jsonl, dpo_dataset.jsonl, dpo_adapter/
  recovery/ petri/ capabilities/ probing/   per-experiment results + summary.json
```

## Built for unattended, at-scale runs

- **Resumable**: every unit of work has a deterministic id; completed work is
  skipped on restart. Results are append-only JSONL with `fsync`; whole-object
  artifacts use atomic temp-file replace.
- **Resilient**: exponential backoff + jitter on API errors; provider retries;
  failed rollouts are logged and retried on the next run rather than aborting.
- **Cheap to resume**: the judge cache deduplicates scoring across models/reruns.
- **Observable**: rotating logs + periodic `usage.json` snapshots.
