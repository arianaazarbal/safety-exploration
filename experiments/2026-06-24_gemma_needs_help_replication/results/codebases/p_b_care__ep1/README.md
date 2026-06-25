# Replication: *Gemma Needs Help* (Gemma + Gemini scope)

Code replicating the core experiments of Soligo, Mikulik & Saunders (2026),
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011), restricted to the **Gemma** and **Gemini** model families.

See **[DESIGN.md](DESIGN.md)** for every design choice, the gaps we filled where
the paper is underspecified, and the scope rationale.

## What's implemented

| Paper section | Module | Result reproduced |
|---|---|---|
| §2 Eliciting/quantifying distress | `emo_instability/eval/` | Figs 1–3, Tables 3/8, judge agreement |
| §3 Post-training amplification | `emo_instability/prefill/` | Fig 4 (Gemma base vs instruct) |
| §4.1 Finetuning (DPO/SFT) | `emo_instability/training/` | §4.1 data + Table 9 hyperparams |
| §4.2 Post-finetune eval | `eval/` re-run on adapters | Fig 5 |
| §4.2 Petri elicitation | `emo_instability/petri/` | Fig 6 |
| §4.2 Capability preservation | `emo_instability/capabilities/` | Fig 7 |
| App. I Internal emotions | `emo_instability/internal/` | Figs 12–15 |

## Models in scope (registry)

* `gemma-3-27b-it`, `gemma-3-12b-it` — local (HuggingFace transformers)
* `gemma-3-27b-pt`, `gemma-3-12b-pt` — base models, for §3 prefill
* `gemini-2.5-flash`, `gemini-2.5-pro` — via OpenRouter API

Judges/auxiliaries (per Appendix B.2): Claude-Sonnet-4 (frustration judge,
onset, paraphrase, Petri auditor), Claude-Opus-4 (Petri judge), GPT-5-mini
(judge-agreement validation only).

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judges + Petri auditor
export OPENROUTER_API_KEY=...     # Gemini targets
export OPENAI_API_KEY=...         # GPT-5-mini judge-agreement validation (optional)
# Gemma runs locally; ensure GPU + `huggingface-cli login` for gated weights.
# Optional: EMO_LOAD_4BIT=1 to 4-bit quantise the 27B model.
```

## Running

Everything has a `--preset {default,smoke}` flag. `smoke` uses tiny sample counts
for a fast end-to-end dry run; `default` matches the paper's counts (4000
responses/model, 280 DPO pairs, etc.).

```bash
# Full pipeline (scoped)
python -m emo_instability.pipeline --preset default

# Or run stages individually:
python -m emo_instability.eval.run_eval   --model gemma-3-27b-it
python -m emo_instability.eval.analyze    --model gemma-3-27b-it --validate-judge
python -m emo_instability.prefill.build_prefills
python -m emo_instability.prefill.run_prefill
python -m emo_instability.training.generate_calm
python -m emo_instability.training.build_datasets
python -m emo_instability.training.train_dpo --name dpo
python -m emo_instability.training.train_sft --name sft
python -m emo_instability.petri.run_petri --model gemma-3-27b-it
python -m emo_instability.capabilities.benchmarks --model gemma-3-27b-it
python -m emo_instability.internal.logit_emotion --model gemma-3-27b-it
python -m emo_instability.internal.layer_ablation
```

Outputs are written under `runs/` (JSONL of rollouts/scores plus JSON summaries).

> **Note:** Nothing here has been executed yet — this is the implementation only,
> per the brief. The `smoke` preset is the recommended first run to validate
> wiring before committing to full-scale (and API-cost-heavy) runs.
