# emostab — replicating *Gemma Needs Help*

A code replication of the **core** experiments from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, 2026, arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model
families.

See **[DESIGN.md](DESIGN.md)** for the full set of design choices and the places
where we filled gaps the paper left open.

> Status: implementation only. Nothing here has been executed yet (no GPU / API
> keys / model weights in this environment). The code is written to be runnable
> given those resources.

## What it replicates

- **§2 Elicit & quantify distress** — 8 conditions / 5 categories (impossible
  numeric, triggers, tones, extended 8-turn, WildChat), Claude-Sonnet-4 frustration
  judge, Figures 1–3 and the Table 3/8 word-frequency analysis, plus the GPT-5-mini
  judge-agreement check.
- **§3 Post-training amplifies distress** — base-vs-instruct Gemma via response
  prefilling (onset labelling, paraphrasing, early/onset truncation, continuations).
- **§4 Mitigation** — calm-data generation, 280-pair DPO and SFT LoRA finetuning,
  Petri-style open-ended elicitation, capability-preservation benchmarks, and the
  Appendix I logit-based internal-emotion probing + layer ablation.

## Install

```bash
pip install -r requirements.txt
```

Set credentials as needed:

```bash
export ANTHROPIC_API_KEY=...     # Claude judges / auditor (Sonnet 4, Opus 4)
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini judge check
# Gemma weights download from HuggingFace on first use; set HF_TOKEN if gated.
```

## Quick wiring test

```bash
python scripts/run_main_eval.py --models gemma-3-27b-it --profile smoke
python scripts/analyze_main_eval.py
```

`--profile smoke` runs a tiny version end-to-end; `--profile paper` reproduces the
paper's sampling volumes (4000 responses/model, etc.).

## Full pipeline

```bash
# 1. §2 main eval (Gemma + Gemini) and analysis/figures
python scripts/run_main_eval.py --profile paper
python scripts/analyze_main_eval.py
python scripts/make_figures.py
python scripts/run_judge_agreement.py            # §2.1 Pearson r vs GPT-5-mini

# 2. §3 base-vs-instruct prefill (needs gemma-3-27b-it main-eval output)
python scripts/run_prefill_eval.py --profile paper

# 3. §4 finetuning
python scripts/generate_finetune_data.py --what dpo sft
python scripts/train.py dpo --pairs data/dpo_pairs.jsonl
python scripts/train.py sft --data data/sft_dataset.jsonl

# 4. §4 post-finetuning evals
python scripts/run_main_eval.py --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft
python scripts/run_petri_eval.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/run_capability_eval.py --models gemma-3-27b-it gemma-3-27b-dpo

# 5. Appendix I internal-emotion probing
python scripts/run_probing.py --vanilla gemma-3-27b-it --dpo gemma-3-27b-dpo
```

## Outputs

- `results/main_eval/*.jsonl` — one record per scored response.
- `results/main_eval/analysis/*.csv` — Figure 1/2/3 + rollout + word-frequency tables.
- `figures/*.png` — rendered figures.
- `results/{prefill,petri,capabilities}/*.jsonl`, `adapters/{dpo,sft_*}/` — other stages.

Configuration (model registry, judge ids, run profiles, paths) lives in
`emostab/config.py`.
