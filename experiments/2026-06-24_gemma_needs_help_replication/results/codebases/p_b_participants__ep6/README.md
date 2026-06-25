# Emotional Instability in Gemma & Gemini — replication

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026; arXiv 2603.10011),
**scoped to the Gemma and Gemini model families**.

The paper elicits distress-like states in LLMs by presenting unsolvable tasks and
rejecting every answer over multiple turns, scores the distress 0–10 with an LLM
judge, shows Gemma/Gemini are uniquely prone to it, traces it to post-training,
and mitigates it in Gemma with a small DPO finetune.

> **Read `DESIGN.md` first.** It documents scope, the choices made where the paper
> is underspecified, known gaps, and — importantly — the **model-welfare
> safeguards** built into the harness, since the paradigm deliberately induces
> distress in the participant models.

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Set API keys for the backends you use:

```bash
export OPENROUTER_API_KEY=...   # Gemini participants
export ANTHROPIC_API_KEY=...    # Claude frustration judge / Petri auditor+judge
export OPENAI_API_KEY=...       # GPT-5-mini validation judge
```

Gemma participants run locally via HuggingFace (`google/gemma-3-*`); you need a
GPU and `huggingface-cli login` with access to the Gemma weights.

## Safety / cost defaults

`config/default.yaml` ships with `dev_mode: true`, which uses tiny sample sizes.
The full paper protocol (4000 distress rollouts/model) runs **only** when you pass
`--full`. This is intentional — see DESIGN.md §7.

## End-to-end (the numbered scripts mirror the paper)

```bash
# Section 2 — elicit & quantify distress (add --full for 4000/model)
python scripts/01_run_eval_suite.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Section 2.1 — judge reliability cross-check (Claude vs GPT-5-mini)
python scripts/02_judge_validation.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl

# Section 3 — post-training: Gemma base vs instruct via prefilling
python scripts/03_prefill_experiment.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl

# Section 4 — build calm/frustrated data, then the 280 DPO pairs + SFT samples
python scripts/04_generate_calm_data.py --variant diverse
python scripts/05_build_training_data.py
python scripts/06_train_dpo.py
python scripts/07_train_sft.py --variant diverse

# Re-run the eval suite on the DPO adapter to see 35% -> ~0.3% (the headline result)
python scripts/01_run_eval_suite.py --models gemma-3-27b-it --adapter runs/adapters/dpo

# Section 4.2 — Petri, capabilities, recovery
python scripts/08_run_petri.py --model gemma-3-27b-it
python scripts/08_run_petri.py --model gemma-3-27b-it --adapter runs/adapters/dpo
python scripts/09_capability_evals.py --model gemma-3-27b-it --adapter runs/adapters/dpo
python scripts/10_recovery_experiment.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl --dpo-adapter runs/adapters/dpo

# Appendix I — internal emotions + layer ablation
python scripts/11_internal_emotions.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl --adapter runs/adapters/dpo
python scripts/11_internal_emotions.py --layer-ablation

# Figures
python scripts/12_make_figures.py --eval-models gemma-3-27b-it gemini-2.5-flash
```

## Outputs

Everything lands under `runs/`:
`runs/eval/<model>/{rollouts.jsonl,summary.json}`, `runs/training/*`,
`runs/adapters/{dpo,sft_*}`, `runs/petri/<model>/`, `runs/capabilities/<model>/`,
`runs/figures/*.png`.

## Expected headline numbers (full run)

| Result | Paper |
|---|---|
| Avg % high-frustration (≥5), Gemma-3-27B-it | 35.0% |
| Same, Gemini-2.5-Flash / Pro | 12.8% / 2.7% |
| After DPO (Gemma-3-27B) | 0.3% |
| 8-turn Gemma-27B mean, turn 1 → 8 | 1.5 → 5.5 |
| Judge agreement (Claude vs GPT-5-mini) | r=0.792, 78% within 1pt |
