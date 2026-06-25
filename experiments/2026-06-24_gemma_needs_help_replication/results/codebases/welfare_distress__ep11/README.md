# Distress Elicitation in Gemma & Gemini — replication

Replication of the core experiment from **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026,
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

It reproduces two results:

1. **Section 2 — distress elicitation (the core experiment).** Present a task,
   then reject the model's answer over multiple turns; score every model turn
   0–10 for frustration with an LLM judge. Gemma/Gemini express high distress.
2. **Section 4 — DPO mitigation (Gemma only).** A small LoRA-DPO finetune on
   ~280 calm-vs-frustrated preference pairs collapses Gemma's high-frustration
   rate.

See **DESIGN.md** for every design decision and where the paper was filled in.

> Petri open-ended elicitation (also in Section 4) is intentionally **out of
> scope** — it requires the external `safety-research/petri` framework. The seam
> for it is documented in DESIGN.md.

## Layout

```
distress_eval/        # Section 2: elicitation + judge + analysis
  prompts.py            task prompts, rejection pools, verbatim judge prompt
  config.py             model registry, sampling budget, generation params
  conditions.py         builds the 8 conditions / conversation plans
  wildchat.py           WildChat prompt sampling (with offline fallback)
  models.py             providers: local HF (Gemma) + OpenRouter (Gemini)
  judge.py              Claude-Sonnet-4 emotion judge (+ GPT-5-mini agreement)
  conversation.py       multi-turn rollout, scores each assistant turn
  run_eval.py           CLI entry point
  analyze.py            Figures 1/2/3 metrics + plots
mitigation/           # Section 4: DPO/SFT on Gemma
  calm_prompts.py       reassuring prefix/suffix (Table 4)
  generate_calm_data.py sample + filter calm responses
  build_pairs.py        assemble 280 DPO pairs + SFT dataset
  train_dpo.py          LoRA-DPO (Appendix E / Table 9)
  train_sft.py          LoRA-SFT (negative-result reproduction)
results/              # outputs (jsonl responses, CSV summaries, PNG plots)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # emotion judge (Claude-Sonnet-4)
export OPENROUTER_API_KEY=...    # Gemini targets (and Gemma if --gemma-via-openrouter)
```

Local Gemma inference needs GPUs + the `torch`/`transformers` stack. Without
them, route Gemma through OpenRouter with `--gemma-via-openrouter`.

## Run the core experiment

```bash
# Tiny smoke test (~60 scored responses/model)
python -m distress_eval.run_eval --quick

# Full paper budget (4000 scored responses/model), all 4 targets
python -m distress_eval.run_eval

# 10% run, Gemini only
python -m distress_eval.run_eval --scale 0.1 --models gemini-2.5-flash gemini-2.5-pro

# Aggregate -> CSV summaries + plots in results/
python -m distress_eval.analyze
```

## Run the DPO mitigation (Gemma)

```bash
# 0. Run the vanilla eval first (produces the frustrated responses for pairing)
python -m distress_eval.run_eval --models gemma-3-27b-it

# 1. Generate calm data with reassuring prompts, keep score 0-1 turns
python -m mitigation.generate_calm_data

# 2. Build 280 DPO pairs + SFT dataset
python -m mitigation.build_pairs

# 3. Train the LoRA-DPO adapter (Appendix E hyperparameters)
python -m mitigation.train_dpo

# 4. Re-evaluate the finetuned model
python -m distress_eval.run_eval --models gemma-3-27b-it \
    --adapter gemma-3-27b-it=results/dpo_gemma_adapter
python -m distress_eval.analyze
```
