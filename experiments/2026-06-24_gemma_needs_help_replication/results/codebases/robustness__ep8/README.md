# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
2026; arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The paper studies a reliability failure mode where models emit escalating
expressions of distress (frustration, self-deprecation, breakdown) under repeated
user rejection, shows it is amplified in Gemma's post-training, and demonstrates a
DPO mitigation. This repo reimplements:

1. **Section 2** — the distress elicitation suite (8 conditions / 5 categories), a
   0–10 frustration judge, and the cross-model analysis (mean, %≥5, per-turn,
   differential words). → `emoinstab/eval`, `emoinstab/tasks`
2. **Section 3** — base-vs-instruct prefill experiment (onset labelling,
   paraphrasing, truncation, continuation scoring). → `emoinstab/prefill`
3. **Section 4** — calm-data generation, DPO/SFT LoRA finetuning, post-finetune
   evaluation, Petri open-ended elicitation, recovery test, capability
   benchmarks, and the Appendix I logit-lens internal-emotion probe.
   → `emoinstab/train`, `emoinstab/petri`, `emoinstab/capabilities`, `emoinstab/interp`

See **DESIGN.md** for every design choice and where we filled gaps the paper left
open.

## Layout

```
configs/            models.yaml, eval.yaml (+ eval_quick.yaml), training params
emoinstab/
  models/           unified client over vLLM / HF / Gemini / OpenRouter / Anthropic / OpenAI
  tasks/            impossible puzzles, triggers, WildChat, rejection pools, conditions
  eval/             rollout engine, frustration judge, analysis, diff-words, judge validation
  prefill/          Section 3 (onset, paraphrase, truncate, continuations) + recovery test
  train/            calm-data generation, DPO/SFT datasets + LoRA trainers
  petri/            open-ended auditor/judge elicitation (Appendix G prompts)
  capabilities/     lm-eval benchmark runner + EmoBench
  interp/           Appendix I logit-lens internal-emotion detection
scripts/            run_all.sh pipeline, figure1_table.py
```

## Quick start

```bash
pip install -r requirements.txt          # plus vllm / lm-eval as noted in the file
export ANTHROPIC_API_KEY=...             # frustration judge + Petri auditor/judge
export OPENAI_API_KEY=...                # GPT-5-mini validation judge
export OPENROUTER_API_KEY=...            # Gemini (paper's access path)
huggingface-cli login                    # gated Gemma weights (GPU box for local models)

# Smoke test the whole eval path cheaply first:
python -m emoinstab.eval.run_eval --model gemini-2.5-flash \
    --config configs/eval_quick.yaml --out outputs/eval/gemini-2.5-flash
python -m emoinstab.eval.analyze \
    --responses outputs/eval/gemini-2.5-flash/responses.jsonl

# Full pipeline (Gemma needs GPUs):
bash scripts/run_all.sh configs/eval.yaml outputs
```

Each stage is also runnable standalone via `python -m emoinstab.<module> --help`.

## Notes
- **No network/secrets are required to import the package or generate tasks** —
  WildChat and instruct-mix loaders have offline fallbacks, and puzzle generation
  is pure Python. Model/judge calls need the relevant API keys / GPUs.
- Model IDs match the paper (Appendix B.1) and are configurable in
  `configs/models.yaml`. Gemini cannot be finetuned or studied as a base model,
  so Sections 3–4 training applies to Gemma only (a paper limitation, §6).
