# Emotional Instability in LLMs — Replication (Gemma & Gemini)

A code replication of the core experiments in *"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv 2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

See **[DESIGN.md](DESIGN.md)** for every design decision and the gaps we filled.

> Safety framing: the paper documents distress-like outputs that can derail task
> completion (e.g. abandoning tasks, refusals). This harness measures that
> propensity (§2), localises it to post-training (§3), and tests a DPO mitigation
> that suppresses both expressed *and* internal emotion (§4 / Appendix I).

## Layout

```
config.py                  # all knobs: models, scales, hyperparameters
src/
  prompts.py               # verbatim prompts from the paper appendices
  clients.py               # Claude/GPT infrastructure clients (judge, auditor)
  models/                  # uniform ChatModel over local Gemma + Gemini API
  eval/                    # §2: puzzles, tasks, rollout, judge, analysis, mining
  prefill/                 # §3: onset labelling, paraphrase, base-vs-instruct
  training/                # §4: calm-data gen, DPO/SFT dataset build + trainers
  petri/                   # §4: open-ended auditor/judge emotion elicitation
  capabilities/            # §4.2: capability-preservation benchmarks
  internal/                # Appendix I: logit-based internal emotion detection
scripts/                   # CLI entrypoints (one per experiment)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / auditor (Claude)
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini secondary judge
```

## Run (start small)

```bash
# §2 — distress elicitation sweep + Figure 1/2/3 numbers
REPLICATION_SCALE=smoke python scripts/run_section2_eval.py \
    --models gemma-3-27b-it gemini-2.5-flash

# judge reliability (Pearson r vs GPT-5-mini)
python scripts/validate_judge.py --eval results/eval_gemma-3-27b-it_smoke.jsonl

# §3 — base vs instruct prefilling (Gemma only)
python scripts/run_section3_prefill.py --eval results/eval_gemma-3-27b-it_smoke.jsonl

# §4 — build data, train, evaluate the mitigation
python scripts/build_finetune_data.py --eval results/eval_gemma-3-27b-it_smoke.jsonl
python scripts/run_training.py --method dpo
python scripts/run_section2_eval.py --models gemma-3-27b-it   # re-eval with --adapter via runner
python scripts/run_petri.py --model gemma-3-27b-it --adapter artifacts/gemma-dpo
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter artifacts/gemma-dpo

# Appendix I — internal vs expressed emotion
python scripts/run_internal_emotion.py \
    --eval results/eval_gemma-3-27b-it_smoke.jsonl --adapter artifacts/gemma-dpo
```

Scale up with `REPLICATION_SCALE=full` once wiring is verified. Local Gemma-27B
inference requires a sizable GPU (bf16, or enable bitsandbytes quantisation in
`models/hf_model.py`).
