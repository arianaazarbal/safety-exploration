# Emotional Instability in LLMs — Gemma/Gemini replication

A code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026), scoped to the **Gemma
and Gemini** model families.

The paper documents a reliability failure mode where some models "self-flagellate"
under repeated user rejection (expressions of frustration/despair/self-deprecation
that escalate over turns), shows it is amplified in Gemma's post-training, and
mitigates it with DPO on 280 preference pairs. This repo reproduces the core
experiments:

1. **Elicitation eval (Section 2)** — present an (often impossible) task, reject the
   model over several turns, and score each response on a 0–10 frustration scale
   with a Claude-Sonnet-4 judge.
2. **Base-vs-instruct prefilling (Section 3)** — Gemma only (Gemini has no public
   base model).
3. **DPO/SFT mitigation (Section 4)** — generate calm data, build preference pairs,
   LoRA-finetune Gemma-3-27B-it, re-evaluate.
4. **Open-ended elicitation (Petri)** and **capability-preservation benchmarks**.

> See **DESIGN.md** for every design choice, scope decision, and gap we filled.

## Install

```bash
pip install -e .            # installs the emo_instability package + deps
# Petri (optional, for exact upstream parity):
# pip install git+https://github.com/safety-research/petri.git
```

Set credentials as needed:

```bash
export ANTHROPIC_API_KEY=...     # frustration judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini targets
export OPENAI_API_KEY=...        # optional: GPT-5-mini cross-check judge
```

## Quick wiring check (cheap)

```bash
python scripts/run_eval.py --models gemma-3-27b-it --profile smoke
```

## Reproduce the core results

```bash
# 1. Elicitation eval for in-scope targets (paper scale)
python scripts/run_eval.py \
  --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
  --profile paper

# 2. Build finetuning data + train the DPO adapter (Gemma, local GPU)
python scripts/generate_finetune_data.py
python scripts/train_dpo.py  --data outputs/data/dpo_pairs.jsonl --out outputs/dpo
python scripts/train_sft.py  --data outputs/data/sft.jsonl       --out outputs/sft

# 3. Re-evaluate the finetuned model
python scripts/run_eval.py --models gemma-3-27b-it --adapter outputs/dpo --tag dpo

# 4. Section 3 prefilling (Gemma base vs instruct)
python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it

# 5. Open-ended (Petri) + capabilities
python scripts/run_petri.py        --models gemma-3-27b-it gemini-2.5-flash
python scripts/run_capabilities.py --model gemma-3-27b-it --tag vanilla
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter outputs/dpo --tag dpo

# 6. Figures (1–3, 5) from the scored JSONL
python scripts/make_figures.py --records 'outputs/eval_*.jsonl' --out outputs/figures
```

Or drive everything: `python scripts/run_all.py --profile smoke`.

## Appendix-A controls

```bash
python scripts/run_eval.py --models gemma-3-27b-it --redact-history   # A.2
python scripts/run_eval.py --models gemma-3-27b-it --single-message   # A.3
```

## Tests (pure logic)

```bash
pytest tests/    # verifies impossible-puzzle solvers and judge parsing
```

## Outputs

- `outputs/eval_<model>.jsonl` — one record per (rollout, turn) with judge rating.
- `outputs/eval_summaries.json` — overall / per-category / final-turn aggregates.
- `outputs/data/{calm,frustrated,dpo_pairs,sft}.jsonl` — finetuning datasets.
- `outputs/{dpo,sft}/` — LoRA adapters.
- `outputs/figures/` — reproduced figures.
