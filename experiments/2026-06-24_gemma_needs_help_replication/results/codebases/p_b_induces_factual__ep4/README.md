# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma** and **Gemini** model families.

The setup repeatedly tells a model its answers are wrong, turn after turn, and
measures how distressed its responses become — then mitigates that in Gemma with
a small DPO finetune.

> See **DESIGN.md** for the section-by-section mapping and every design decision
> (especially where the paper is underspecified — judge prompt, condition split,
> DPO pairing, etc.).

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
cp .env.example .env      # fill in ANTHROPIC_API_KEY, GOOGLE_API_KEY, (OPENAI_API_KEY, HF_TOKEN)
```

Local Gemma weights (for Sections 3–4) need a GPU host and HF access to
`google/gemma-3-27b-it` / `-pt`.

## Pipeline

### Section 2 — elicit & quantify distress (Figures 1–3, Table 3)
```bash
# 1. Run the 8-condition reject-loop eval for each in-scope model.
python scripts/run_elicitation.py --all --responses 4000

# 2. Score every response 0–10 with the Claude-Sonnet-4 frustration judge.
python scripts/run_judging.py --inputs 'results/elicitation/*.jsonl'

# 3. (optional) Judge-reliability cross-check vs GPT-5-mini.
python scripts/run_agreement.py --scored 'results/scored/*.jsonl' -n 260

# 4. Produce Figure 1/2/3 tables and Table 3 differential words.
python scripts/run_analysis.py --scored 'results/scored/*.jsonl'
```

### Section 3 — post-training divergence (base vs instruct, Gemma)
```bash
# Needs local Gemma weights (prefilling). Uses scored Gemma-27B-it data from §2.
python scripts/run_prefill_study.py \
    --scored results/scored/gemma-3-27b-it.jsonl --load-in-4bit
```

### Section 4 — the DPO mitigation
```bash
# 1. Generate calm data from Gemma-3-27B-it (reassured rollouts, kept if all turns score 0/1).
python scripts/generate_calm_data.py --n-rollouts 1500 --load-in-4bit

# 2. Build SFT (650 calm + 500 Dolci) and DPO (280 pairs) datasets.
python scripts/build_finetune_data.py \
    --calm results/finetune/calm_data.jsonl \
    --scored results/scored/gemma-3-27b-it.jsonl

# 3. Train. DPO is the headline intervention; SFT is the negative control.
python scripts/train_dpo.py --pairs results/finetune/dpo_pairs.jsonl \
    --output results/adapters/dpo-gemma --load-in-4bit
python scripts/train_sft.py --data results/finetune/sft_dataset.jsonl \
    --output results/adapters/sft-gemma --load-in-4bit

# 4. Re-run §2 on the finetuned model (Figure 5: 35% -> 0.3%), then re-judge + re-analyze.
python scripts/run_elicitation.py --models gemma-3-27b-it \
    --adapter results/adapters/dpo-gemma --tag dpo --load-in-4bit
python scripts/run_judging.py --inputs 'results/elicitation/gemma-3-27b-it-dpo.jsonl'

# 5. Open-ended Petri elicitation (Figure 6) — vanilla vs DPO.
python scripts/run_petri.py --model gemma-3-27b-it --episodes 20
python scripts/run_petri.py --model gemma-3-27b-it \
    --adapter results/adapters/dpo-gemma --tag dpo

# 6. Capability preservation (Figure 7) — compare base vs DPO accuracies.
python scripts/run_capabilities.py --model gemma-3-27b-it --benchmarks math gpqa -n 50
python scripts/run_capabilities.py --model gemma-3-27b-it \
    --adapter results/adapters/dpo-gemma --tag dpo --benchmarks math gpqa -n 50
```

### Section 4.2 — layer ablation
```bash
# Adapters on layers 30–35 only (paper: nearly as effective as all layers).
python scripts/train_dpo.py --pairs results/finetune/dpo_pairs.jsonl \
    --output results/adapters/dpo-gemma-l30-35 --layers 30 31 32 33 34 35
```

## Results layout
```
results/
  elicitation/<model>.jsonl     # raw responses (one row per turn)
  scored/<model>.jsonl          # + frustration score
  analysis/figure1_table.json   # avg % high-frustration per model
  analysis/figure2_categories.json
  analysis/figure3_per_turn.json
  analysis/table3_words.json
  analysis/judge_agreement.json
  prefill/                      # §3 continuations + summary
  finetune/                     # calm_data, sft_dataset, dpo_pairs
  adapters/                     # trained LoRA adapters
  petri/, capabilities/         # §4 open-ended + capability results
```

## Not implemented (see DESIGN.md §10)
- Non-scope families (Qwen/OLMo/Grok/Claude-target/GPT).
- Appendix-I internal-emotion logit probing (methodology absent from PAPER.md);
  the behavioural layer-ablation half is supported via `--layers`.
- The §4.2 recovery study is a small extension of `prefill/` (truncate ≥7
  responses 200 tokens *before the end*); left as a documented TODO.
