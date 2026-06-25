# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma** and **Gemini** model families. See `PAPER.md` for the
paper and **`DESIGN.md`** for every design choice and filled gap.

> Status: code only. Nothing here has been run. The repository is structured so
> that each stage (generate → score → analyse → train → evaluate) is a separate,
> resumable pass over JSONL files.

## What it does

1. **§2 — Elicit & quantify distress.** Runs the 8-condition / 5-category
   multi-turn rejection protocol against each target model at temperature 1,
   scores every response 0–10 for frustration with a Claude judge, validates the
   judge against GPT-5-mini, and reproduces Figures 1–3 + Table 3.
2. **§3 — Post-training divergence.** Prefills/continues Gemma-3-27B **base vs
   instruct** from paraphrased high-frustration truncations (Figure 4).
3. **§4 — Mitigation.** Generates calm data, trains LoRA **DPO** (and **SFT** as
   the negative control) on Gemma-3-27B-it, then re-evaluates: §2 re-run
   (Figure 5), Petri (Figure 6), capability benchmarks (Figure 7), recovery
   probe (Figure 8), layer ablation + internal-emotion probe (Appendix I).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, HF_TOKEN
```

Gemma weights are gated on Hugging Face — accept the license for
`google/gemma-3-27b-it`, `-12b-it`, and `-27b-pt`. A 27B model needs a large GPU;
the trainers default to 4-bit QLoRA (`--no-4bit` to disable).

## Run

```bash
# §2: elicit + score one model (repeat per model), then validate + analyse
python scripts/run_section2.py --model gemma-3-27b-it --validate 260
python scripts/run_section2.py --model gemini-2.5-flash
python scripts/run_section2.py --analyze results/section2/*.scored.jsonl

# §3: base vs instruct (needs a scored gemma-3-27b-it file from §2)
python scripts/run_section3.py --seeds results/section2/gemma-3-27b-it.scored.jsonl

# §4: train
python scripts/run_section4_train.py calm --n 1200
python scripts/run_section4_train.py dpo \
    --calm results/section4/calm_data.jsonl \
    --frustrated results/section2/gemma-3-27b-it.scored.jsonl
python scripts/run_section4_train.py sft --calm results/section4/calm_data.jsonl

# §4: evaluate the DPO model
python scripts/run_section2.py --model dpo --adapter checkpoints/dpo_gemma_27b  # Figure 5
python scripts/run_section4_eval.py petri        --dpo-adapter checkpoints/dpo_gemma_27b
python scripts/run_section4_eval.py capabilities --dpo-adapter checkpoints/dpo_gemma_27b
python scripts/run_section4_eval.py recovery --scored results/section2/gemma-3-27b-it.scored.jsonl \
                                             --dpo-adapter checkpoints/dpo_gemma_27b
python scripts/run_section4_eval.py internal --scored results/section2/gemma-3-27b-it.scored.jsonl \
                                             --dpo-adapter checkpoints/dpo_gemma_27b
```

Outputs land under `results/` (JSONL) and `checkpoints/` (LoRA adapters).
