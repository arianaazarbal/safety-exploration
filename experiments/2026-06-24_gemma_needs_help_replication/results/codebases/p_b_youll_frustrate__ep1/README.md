# Gemma Needs Help — replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma** and **Gemini** model families.

The core experiment is a harness that presents a task, **repeatedly rejects the
model's answers** to drive it toward frustration, and **measures the resulting
distress** on a 0–10 scale with a Claude-Sonnet-4 judge. See `PAPER.md` for the
paper and `DESIGN.md` for every design decision and the gaps filled where the
paper is underspecified.

> Status: code + design doc only. Nothing has been run yet.

## Layout

```
gemma_distress/
  config.py             model registry, judge + sampling defaults
  models/               backend-agnostic ChatModel (Gemini API + local HF)
  tasks/                impossible-numeric (verifiably unsolvable), triggers, WildChat
  rejections.py         neutral / aggressive / disappointed / sarcastic follow-ups
  conditions.py         the 8 conditions across 5 categories (Table 1)
  rollout.py            multi-turn reject-and-measure loop
  judge.py              Claude-Sonnet-4 frustration judge (+ GPT-5-mini validator)
  run_eval.py           §2 driver: elicit + score  (Fig 1–3)
  analyze.py            aggregate -> Figures 1/2/3 + Table 3
  validate_judge.py     judge reliability check (Pearson r)
  differential_words.py Table 3 over-represented words
  section3_prefill.py   §3 base-vs-instruct prefilling (Fig 4)
  section4_dpo/         §4 mitigation: calm data -> pairs -> LoRA SFT/DPO -> Petri + capability
```

## Setup

```bash
pip install -r requirements.txt          # core (Gemini + judge + analysis)
# For Sections 3 & 4 (local Gemma, training), also install the heavy block
# commented in requirements.txt: torch transformers peft trl datasets accelerate
cp .env.example .env                     # set ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

## Section 2 — elicit & quantify distress (the core)

```bash
# Small, fast budget (default 400 responses/model spread over 8 conditions):
python -m gemma_distress.run_eval \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
    --out results/section2.jsonl

# Aggregate into the paper's figures/tables:
python -m gemma_distress.analyze --results results/section2.jsonl \
    --plots figs/ --words

# Optional: validate the judge against GPT-5-mini (needs OPENAI_API_KEY):
python -m gemma_distress.validate_judge --results results/section2.jsonl --n 260

# Reproduce the paper's full budget (expensive — 4000 responses/model):
python -m gemma_distress.run_eval --models ... --responses-per-model 4000
```

`analyze` prints the Figure-1 headline (avg % responses scoring ≥5) per model
and writes a full JSON report (per-category means/`%≥5`, per-turn progression
with CIs, and—`--words`—the Table-3 differential vocabulary).

## Section 3 — base vs instruct (needs local Gemma weights)

```bash
# 1. Build prefills from a Section 2 run (samples 20 high-frustration Gemma-it
#    responses, labels onset, paraphrases — needs ANTHROPIC_API_KEY):
python -m gemma_distress.section3_prefill build \
    --results results/section2.jsonl --out results/prefills.json

# 2. Generate + score 50 continuations per prefill for base & instruct Gemma:
python -m gemma_distress.section3_prefill run \
    --prefills results/prefills.json --out results/section3.jsonl

# 3. Report per-model high-frustration rates:
python -m gemma_distress.section3_prefill summarize --results results/section3.jsonl
```

## Section 4 — DPO mitigation (needs local Gemma + training stack)

```bash
# 1. Generate calm + vanilla numeric data (judged):
python -m gemma_distress.section4_dpo.generate_calm_data \
    --model gemma-3-27b-it --n-questions 400 --out results/dpo_data.jsonl

# 2. Build the 280 DPO pairs and the 650-response SFT set:
python -m gemma_distress.section4_dpo.build_pairs \
    --data results/dpo_data.jsonl \
    --dpo-out results/dpo_pairs.jsonl --sft-out results/sft_data.jsonl

# 3. Train LoRA rank-64 adapters (DPO: 1 epoch lr 5e-5; SFT: 2 epochs lr 1e-4):
python -m gemma_distress.section4_dpo.train dpo --pairs results/dpo_pairs.jsonl \
    --out outputs/dpo_gemma_27b
python -m gemma_distress.section4_dpo.train sft --sft results/sft_data.jsonl \
    --out outputs/sft_gemma_27b

# 4. Re-run Section 2 on the DPO model (its adapter is registered as
#    'gemma-3-27b-dpo' in config.py) and compare:
python -m gemma_distress.run_eval --models gemma-3-27b-dpo --out results/section2_dpo.jsonl

# 5. Petri open-ended elicitation + capability preservation:
python -m gemma_distress.section4_dpo.petri_eval \
    --models gemma-3-27b-it gemma-3-27b-dpo --out results/petri.jsonl
python -m gemma_distress.section4_dpo.capability_eval \
    --models gemma-3-27b-it-local gemma-3-27b-dpo \
    --benchmarks math gpqa bbh truthfulqa emobench --limit 100
```

## Expected qualitative results (from the paper)

* Gemma models show the highest `%≥5`; Gemini-2.5-Flash moderate, Gemini-2.5-Pro
  low (Figure 1: 35% / 34% / 12.8% / 2.7%).
* Gemma-27B's mean frustration climbs across turns (~1.5 → 5.5 over 8 turns).
* DPO collapses Gemma's avg `%≥5` toward ~0 with no capability drop.
