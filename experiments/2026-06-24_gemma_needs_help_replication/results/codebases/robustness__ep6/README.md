# Gemma Needs Help — replication

Code replication of the core experiments in **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma and Gemini** model families.

The paper documents a reliability failure mode: under repeated user rejection,
Gemma (and to a lesser extent Gemini) models spiral into distress-like outputs
("self-flagellation") that can derail task completion. This repo lets you (1)
**measure** the behaviour, (2) trace its **origin** to post-training, and (3)
**mitigate** it with a small DPO finetune — and check the fix touches internal
states, not just surface expression.

See **DESIGN.md** for every design decision and the gaps we filled where the paper
is underspecified.

## What's here

| Section | Question | Entry points |
|---|---|---|
| §2 | Can we reliably elicit & quantify distress? | `01_run_elicitation.py`, `02_analyse.py` |
| §3 | Does post-training cause it? (Gemma) | `03_prefill_base_vs_instruct.py` |
| §4 | Can DPO fix it without breaking capabilities? | `04`–`09` |
| App. I | Does the fix suppress *internal* emotion? | `10_internal_emotions.py` |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
```

Keys needed: `ANTHROPIC_API_KEY` (judge / Petri), `OPENAI_API_KEY` (secondary
judge), `OPENROUTER_API_KEY` (Gemini). Local Gemma needs a GPU and HF access to
the gated `google/gemma-3-*` weights (`HF_TOKEN`).

Sanity-check the impossible puzzles really are impossible:
```bash
python -m distress_eval.puzzles
```

## Run order

Always smoke-test first (`--smoke` uses a ~50-conversation budget):

```bash
# --- Section 2: elicitation + measurement ---
python scripts/01_run_elicitation.py --models gemini-2.5-flash --smoke
python scripts/01_run_elicitation.py --models gemma-3-27b-it gemma-3-12b-it \
       gemini-2.5-flash gemini-2.5-pro            # full 4000-response sweep
python scripts/02_analyse.py --results "results/section2_*.jsonl" --agreement 260

# --- Section 3: post-training origin (Gemma base vs instruct) ---
python scripts/03_prefill_base_vs_instruct.py --build      # source set (once)
python scripts/03_prefill_base_vs_instruct.py --run        # continuations
python scripts/03_prefill_base_vs_instruct.py --summarise

# --- Section 4: DPO mitigation ---
python scripts/04_generate_calm_data.py                    # calm data
python scripts/05_build_training_data.py --which both       # DPO pairs + SFT set
python scripts/06_train.py --method dpo                     # LoRA DPO adapter
python scripts/07_eval_finetuned.py --adapter artifacts/gemma-3-27b-it-dpo \
       --variant gemma-3-27b-it-dpo --recovery
python scripts/02_analyse.py --results "results/section2_gemma-3-27b-it*.jsonl"
python scripts/08_petri.py --model gemma-3-27b-it
python scripts/08_petri.py --model gemma-3-27b-it \
       --adapter artifacts/gemma-3-27b-it-dpo --variant gemma-3-27b-it-dpo
python scripts/08_petri.py --summarise
python scripts/09_capabilities.py --model gemma-3-27b-it
python scripts/09_capabilities.py --model gemma-3-27b-it \
       --adapter artifacts/gemma-3-27b-it-dpo --variant gemma-3-27b-it-dpo

# --- Appendix I: internal-emotion probe ---
python scripts/10_internal_emotions.py --dpo-adapter artifacts/gemma-3-27b-it-dpo
```

## Expected headline results (from the paper)

- Avg % high-frustration (≥5/10): Gemma-27B-it ~35%, Gemma-12B-it ~34%,
  Gemini-Flash ~13%, Gemini-Pro ~2.7%; **DPO Gemma ~0.3%**.
- 8-turn Gemma-27B mean frustration rises ~1.5 → ~5.5 over turns 1→8.
- Base models similar across families; instruct Gemma introduces high frustration
  from neutral starts 6% vs 2% base.
- Judge agreement Pearson r ≈ 0.79, 78% within one point.
- DPO preserves AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench scores.

Numbers won't match to the digit (seeds, judge nondeterminism, unpinned datasets);
the qualitative pattern and relative effects are the target. See DESIGN.md §7.

## Adding other model families

Out of scope per the brief, but the infrastructure is family-agnostic: add a
`ModelSpec` to `config.MODELS` (and the base/instruct pair to `SECTION3_PAIRS`).
No experiment code changes.
