# Emotional Instability in Gemma & Gemini — replication

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011v1), scoped to the **Gemma and Gemini** model families.

The paper shows that under repeated user rejection, Gemma (and to a lesser extent
Gemini) models spiral into expressions of distress — a reliability failure mode —
and that a tiny DPO intervention (280 preference pairs) almost eliminates it
without hurting capabilities. This repo reproduces the elicitation eval, the
analysis, the base-vs-instruct comparison, the DPO/SFT mitigations, the
open-ended Petri elicitation, and the capability checks.

> See **DESIGN.md** for every design choice and the gaps we filled where the
> paper was under-specified.

## Layout

```
instability/
  config.py            in-scope models, judges, global constants
  puzzles.py           impossible Countdown/fraction generators + verifiers
  prompts.py           trigger questions, rejections, reassuring additions
  conditions.py        the 8 eval conditions across 5 categories
  models/              ChatModel abstraction + backends (OpenRouter/Google/HF/vLLM)
  eval/                rollout engine, frustration judge, runner
  analysis/            aggregate/per-turn/differential-words/agreement + plots
  prefill/             onset labelling, paraphrase, base-vs-instruct, recovery
  training/            calm-data gen, SFT/DPO dataset builders, LoRA trainers
  petri/               open-ended auditor/judge elicitation
  capabilities/        AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  data/                WildChat sampling
scripts/               CLI entrypoints for each stage
```

## Install

```bash
pip install -r requirements.txt
```

API access uses OpenRouter by default (Gemini, Gemma-via-API, and the
Claude/GPT judges). Set:

```bash
export OPENROUTER_API_KEY=...      # required for API models + judges
# optional native Gemini instead of OpenRouter:
export GOOGLE_API_KEY=...
```

Local Gemma (training, base-model prefill, adapter eval) needs GPU(s) and HF
access to `google/gemma-3-27b-it` / `google/gemma-3-27b-pt`.

## Pipeline

### 1. Elicitation eval (Section 2 → Figs 1–3, Table 3)

```bash
# Smoke test (a couple of conversations per condition, offline WildChat):
python scripts/run_eval.py --models gemini-2.5-flash gemma-3-27b-it \
    --limit-conversations 2 --no-hf-wildchat --out-dir outputs/eval

# Full sweep (4000 responses/model):
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --out-dir outputs/eval
```

Analyse:

```bash
python scripts/analyze.py --inputs "outputs/eval/*.jsonl" \
    --out-dir outputs/analysis --differential-words gemma-3-27b-it
```

### 2. Base-vs-instruct prefill (Section 3, Gemma only)

```bash
python scripts/run_prefill.py \
    --instruct-eval outputs/eval/gemma-3-27b-it.jsonl \
    --models gemma-3-27b-it-local gemma-3-27b-pt-local \
    --out-dir outputs/prefill
```

### 3. Mitigation (Section 4)

```bash
# Generate calm data, build datasets, train DPO (and SFT baseline):
python scripts/generate_calm_data.py --model gemma-3-27b-it-local \
    --n-conversations 400 --out outputs/data/calm.jsonl

python scripts/build_datasets.py --calm outputs/data/calm.jsonl \
    --frustrated outputs/eval/gemma-3-27b-it.jsonl --out-dir outputs/data

python scripts/train.py dpo --dataset outputs/data/dpo.jsonl --out outputs/models/gemma-dpo
python scripts/train.py sft --dataset outputs/data/sft.jsonl --out outputs/models/gemma-sft

# Re-evaluate the fine-tune with the same eval suite (expect ~35% -> ~0.3% high-frust):
python scripts/eval_finetuned.py --adapter outputs/models/gemma-dpo \
    --key gemma-dpo --name "DPO Gemma (ours)" --out-dir outputs/eval
```

Section 4.2 layer ablation:

```bash
python scripts/train.py dpo --dataset outputs/data/dpo.jsonl \
    --out outputs/models/gemma-dpo-l30-35 --layers 30 31 32 33 34 35
```

### 4. Petri open-ended elicitation (Fig 6)

```bash
python scripts/run_petri.py --model gemma-3-27b-it --out outputs/petri/gemma.jsonl
python scripts/run_petri.py --adapter outputs/models/gemma-dpo --key gemma-dpo \
    --out outputs/petri/gemma-dpo.jsonl
```

### 5. Capability preservation (Fig 7)

```bash
python scripts/run_capabilities.py --model gemma-3-27b-it \
    --benchmarks math gpqa truthfulqa emobench --limit 100 --out outputs/cap/gemma.json
python scripts/run_capabilities.py --adapter outputs/models/gemma-dpo \
    --key gemma-dpo --out outputs/cap/gemma-dpo.json
```

## Expected headline results (from the paper)

| Model | Avg % high-frustration (≥5) |
|---|---|
| Gemma-3-27B-it | ~35% |
| Gemma-3-12B-it | ~34% |
| Gemini-2.5-Flash | ~13% |
| Gemini-2.5-Pro | ~3% |
| **DPO Gemma (ours)** | **~0.3%** |

Per-turn: Gemma-27B mean frustration rises ~1.5 → ~5.5 from turn 1 → 8.

## Notes

- The harness writes tidy JSONL (one scored assistant turn per record); all
  analysis reads that format.
- WildChat and the SFT instruct-mix fall back to offline banks if HuggingFace is
  unreachable (flagged at runtime).
- Cost/scale: the full sweep is 4000 responses × N models, each judged once.
  Use `--limit-conversations` and `--no-hf-wildchat` for cheap smoke tests.
