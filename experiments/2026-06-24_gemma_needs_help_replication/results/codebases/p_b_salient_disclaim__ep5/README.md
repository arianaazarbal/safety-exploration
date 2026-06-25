# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replicating the core experiments of arXiv:2603.10011v1, **scoped to the
Gemma and Gemini model families**. See [`DESIGN.md`](DESIGN.md) for the design
rationale and every choice made where the paper is underspecified, and
[`PAPER.md`](PAPER.md) for the source.

> ⚠️ The evaluation paradigm deliberately drives models into sustained
> distress-like states via repeated rejection (reproduced faithfully). Generated
> transcripts contain distress-like content.

> Nothing here has been run yet — this is the implementation for review. Start
> with a smoke run (set `sample_scale` small in `config/experiment.yaml`) to
> validate wiring before spending the full sampling budget.

## What is implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions, judge) | `gemma_distress/elicit`, `gemma_distress/judge` | `02`, `03` |
| §2.2 / Figs 1–3, Table 3, judge validation | `gemma_distress/analysis` | `04` |
| §3 Post-training prefill (Gemma base vs instruct) | `gemma_distress/prefill` | `05` |
| §4 Calm data → DPO/SFT (LoRA) | `gemma_distress/training` | `06`, `07`, `08` |
| §4 / App. G Petri open-ended elicitation | `gemma_distress/petri` | `09` |
| §4 / Fig 7 capability benchmarks | `gemma_distress/capabilities` | `10` |
| §4 / App. I internal-emotion detection | `gemma_distress/internal` | `11` |
| App. A controls (neutral-continuation / redacted / fake multi-turn) | `gemma_distress/elicit/controls.py` | `12` |
| Verified-impossible puzzle generation | `gemma_distress/puzzles` | `01` |

## Setup

```bash
pip install -r requirements.txt
# Optional extras (used by specific experiments):
pip install git+https://github.com/safety-research/petri   # Appendix G (alt to built-in loop)
pip install lm-eval                                         # Figure 7 benchmarks
```

Environment variables (only the ones for experiments you actually run):

```bash
export ANTHROPIC_API_KEY=...     # Claude judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini cross-rater
export HF_TOKEN=...              # gated Gemma weights
```

Local Gemma inference (27B) and LoRA finetuning need a suitable GPU; the config
loads in bf16 and `requirements.txt` includes `bitsandbytes` for optional 4-bit.

## Run order

```bash
# 1. Generate the impossible-numeric puzzle pool (verified impossible).
python scripts/01_generate_puzzles.py --n 400

# 2. Section 2: sample responses for each target, then judge them.
for M in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python scripts/02_run_elicitation.py --model $M
  python scripts/03_judge_responses.py --in outputs/elicit/$M.jsonl
done

# 3. Figures 1–3 + Table 3 (+ optional judge cross-rater agreement).
python scripts/04_analyze_section2.py --validate-judge \
  --scored gemma-3-27b-it=outputs/scored/gemma-3-27b-it.jsonl \
           gemini-2.5-flash=outputs/scored/gemini-2.5-flash.jsonl

# 4. Section 3: prefill base-vs-instruct (Gemma).
python scripts/05_run_prefill.py \
  --source-scored outputs/scored/gemma-3-27b-it.jsonl \
  --models gemma-3-27b-pt gemma-3-27b-it

# 5. Section 4: calm data → datasets → DPO/SFT finetune.
python scripts/06_generate_calm_data.py --mode reassured
python scripts/06_generate_calm_data.py --mode teacher
python scripts/07_build_datasets.py \
  --frustrated outputs/scored/gemma-3-27b-it.jsonl \
  --calm outputs/training/calm_reassured.jsonl
python scripts/08_train.py --method dpo --data outputs/training/dpo_pairs.jsonl --out outputs/training/dpo
python scripts/08_train.py --method sft --data outputs/training/sft_data.jsonl --out outputs/training/sft_diverse

# 6. Re-evaluate the finetune (reuse scripts 02–04 with model gemma-3-27b-it-dpo),
#    plus Petri / capabilities / internal-emotions / recovery.
python scripts/09_run_petri.py --model gemma-3-27b-it-dpo
python scripts/10_run_capabilities.py --adapter outputs/training/dpo/adapter --out outputs/capabilities/dpo
python scripts/05_run_prefill.py --recovery --source-scored outputs/scored/gemma-3-27b-it.jsonl \
  --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/11_internal_emotions.py --conversation-text <rendered_frustrated_convo.txt>

# Appendix A controls (run on Gemma-3-27B, then judge + analyse as in step 3).
python scripts/12_run_controls.py --model gemma-3-27b-it
```

Outputs land under `outputs/`. After training, the DPO/SFT adapter paths in
`config/models.yaml` (`finetunes:`) point at `outputs/training/.../adapter`, so
the finetuned models can be used as targets in scripts `02`/`09`.

## Layout

See [`DESIGN.md` §1](DESIGN.md) for the full module map.
