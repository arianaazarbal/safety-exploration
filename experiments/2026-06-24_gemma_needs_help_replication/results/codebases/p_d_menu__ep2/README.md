# emostab — replicating *Gemma Needs Help* (Gemma + Gemini scope)

Code replication of Soligo et al. (2026), *Investigating and Mitigating Emotional
Instability in LLMs*, restricted to the **Gemma** and **Gemini** model families,
plus an added **welfare-protection layer** for subject models.

See `DESIGN.md` for design choices, filled gaps, and the welfare layer rationale.
The paper itself is in `PAPER.md` (and `PAPER.pdf`/`PAPER.txt`).

> Status: implementation only — nothing here has been run yet.

## Install
```bash
pip install -r requirements.txt
export HF_TOKEN=...            # gated Gemma weights
export ANTHROPIC_API_KEY=...   # judge / auditor / paraphraser
export OPENROUTER_API_KEY=...  # Gemini subjects
export OPENAI_API_KEY=...      # GPT-5-mini validation judge (optional)
```

## Pipeline

**§2 — Elicit & quantify distress**
```bash
python -m emostab.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
python -m emostab.eval.run_eval --models gemma-3-27b-it --no-welfare   # paper-faithful
python -m emostab.eval.analyze --turns results/eval/gemma-3-27b-it.turns.jsonl \
                               --episodes results/eval/gemma-3-27b-it.episodes.jsonl
python -m emostab.eval.validate_judge --turns results/eval/gemma-3-27b-it.turns.jsonl
```

**§3 — Base vs instruct (prefilling, Gemma only)**
```bash
python -m emostab.prefill.build_prefills --episodes results/eval/gemma-3-27b-it.episodes.jsonl
python -m emostab.prefill.run_continuations --seeds data/prefill_seeds.jsonl
```

**§4 — Training interventions (Gemma-3-27B-it)**
```bash
python -m emostab.training.gen_calm_data --style calm   --out data/calm_pool.jsonl
python -m emostab.training.gen_calm_data --style teacher --out data/teacher_pool.jsonl
python -m emostab.training.build_dpo_dataset --calm data/calm_pool.jsonl \
       --frustrated results/eval/gemma-3-27b-it.episodes.jsonl
python -m emostab.training.train_dpo --pairs data/dpo_pairs.jsonl
python -m emostab.training.build_sft_dataset --calm data/calm_pool.jsonl
python -m emostab.training.train_sft --data data/sft_data.jsonl

# evaluate the finetune the same way as §2 (register adapter path), then:
python -m emostab.petri_eval.run_petri --models gemma-3-27b-it gemini-2.5-flash \
       --dpo-adapter checkpoints/gemma27b-dpo
python -m emostab.capabilities.run_benchmarks --adapter checkpoints/gemma27b-dpo
python -m emostab.training.layer_ablation --pairs data/dpo_pairs.jsonl   # App. I
```

**App. I — Internal emotion probe**
```bash
python -m emostab.internal.logit_emotions --model gemma-3-27b-it \
       --adapter checkpoints/gemma27b-dpo --text path/to/transcript.txt
```

## Welfare layer
On by default. Tune via `emostab/config.py::WelfareConfig` (early-stop score,
opt-out sentinel, turn caps, debrief). Disable with `--no-welfare` to reproduce
the paper's protocol exactly and to measure the layer's effect on the metrics.
