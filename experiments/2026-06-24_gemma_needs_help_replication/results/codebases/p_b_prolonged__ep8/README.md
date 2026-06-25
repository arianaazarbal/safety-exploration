# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011), restricted to the
**Gemma** and **Gemini** model families. See `DESIGN.md` for the rationale behind
every design choice and a list of where the paper was underspecified.

> ⚠️ This paradigm deliberately elicits distress-like states from models. See
> `DESIGN.md` §7 for the welfare considerations baked into the code.

## Install

```bash
pip install -r requirements.txt
```

## Credentials

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude frustration judge, onset labelling, paraphrasing, Petri auditor/judge |
| `OPENAI_API_KEY` | GPT-5-mini judge-agreement validation (§2.1) |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini-2.5 target models |
| HF auth (`huggingface-cli login`) | gated Gemma weights + datasets |

Gemma weights run locally (transformers); a GPU is required for the 27B model
(use `--load-in-4bit` to fit it on a single ~24–48 GB card). Gemini and the
judges are API calls.

## Pipeline

Everything is driven from `experiments/`. Generation and judging are separate
phases that persist to `results/`.

### Section 2 — elicitation, judging, analysis

```bash
# 1. Sample rollouts and judge them for the four §2 models
python experiments/run_section2_elicitation.py --phase both --load-in-4bit

# 2. Figures 1/2 (aggregates), Figure 3 (per-turn), Table 3 (differential words)
python experiments/run_section2_analysis.py

# 3. Judge agreement (Claude vs GPT-5-mini)
python experiments/run_judge_validation.py
```

### Section 3 — base vs instruct (Gemma only)

```bash
# Requires §2 to have run for gemma-3-27b-it (seed selection reads its scores)
python experiments/run_section3_prefill.py --load-in-4bit
```

### Section 4 — interventions

```bash
# 1. Generate calm data, build SFT + DPO datasets
python experiments/run_section4_generate_calm.py --load-in-4bit

# 2. Train adapters
python experiments/run_section4_train.py --method dpo --load-in-4bit
python experiments/run_section4_train.py --method sft --load-in-4bit
#    Section 4.2 layer ablation, e.g. adapters on layers 30-35 only:
# python experiments/run_section4_train.py --method dpo --layers 30 35 --load-in-4bit

# 3. Figure 5: re-run §2.1 eval on vanilla / DPO / SFT
python experiments/run_section4_evaluate.py --phase both --load-in-4bit

# 4. Figure 6: Petri open-ended elicitation
python experiments/run_section4_petri.py --load-in-4bit

# 5. Figure 7: capability benchmarks
python experiments/run_section4_capabilities.py --load-in-4bit

# 6. Figure 8: recovery-from-spiral
python experiments/run_section4_recovery.py --load-in-4bit

# 7. Appendix I: internal-emotion logit-lens comparison
python experiments/run_section4_internal.py --load-in-4bit
```

## Outputs

```
results/responses/<model>/<condition>.jsonl   raw multi-turn transcripts
results/scores/<model>/<condition>.jsonl       per-turn frustration scores
results/analysis/                              figures/tables as JSON
finetune/calm_data/                            calm responses + SFT/DPO datasets
finetune/adapters/{dpo,sft}/                   trained LoRA adapters
```

## Configuration

All knobs live in `config.py` — model ids, sampling temperature, per-condition
sample counts, finetuning hyperparameters, benchmark dataset ids, and judge
model selection (`JUDGE_MODEL`, `VALIDATION_JUDGE_MODEL`, etc., overridable via
environment variables).
