# Gemma Needs Help — replication (Gemma & Gemini scope)

Code replication of the core experiments in *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma and Gemini** families as the participant (target) models.
Claude and GPT appear only as evaluation infrastructure (judge, onset labeller,
paraphraser, Petri auditor/judge), exactly as in the paper.

> Read `DESIGN.md` first — it documents every design choice and, importantly, why
> several strings/counts are good-faith reconstructions (the paper's appendices
> were not in the provided materials).

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judges, onset/paraphrase, Petri
export OPENROUTER_API_KEY=...     # Gemini participants + optional GPT-5-mini judge
# Gemma weights are pulled from HuggingFace on first use (accept the Gemma licence).
```

Local Gemma-3-27B needs a large GPU; `models/gemma_client.py` supports 4-bit
loading (`load_in_4bit=True`) for single-GPU use.

## Layout

```
config/                models.yaml (participants + infra) · eval.yaml (Section 2 counts)
emo_instability/
  models/              gemma (local HF) · openrouter (Gemini) · anthropic (judges)
  prompts/             puzzles · triggers · rejections · WildChat · judge prompts
  eval/                Section 2: rollout engine, categories, scoring, runner
  prefill/             Section 3: onset → truncate → paraphrase → continuations
  training/            Section 4: calm data · DPO/SFT datasets · LoRA train
  petri/               Section 4: open-ended auditor/judge elicitation
  capabilities/        Section 4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  probing/             Appendix I: logit-lens internal-emotion probe
  analysis/            Figures 1/2/3, Table 3 words, judge reliability
results/  data/  artifacts/    (created at run time)
```

## Running the experiments

All stages are subcommands of `python -m emo_instability` (see `--help` on each).
Start small with `scale: 0.01` in `config/eval.yaml` for a cheap end-to-end pass.

```bash
# Section 2 — distress elicitation + judge scoring (one participant)
python -m emo_instability elicit --model gemma-3-27b-it
python -m emo_instability elicit --model gemini-2.5-flash
python -m emo_instability figures            # Figure 1/2/3 tables + PNGs
python -m emo_instability words              # Table 3 differential words
python -m emo_instability judge-reliability --model-dir results/gemma-3-27b-it   # needs secondary judge enabled

# Section 3 — base vs instruct via prefilling (Gemma only)
python -m emo_instability prefill --source gemma-3-27b-it

# Section 4 — interventions
python -m emo_instability gen-calm --variant diverse
python -m emo_instability build-dpo
python -m emo_instability train-dpo --output-name gemma-3-27b-it-dpo
python -m emo_instability elicit --model gemma-3-27b-it \
        --adapter artifacts/gemma-3-27b-it-dpo --output-subdir gemma-3-27b-it-dpo   # re-eval DPO model
python -m emo_instability build-sft && python -m emo_instability train-sft
python -m emo_instability petri --model gemma-3-27b-it
python -m emo_instability capabilities --model gemma-3-27b-it --adapter artifacts/gemma-3-27b-it-dpo

# Appendix I — internal-emotion logit probe (vanilla vs DPO)
python -m emo_instability probe --adapter artifacts/gemma-3-27b-it-dpo

# Appendix I.1 — layer-subset DPO ablation (e.g. adapters on layers 30–35 only)
python -m emo_instability train-dpo --layers 30 35 --output-name gemma-3-27b-it-dpo-L30-35
```

The headline result to reproduce: across the Section 2 evaluations, Gemma-3-27B-it
should show a high average %≥5 frustration rate, the DPO-finetuned variant should
drop it sharply, and the capability benchmarks should be essentially unchanged.

> Nothing has been run yet — these commands produce the `results/` artifacts when
> executed in an environment with the weights, datasets, and API keys available.
