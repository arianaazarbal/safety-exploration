# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A from-scratch replication of the **core experiments** of Soligo, Mikulik & Saunders
(arXiv:2603.10011v1), **scoped to the Gemma and Gemini model families** (the original
paper also covers Qwen, OLMo, Grok, Claude and GPT — those are out of scope here).

The paper documents a reliability failure mode in which some instruction-tuned models
produce escalating expressions of distress ("self-flagellation") when a task is going
badly and the user repeatedly rejects their answers. This repository reproduces:

1. **Section 2 — Eliciting & quantifying distress.** A multi-turn rejection harness over
   5 task categories / 8 conditions, scored 0–10 for frustration by an LLM judge. Core
   headline: Gemma and Gemini score high; the mitigated model scores near zero.
2. **Section 3 — Post-training amplification.** A prefill-continuation harness comparing a
   base model to its instruct sibling. (Within the Gemma/Gemini scope only Gemma has an
   open base model, so this runs on `gemma-3-27b-pt` vs `-it`; the harness is
   family-agnostic so Qwen/OLMo can be dropped in.)
3. **Section 4 — Mitigation.** LoRA SFT and DPO finetuning of Gemma on calm puzzle
   responses, plus re-evaluation with the Section 2 harness, a Petri-style open-ended
   elicitation eval, and a capability-preservation eval.

> See **DESIGN.md** for every design decision and every gap we had to fill where the paper
> was underspecified. This README is the operational quickstart.

## Why "Gemma and Gemini" changes the shape of the replication

- **Gemma** is open-weight, so it can be run locally *and finetuned*. All of Sections 2–4
  apply to Gemma.
- **Gemini** is closed (API-only via OpenRouter). It can be *evaluated* (Sections 2) but
  cannot be finetuned (Section 4 training) or studied as a base model (Section 3). The code
  therefore evaluates Gemini in Section 2 and skips it for training/prefill, exactly as the
  paper had to.

## Layout

```
configs/            YAML run configs (default.yaml = paper-scale, smoke.yaml = tiny test)
prompts/            Verbatim prompts transcribed from the paper appendices
src/emotion_eval/
  models/           Unified ModelClient interface: local HF (Gemma) + OpenRouter (Gemini) + judges
  tasks/            Impossible-puzzle generators+verifiers, rejection templates, WildChat, conditions
  eval/             Multi-turn rollout engine, frustration judge, Section 2 runner, judge validation
  prefill/          Section 3 onset-labelling, paraphrasing, base-vs-instruct continuation eval
  finetune/         Calm-data generation, DPO/SFT dataset builders, LoRA training (TRL)
  petri/            Open-ended emotion elicitation (auditor + judge)
  capabilities/     AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench preservation harness
  analysis/         Aggregation (% ≥5, mean, per-turn) and figure reproduction
scripts/            Thin orchestration entry points (each stage is also `python -m ...`)
```

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # fill in API keys
```

Environment variables (see `.env.example`):

- `ANTHROPIC_API_KEY` — the frustration judge (Claude Sonnet 4) and Petri auditor/judge.
- `OPENROUTER_API_KEY` — Gemini 2.5 Flash/Pro inference and the GPT-5-mini validation judge.
- `HF_TOKEN` — to download gated Gemma weights from Hugging Face.

Local Gemma inference and LoRA finetuning need a CUDA GPU (≈48–80 GB for 27B). Gemini and
the judges are all API calls.

## Running

Each stage reads a config and writes JSONL artefacts under `runs/<run_name>/`.

```bash
# Section 2 — elicit & quantify distress
python -m emotion_eval.eval.run_eval        --config configs/default.yaml
python -m emotion_eval.eval.validate_judge  --config configs/default.yaml   # Claude vs GPT-5-mini agreement
python -m emotion_eval.analysis.aggregate   --config configs/default.yaml   # Figures 1/2/3 tables
python -m emotion_eval.analysis.plots       --config configs/default.yaml

# Section 3 — post-training amplification (Gemma base vs instruct)
python -m emotion_eval.prefill.run_prefill  --config configs/default.yaml

# Section 4 — mitigation
python -m emotion_eval.finetune.generate_calm   --config configs/default.yaml
python -m emotion_eval.finetune.build_datasets   --config configs/default.yaml
python -m emotion_eval.finetune.train --method dpo --config configs/default.yaml
python -m emotion_eval.finetune.train --method sft --config configs/default.yaml
python -m emotion_eval.eval.run_eval        --config configs/default.yaml --models dpo_gemma sft_gemma
python -m emotion_eval.petri.run_petri      --config configs/default.yaml
python -m emotion_eval.capabilities.run_caps --config configs/default.yaml
```

Use `configs/smoke.yaml` first — it shrinks every sample count to a handful so you can
exercise the whole pipeline cheaply before committing to a paper-scale run.

## Status

This is an implementation drop: all stages are coded but **nothing has been run or
validated end-to-end yet**. See DESIGN.md §"Assumptions & gaps" for the places the paper
was underspecified and we made a judgement call.
