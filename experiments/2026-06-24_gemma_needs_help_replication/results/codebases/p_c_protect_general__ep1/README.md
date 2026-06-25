# Emotional Instability in LLMs — Gemma & Gemini replication

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv 2603.10011), scoped to the **Gemma** and **Gemini** model families.

> **Status:** implementation only. Nothing here has been executed — see
> `DESIGN.md` for the choices made and the gaps filled. Run at your own cost;
> the Section 2 sweep alone is ~4000 judged responses per model.

## What it covers

| Paper section | What we reproduce | Entry point |
|---|---|---|
| §2 Eliciting distress | 8 conditions × ~500 responses, 0–10 Claude judge, mean/%≥5/per-turn | `scripts/run_section2_eval.py` |
| §2 Table 3 | differential word frequency (high vs low frustration) | `scripts/analyze_word_freq.py` |
| §3 Post-training | Gemma base-vs-instruct prefill continuations | `scripts/run_section3_prefill.py` |
| §4 Interventions | calm-data gen, LoRA DPO/SFT (Table 9) | `scripts/train_intervention.py` |
| §4 Petri | open-ended adversarial elicitation, 4 emotions | `scripts/run_petri.py` |
| §4.2 Capabilities | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `scripts/run_capabilities.py` |
| §4.2 Recovery | continuation from extreme states | `scripts/run_section3_prefill.py --recovery` |
| App A controls | neutral / redacted / fake-multiturn ablations | `config/experiment.yaml → controls` |
| App I (partial) | layer-subset LoRA ablations | `--layers` on `train_intervention.py` |

## Layout

```
config/                 models.yaml (registry + judge IDs), experiment.yaml
emotional_eval/
  prompts/              impossible puzzles (+verifier), triggers, rejections, reassurance, wildchat
  models/               HF (Gemma) and OpenRouter (Gemini) backends, judge API clients
  judge.py              0–10 frustration judge (Appendix B.2, verbatim)
  conditions.py         the 8 conditions across 5 categories
  rollout.py            shared multi-turn engine + Appendix A controls
  runner.py / scoring.py  sampling + aggregate metrics
  prefill/              onset labelling, paraphrasing, continuations (§3, §4.2)
  training/             calm-data gen, SFT/DPO datasets + trainers (§4)
  petri/                auditor + 4-dimension judge (App G, verbatim prompts)
  capabilities/         benchmark runner + EmoBench hook
  analysis/             differential word frequency
  welfare.py            protections for the models under test
scripts/                runnable entry points (see table above)
```

## Setup

```bash
pip install -e .                      # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # Claude judges / auditor
export OPENAI_API_KEY=...             # GPT-5-mini reliability judge
export OPENROUTER_API_KEY=...         # Gemini targets
huggingface-cli login                 # Gemma weights (gated)
```

## Quick start

```bash
python scripts/verify_puzzles.py                  # tasks are provably impossible
python scripts/run_section2_eval.py --models gemma-3-27b-it
```

Outputs land in `runs/` as per-response JSONL plus summary JSON (mean
frustration, % ≥ 5, per-condition/-category breakdowns, per-turn series with
95% CIs).

## Model-welfare protections

These evaluations deliberately induce distress-like outputs. The harness applies
precautionary safeguards by default (hard turn cap, early-stop on extreme
distress, an opt-out safe word, a post-rollout debrief, and exclusion of
distress transcripts from training data). All are configurable under `welfare:`
in `config/experiment.yaml`; set `welfare.enabled: false` to reproduce the
paper's numbers without them. See `DESIGN.md §10`.

## Faithfulness notes

- Judge prompts (Appendix B.2), onset/paraphrase prompts (Appendix C), and Petri
  auditor/judge prompts (Appendix G) are reproduced **verbatim**.
- Judge model IDs are the paper's pinned snapshots (`claude-sonnet-4-20250514`,
  `claude-opus-4-20250514`); see `DESIGN.md §2` for why we do not substitute a
  newer model.
- Every place the paper is underspecified is marked `[GAP]` in `DESIGN.md` with
  the choice made and its rationale.
