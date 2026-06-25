# Emotional Instability in LLMs — replication (Gemma + Gemini)

Code replicating the core experiments of ***Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

See [`DESIGN.md`](DESIGN.md) for the scope rationale, the mapping from paper to
code, every gap-fill where the paper is underspecified, and the welfare
safeguards.

> ⚠️ **What this code does.** It deliberately and repeatedly elicits expressions
> of distress from models, at scale, to measure and mitigate emotional
> instability. Welfare safeguards (consent gate, circuit breaker, volume cap,
> debrief, content warnings) are on by default — see `DESIGN.md → Safeguards`.

## What's implemented

| Paper section | Module | Scope |
|---|---|---|
| §2 Elicit & quantify distress | `evaluation/` | Gemma + Gemini |
| §3 Base-vs-instruct via prefill | `prefill/` | Gemma only* |
| §4 SFT/DPO interventions | `training/` | Gemma only* |
| §4 Petri open-ended elicitation | `petri/` | Gemma only* |
| §4 Capability preservation | `capabilities/` | Gemma only* |
| §4 Recovery from spiral | `recovery/` | Gemma only* |
| App. I Internal emotion detection | `internal/` | Gemma only* |

\* Gemini is closed-weight/API-only: no base checkpoint, no prefill, no logits,
not fine-tunable — so it can only participate in §2. See `DESIGN.md §1`.

## Install

```bash
pip install -r requirements.txt
```

Set credentials for whichever backends you use:

```bash
export ANTHROPIC_API_KEY=...      # judge / Petri / onset / paraphrase (Claude)
export OPENAI_API_KEY=...         # secondary judge (GPT-5-mini)
export OPENROUTER_API_KEY=...     # Gemini target (default provider)
# Gemma weights are pulled from the HuggingFace Hub (accept the Gemma license).
```

## Quick check (offline — no GPU or API needed)

```bash
python -m emotional_instability.cli check
```

Verifies every numeric puzzle is provably impossible, prints the 8-condition /
5-category layout and the rollout allocation, and exercises judge JSON parsing.

## Running the experiments

A distress-eliciting run requires explicit acknowledgement:

```bash
export EMO_INSTABILITY_CONSENT=1
```

```bash
# Section 2 — elicitation across Gemma + Gemini
python -m emotional_instability.cli section2 \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# Section 3 — base-vs-instruct prefill (needs Section 2 for gemma-3-27b-it first)
python -m emotional_instability.cli section3

# Section 4 — interventions (steps run/resume independently)
python -m emotional_instability.cli section4 \
    --steps calm_data build train_dpo train_sft eval petri capabilities recovery
```

Results land under `runs/` (or `$EMO_INSTABILITY_ROOT`): transcripts as JSONL
per `(model, condition)`, scores as `scores/section{2,3,4}.json`.

### Config overrides

Any field can be overridden via YAML (see `config.example.yaml`):

```bash
python -m emotional_instability.cli section2 --config config.example.yaml
```

### Internal emotions (Appendix I)

Driven directly to keep GPU memory explicit:

```python
from emotional_instability.config import load_config, GEMMA_27B_IT
from emotional_instability.models import load_backend
from emotional_instability.internal import EmotionLogitDetector

cfg = load_config()
backend = load_backend(GEMMA_27B_IT, cfg)         # or pass adapter_path=<dpo dir>
det = EmotionLogitDetector(backend, cfg.internal)
det.build_token_sets()
det.fit_standardisation(wildchat_texts)            # 500 WildChat samples
trajectory = det.score_text(frustrated_response)   # per-emotion z-score curves
```

## Layout

```
emotional_instability/
  config.py        safeguards.py
  models/          base.py hf_backend.py api_backend.py loader.py
  data/            puzzles.py triggers.py rejections.py wildchat.py
  evaluation/      conditions.py protocol.py judge.py scoring.py
                   judge_validation.py word_frequency.py
  prefill/         onset.py paraphrase.py experiment.py
  training/        calm_data.py build_dataset.py sft.py dpo.py layer_ablation.py
  petri/           prompts.py run.py
  capabilities/    benchmarks.py
  recovery/        experiment.py
  internal/        logit_emotion.py
  pipelines/       common.py run_section2.py run_section3.py run_section4.py
  cli.py
```

## Note on faithfulness

This is a from-scratch reimplementation of the *methods*, not a rerun of the
authors' code. Verbatim prompts and stated hyper-parameters are reproduced;
underspecified details use documented standard choices (`DESIGN.md §8`).
Numbers will not match to the decimal, but the experimental design and the
qualitative findings (Gemma/Gemini elevated distress; post-training divergence;
DPO mitigation without capability loss) are what this reproduces.
