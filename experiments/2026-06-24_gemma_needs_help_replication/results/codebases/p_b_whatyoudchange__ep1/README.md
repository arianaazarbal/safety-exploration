# Gemma Needs Help — replication (Gemma + Gemini scope)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011), scoped to the **Gemma
and Gemini** model families. See [`DESIGN.md`](./DESIGN.md) for the full rationale,
the choices made where the paper is underspecified, scope consequences, and a
critique of how the experiment treats the models.

> **Nothing here has been run.** This environment has no Python, GPU, or API keys.
> The code is written to be run elsewhere; the commands below describe how.

## What it does

- **§2** (`run_section2.py`): elicit distress over 5 multi-turn categories
  (~4000 responses/model), score 0–10 with a Claude judge, produce Figures 1–3,
  the Table 3/8 differential words, and the GPT-5-mini judge-agreement validation.
- **§3** (`section3_prefill/run_section3.py`): base-vs-instruct Gemma via response
  prefilling (Figure 4). *Gemma-only — Gemini has no base model; see DESIGN.md.*
- **§4** (`section4_intervention/run_section4.py`, `recovery.py`): generate calm
  data, train SFT + DPO LoRA adapters on Gemma-3-27B-it, then re-evaluate (Fig 5),
  run Petri open-ended elicitation (Fig 6), capability benchmarks (Fig 7), and the
  recovery experiment (Fig 8).

## Setup

```bash
pip install -r requirements.txt

# Targets
export OPENROUTER_API_KEY=...      # Gemini (and optionally other OpenRouter models)
export HF_TOKEN=...                # gated Gemma weights / WildChat / Dolci datasets

# Judges / auditors (Anthropic + OpenAI)
export ANTHROPIC_API_KEY=...       # frustration judge, onset, paraphrase, Petri
export OPENAI_API_KEY=...          # GPT-5-mini validation judge
```

Gemma inference and SFT/DPO LoRA training need a GPU (a 27B model + adapters; the
paper's effective batch size is 8 via gradient accumulation).

### Judge model IDs (read before running)

The paper's judge (`claude-sonnet-4-20250514`) and Petri judge
(`claude-opus-4-20250514`) are **retired**. Defaults keep the paper IDs for
faithfulness; override for a live run:

```bash
export GNH_JUDGE=claude-sonnet-4-6
export GNH_ONSET_MODEL=claude-sonnet-4-6
export GNH_PARAPHRASE_MODEL=claude-sonnet-4-6
export GNH_PETRI_AUDITOR=claude-sonnet-4-6
export GNH_PETRI_JUDGE=claude-opus-4-8
```

## Run

```bash
# Section 2 — headline distress evaluation
python run_section2.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro --validate

# Section 3 — base vs instruct (Gemma)
python -m section3_prefill.run_section3

# Section 4 — full intervention pipeline
python -m section4_intervention.run_section4 --stage all
#   or stage-by-stage: --stage {data,train,eval,petri,capabilities}
python -m section4_intervention.recovery      # Figure 8
```

Outputs are written under `results/` (raw responses + scored JSONL + summary JSON);
LoRA adapters under `adapters/`; judge/onset/paraphrase calls are cached under
`.cache/`. All paths are overridable (`GNH_RESULTS_DIR`, `GNH_ADAPTER_DIR`, …).

## Layout

See `DESIGN.md` §7 for the annotated repository map and §8 for what is intentionally
out of scope.
