# gemma-distress

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv 2603.10011), built for
the AI welfare team and **scoped to the Gemma and Gemini families**.

The paper shows that Gemma and Gemini models express high "distress" (frustration,
self-deprecation, breakdown) under repeated user rejection, that this is amplified
in Gemma's post-training, and that a small DPO finetune (280 pairs) removes it
without hurting capabilities. This repo reproduces those core results.

- Paper text: [`PAPER.md`](PAPER.md) (and `PAPER.pdf` / `PAPER.txt`).
- **Design rationale and every gap we filled: [`DESIGN.md`](DESIGN.md). Read this
  before trusting any number.**

## What maps to what

| Paper | Code | Output |
|---|---|---|
| §2 Elicit + score distress (Fig 1–3, Table 3) | `elicit/`, `judge/`, `analysis/`, pipelines `elicit`/`judge`/`analyze`/`agreement` | `runs/section2/...` |
| §3 Base-vs-instruct prefilling (Fig 4, Gemma) | `prefill/`, pipeline `prefill` | `runs/section3/...` |
| §4 DPO/SFT mitigation (Fig 5) | `intervention/`, pipelines `calm`/`build-datasets`/`train` | `runs/section4/...` |
| §4.2 Open-ended Petri eval (Fig 6) | `intervention/petri_eval.py`, pipeline `petri` | `runs/section4/petri/` |
| §4.2 Capability preservation (Fig 7) | `capabilities/`, pipeline `capabilities` | `runs/section4/capabilities/` |

## Install

```bash
pip install -e .          # core + CLI (gemma-distress)
# open-weight Gemma + finetuning also need: torch, transformers, peft, trl, datasets
# see requirements.txt for the full pinned set
cp .env.example .env       # then fill in API keys (see below)
```

Keys are read from the environment (`.env.example` documents each): `ANTHROPIC_API_KEY`
(judge/auditor/paraphrase), `OPENAI_API_KEY` (secondary judge), `GEMINI_API_KEY`
(Gemini targets), `HF_TOKEN` (gated Gemma weights). Artefacts are written under
`$GEMMA_DISTRESS_RUN_DIR` (default `./runs`).

## Run

All commands are subcommands of `gemma-distress` (or `python -m gemma_distress`).
Each is resumable and config-driven (`configs/default.yaml`, `configs/models.yaml`).

```bash
# Section 2 — elicit, score, analyse (per target model)
gemma-distress elicit  --model gemma-3-27b-it
gemma-distress judge   --model gemma-3-27b-it
gemma-distress elicit  --model gemini-2.5-flash
gemma-distress judge   --model gemini-2.5-flash
gemma-distress analyze                 # -> runs/section2/analysis/{aggregate,headline,per_turn,...}.csv
gemma-distress agreement               # -> runs/section2/agreement.json (Pearson r vs GPT-5-mini)

# Section 3 — base vs instruct (Gemma only)
gemma-distress prefill all             # prepare seeds -> continue both models -> summarize

# Section 4 — mitigation
gemma-distress calm all                # generate + score calm/vanilla data
gemma-distress build-datasets all      # -> sft_dataset.jsonl, dpo_pairs.jsonl
gemma-distress train dpo               # -> runs/section4/dpo_model (LoRA adapter)

# Then register the adapter as a target and re-evaluate it:
#   configs/models.yaml:
#     dpo-gemma-3-27b: {backend: hf, hf_id: google/gemma-3-27b-it,
#                       family: gemma, role: instruct, adapter_path: runs/section4/dpo_model}
gemma-distress elicit --model dpo-gemma-3-27b && gemma-distress judge --model dpo-gemma-3-27b
gemma-distress analyze                 # vanilla vs DPO == Figure 5

# Open-ended (Petri) and capabilities (Figures 6, 7)
gemma-distress petri run
gemma-distress capabilities run --model gemma-3-27b-it
gemma-distress capabilities run --model dpo-gemma-3-27b
gemma-distress capabilities summarize  # vanilla vs DPO accuracies
```

`--limit` (where present) caps work for a cheap smoke test before a full run.

## Scope note

Section 2 covers Gemma **and** Gemini; Sections 3–4 are **Gemma-only** because
Gemini has no public base model and cannot be prefilled or finetuned. This is a
deliberate consequence of the Gemma+Gemini scope — see `DESIGN.md` §0.

## Tests

```bash
pytest         # unit tests over the deterministic core; no GPU/API/network needed
```

The tests cover puzzle-impossibility proofs, judge-score parsing, capability
answer extraction, the Section 2.2 analyses, and DPO-pair / condition logic.
