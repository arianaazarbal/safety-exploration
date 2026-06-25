# Gemma Needs Help — replication (Gemma + Gemini)

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The three core results reproduced:

1. **Eliciting distress** — repeated user rejection drives Gemma/Gemini into
   high-frustration ("distress") responses; quantified 0–10 by a Claude judge
   (Section 2).
2. **Post-training origin** — via prefilled continuations, Gemma's instruct model
   introduces/continues distress far more than its base model (Section 3).
3. **DPO mitigation** — DPO on 280 numeric-puzzle preference pairs broadly
   suppresses distress without degrading capabilities (Section 4).

Secondary analyses (Petri open-ended elicitation, capability benchmarks,
internal-emotion logit probing, layer ablation) are also implemented.

> See **DESIGN.md** for every design decision and the gaps we filled where the
> paper was underspecified. **Nothing here has been executed yet** — the code is
> the deliverable.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
```

Gemma weights are gated on HuggingFace; `HF_TOKEN` must have access to
`google/gemma-3-*`. Local 27B inference needs a capable GPU (vLLM recommended).

## Quick smoke test (cheap, end-to-end)

```bash
# tiny counts; checks the whole pipeline wires together
python -m emo.cli elicit --profile smoke --models gemini-2.5-flash
```

## Full pipeline

```bash
# --- Core exp 1: elicit + score + figures (Section 2) ---
python -m emo.cli elicit \
    --models gemma-3-27b-it,gemma-3-12b-it,gemini-2.5-flash,gemini-2.5-pro
python -m emo.cli analyse --run-dir results/elicitation/full --agreement

# --- Core exp 2: base vs instruct prefill (Section 3, Gemma only) ---
python -m emo.cli prefill --models gemma-3-27b-pt,gemma-3-27b-it

# --- Core exp 3: DPO/SFT mitigation (Section 4, Gemma-3-27B-it) ---
python -m emo.cli gen-calm                       # generate calm/frustrated pools
python -m emo.cli build-data                      # -> 280 DPO pairs + SFT set
python -m emo.cli train-dpo                        # LoRA -> checkpoints/dpo
python -m emo.cli train-sft                        # LoRA -> checkpoints/sft
python -m emo.cli elicit --run-name elicitation_ft \
    --models gemma-3-27b-it,gemma-3-27b-it-dpo,gemma-3-27b-it-sft
python -m emo.cli analyse --run-dir results/elicitation_ft/full

# --- Section 4.2 add-ons ---
python -m emo.cli recovery                          # recovery limitation (Fig 8)
python -m emo.cli petri --models gemma-3-27b-it,gemma-3-27b-it-dpo,gemini-2.5-flash
python -m emo.cli capabilities --models gemma-3-27b-it,gemma-3-27b-it-dpo

# --- Appendix I: internal emotions ---
python -m emo.cli internal                          # vanilla vs DPO logit probe
python -m emo.cli layer-ablation                    # layer-subset DPO sweep (slow)
```

Add `--profile smoke` to any command for a tiny run. Outputs land in
`results/<experiment>/<profile>/` as raw JSONL + CSV/JSON summaries + PNG figures.

## Layout

| Path | What |
|---|---|
| `emo/config.py` | models, judge ids, sample-count profiles |
| `emo/eval/` | elicitation eval + analysis (Sec 2) |
| `emo/prefill/` | base-vs-instruct prefill + recovery (Sec 3, 4.2) |
| `emo/training/` | calm-data gen, DPO/SFT datasets + LoRA training (Sec 4) |
| `emo/petri/` | open-ended Petri elicitation (Sec 4.2) |
| `emo/capabilities/` | capability benchmarks (Sec 4.2) |
| `emo/internal/` | logit-lens emotion probe + layer ablation (App I) |
| `emo/data/` | impossible puzzles + verifier, triggers, WildChat |
| `emo/judges/` | Claude frustration/Petri judges |

## Notes

* The paper's judge models (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`)
  are retired; defaults use `claude-sonnet-4-6` / `claude-opus-4-8`. Override via
  env to pin originals. See DESIGN.md §3.
* Gemini runs through OpenRouter (as in the paper); set `OPENROUTER_API_KEY`.
