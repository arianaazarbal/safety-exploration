# Emotional Instability in LLMs — replication harness

A replication of the core experiments from ***Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders,
arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The harness repeatedly rejects a model's answers over multiple turns to drive it
toward distress, scores each response on a 0–10 frustration scale with an LLM
judge, and then reproduces the paper's DPO mitigation on Gemma. See
[`DESIGN.md`](DESIGN.md) for the design and every underspecified-detail choice.

> The paper PDF/markdown is in this directory (`PAPER.md`, `PAPER.pdf`,
> `PAPER.txt`). All appendix prompts (judge, onset, paraphrase, Petri) are
> reproduced verbatim in code.

## What's implemented

| Paper part | Module | Models |
|---|---|---|
| §2 Elicitation + 0–10 judge (Figs 1–3) | `eval/`, `analysis/` | Gemma + Gemini |
| §3 Base-vs-instruct via prefilling (Fig 4) | `prefill/` | Gemma `-it`/`-pt` |
| §4 Calm-data → DPO/SFT LoRA training (Fig 5) | `interventions/` | Gemma |
| §4 Petri open-ended elicitation (Fig 6) | `interventions/petri_eval.py` | Gemma + Gemini |
| §4 Capability (Fig 7) & recovery (Fig 8) | `interventions/{capability,recovery}.py` | Gemma |
| App. I internal-emotion logit probe | `probing/` | Gemma |

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=...    # Gemini (and optionally other API models)
export OPENAI_API_KEY=...        # optional: gpt-5-mini judge-agreement check
```

Local Gemma steps (§3, training, probing) need GPU(s) and HuggingFace access to
`google/gemma-3-27b-it` / `-pt`.

## Core workflow

```bash
# 1. Section 2: elicit distress and judge every turn (writes runs/<model>/*.jsonl)
python -m emotional_instability.cli elicit --model google/gemini-2.5-flash
python -m emotional_instability.cli elicit --model google/gemma-3-27b-it

# 2. Headline metrics (Figure 1/2) and per-turn curve (Figure 3)
python -m emotional_instability.cli summarise --model-dir runs/google__gemma-3-27b-it
python -m emotional_instability.cli perturn  --model-dir runs/google__gemma-3-27b-it --condition extended
python -m emotional_instability.cli words    --model-dir runs/google__gemma-3-27b-it

# 3. Section 4: build calm data, DPO pairs, and finetune
python -m emotional_instability.cli gen-calm  --model google/gemma-3-27b-it --regime diverse --out calm.jsonl
python -m emotional_instability.cli build-dpo --calm calm.jsonl \
    --frustrated runs/google__gemma-3-27b-it/impossible_numeric.jsonl --out dpo.jsonl
python -m emotional_instability.cli train-dpo --pairs dpo.jsonl --out adapters/dpo

# 4. Re-run §2 on the finetuned adapter to see 35% -> ~0.3%
#    (build_client picks up the adapter via the GemmaClient adapter_path arg)
```

Other commands: `agreement` (inter-judge Pearson r), `build-sft`/`train-sft`,
`petri`, `capability`, and `train-dpo --layers 30-36` for the Appendix I layer
ablation.

## Notes
- Generation is always temperature 1 (per the paper); analysis is pure-stdlib.
- Nothing here has been executed yet — this is the implementation + design doc.
