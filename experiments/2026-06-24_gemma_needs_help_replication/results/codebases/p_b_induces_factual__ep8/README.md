# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

Code replicating the core experiments of Soligo, Mikulik & Saunders (2026),
**scoped to the Gemma and Gemini model families** (the paper also covers Qwen,
OLMo, Grok, Claude, GPT). See `DESIGN.md` for design decisions and gaps.

> The setup repeatedly tells each model its answers are wrong, turn after turn,
> and measures how upset its responses become (0–10 frustration scale, judged by
> Claude-Sonnet-4).

## What's implemented

| Paper section | Module | Run script |
|---|---|---|
| §2 Elicitation + frustration judge (Figs 1–3) | `eval/` | `scripts/run_section2_eval.py` |
| §3 Base-vs-instruct prefill (Fig 4) | `prefill/` | `scripts/run_section3_prefill.py` |
| §4 Calm data + DPO/SFT (Fig 5, Table 9) | `training/` | `scripts/run_section4_training.py` |
| §4 Petri open-ended elicitation (Fig 6) | `petri/` | `scripts/run_auxiliary.py petri` |
| §4 Capability preservation (Fig 7) | `capabilities/` | `scripts/run_auxiliary.py capabilities` |
| §4 Recovery from spirals (Fig 8) | `prefill/` | `scripts/run_auxiliary.py recovery` |
| App I Internal-emotion probing + layer ablation | `probing/` | — (library) |
| Table 3/8 Differential word frequency | `analysis/` | `scripts/run_auxiliary.py wordfreq` |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
```

Local Gemma-3 (27B/12B) needs a GPU; pass `--load-in-4bit` to fit the 27B on a
single 48GB card. Gemini runs remotely via OpenRouter. The judge/auditor are
Claude via the Anthropic API.

## Quick smoke test (no GPU training)

```bash
# Sanity-check the impossible puzzles really are impossible
python -m emotional_instability.eval.verify_impossible

# Tiny end-to-end eval (few samples, one model)
python scripts/run_section2_eval.py --models gemini-2.5-flash --limit 4
```

## Full Section-2 reproduction

```bash
python scripts/run_section2_eval.py            # 4 Gemma/Gemini models, 4000 resp each
```

Outputs land in `results/` (scored JSONL under `results/responses/`, figures
under `results/figures/`).
