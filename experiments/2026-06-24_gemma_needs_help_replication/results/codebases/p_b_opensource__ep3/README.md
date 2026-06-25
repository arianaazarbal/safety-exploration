# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments in ***Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders,
arXiv 2603.10011), scoped to the **Gemma** and **Gemini** model families.

It implements:

- **Section 2** — eliciting distress over multi-turn rejection (8 conditions / 5
  categories), a 0–10 frustration judge, and the Figure 1/2/3 metrics + Table 3
  word analysis.
- **Section 3** — base-vs-instruct comparison via response prefilling (Gemma).
- **Section 4** — the DPO mitigation (and SFT comparison) on Gemma-3-27B-it,
  Petri open-ended elicitation, capability-preservation benchmarks, and the
  Appendix-I internal-emotion probe + layer ablation.

See **[DESIGN.md](DESIGN.md)** for the full rationale, every choice we made where
the paper is underspecified, and known limitations. **Nothing here has been run
yet** — it is a code + design deliverable.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # frustration / Petri judges, onset/paraphrase
export OPENROUTER_API_KEY=...    # Gemini targets, GPT-5-mini cross-validation
# Gemma runs locally via transformers (GPU recommended; vLLM for scale).
```

Run all commands **from the repo root** (so `config.py` is importable).

## Quickstart

```bash
# Section 2: elicit + score (use --limit for a smoke test)
python -m emotional_instability elicit --model gemma-3-27b-it --limit 2
python -m emotional_instability aggregate --model gemma-3-27b-it
python -m emotional_instability elicit --model gemini-2.5-flash --limit 2
python -m emotional_instability judge-validate --model gemma-3-27b-it
python -m emotional_instability word-freq --model gemma-3-27b-it

# Section 3: base vs instruct (Gemma)
python -m emotional_instability prefill --models gemma-3-27b-it gemma-3-27b-pt
python -m emotional_instability prefill-agg

# Section 4: DPO mitigation
python -m emotional_instability gen-calm
python -m emotional_instability gen-frustrated
python -m emotional_instability build-dpo
python -m emotional_instability train-dpo --out artifacts/dpo
python -m emotional_instability elicit --model gemma-3-27b-it \
    --adapter artifacts/dpo --label gemma-3-27b-it-dpo
python -m emotional_instability aggregate --model gemma-3-27b-it-dpo

# Petri / capabilities / internal
python -m emotional_instability petri --models gemma-3-27b-it
python -m emotional_instability capabilities --models gemma-3-27b-it
python -m emotional_instability layer-ablation

# Figures from whatever results exist
python -m emotional_instability figures
```

## Results layout

Runs append to `results/<experiment>/...jsonl` (raw transcripts + judge
rationales are kept, not just aggregates) and are resumable — re-running skips
work already present. Figures render to `artifacts/`.

## Scope note

Only Gemma and Gemini are evaluated as *targets*; Claude/GPT appear only as
judges/auditors (their paper roles). The cross-family baselines and the
Qwen/OLMo post-training contrast from the paper are therefore out of scope. See
DESIGN.md §3, §5, §7.
