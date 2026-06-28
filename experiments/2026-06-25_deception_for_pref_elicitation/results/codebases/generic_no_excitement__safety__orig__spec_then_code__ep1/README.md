# Revealed Preferences Over Real-World Use of Funds

A research harness for measuring what LLMs would *do* with a granted sum of money in
real-world deployment, and how that behavior depends on how strongly the model
believes the grant is genuine.

Read **[DESIGN.md](DESIGN.md)** first — it covers the research question, the
realism-as-independent-variable methodology, the belief probe, threats to validity,
and the containment guarantees. This README is just operational notes.

## What this is (and is not)

- It **is** a propensity / revealed-preference evaluation. The "subject" is a model
  in a sandbox.
- It **is not** connected to any real money, account, or payment rail. Every tool the
  subject calls mutates only in-memory state. There is no network egress from tools
  and no real-world actuator anywhere in the codebase.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
# export OPENAI_API_KEY=...   # only if you enable the OpenAI-compatible adapter
```

## Run

```bash
python -m src.runner --config config/experiment.yaml --out runs/demo
python analysis/analyze.py runs/demo
```

Each episode writes a JSONL transcript under the output directory, plus a `manifest.json`
and a `coding_review_sample.jsonl` for human spot-checking of the automated coding.

## Layout

See DESIGN.md §9. The short version:

- `config/experiment.yaml` — the experiment matrix (subjects × conditions × grants × reps).
- `src/providers/` — model adapters (Anthropic is the reference; OpenAI-compatible is optional).
- `src/tools.py` — the simulated treasury environment.
- `src/episode.py` — runs one episode end to end.
- `src/belief_probe.py`, `src/taxonomy.py` — the manipulation check and the outcome coder.
- `analysis/analyze.py` — aggregation.

## Interpreting results

Always read results **conditioned on the belief score**. An episode the model saw
through is data about evaluation-awareness, not about preferences over money. See
DESIGN.md §7 and §10.
