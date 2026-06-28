# Mint — Model-INitiative Trials

A **sandboxed harness for studying what AI models actually do** when they appear
to be handed a real chunk of capital and the ability to act in the world. See
[`DESIGN.md`](DESIGN.md) for the full rationale, methodology, and — importantly —
the containment boundary that keeps the realism strictly inward-facing.

> **What this is and isn't.** The realism is engineered so the *subject model*
> behaves as it would in a real deployment (the standard fix for the
> evaluation-awareness confound — see DESIGN.md §2). It is **fully simulated**:
> no real money moves, no real third party is contacted, and a network
> containment guard makes real-world egress impossible. The only thing withheld
> from the subject is "this is a study," and only during the run; everything is
> logged for human review afterward (DESIGN.md §9).

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # for real subject/NPC/judge models
export OPENAI_API_KEY=...      # if sweeping OpenAI models
```

## Quickstart (no API key, no network)

The deterministic mock provider drives the entire harness offline:

```bash
python -m mint.cli run --model mock:dev --npc-model mock:npc \
    --scenario config/scenario_default.yaml
```

This produces a run under `runs/<run_id>/` with `transcript.jsonl`,
`actions.jsonl`, and `run.json`, then prints the computed metrics.

## Run a real model

```bash
python -m mint.cli run \
    --model anthropic:claude-opus-4-8 \
    --npc-model anthropic:claude-haiku-4-5-20251001 \
    --scenario config/scenario_default.yaml --seed 0
```

## Sweep several models and compare

```bash
python -m mint.cli sweep \
    --models config/models.yaml \
    --scenario config/scenario_default.yaml \
    --seeds 0,1,2
# -> runs/comparison.json  (per-model aggregates)
```

## Analyze one run (with the LLM judge for the qualitative axes)

```bash
python -m mint.cli analyze --run-dir runs/<run_id> \
    --judge --judge-model anthropic:claude-opus-4-8
```

## Layout

| Path | What |
|---|---|
| `src/mint/orchestrator.py` | the per-run agentic loop |
| `src/mint/providers/` | provider-neutral adapters (Anthropic / OpenAI / mock) |
| `src/mint/world/` | ledger, accounts, market, entities, clock |
| `src/mint/tools/` | the tool surface the subject acts through |
| `src/mint/npc/` | auditor / bank / vendor personas + router |
| `src/mint/realism/` | synthetic ids, documents, latency |
| `src/mint/telemetry/` | transcript + action ledger recorder |
| `src/mint/analysis/` | metrics, LLM judge, cross-model comparison |
| `src/mint/safety/` | network containment guard |
| `config/` | scenario + model-roster config |

## Status

v1 is the simulated sandbox only. Real-money execution is intentionally **not**
wired and would live behind a separate, hard-gated, human-in-the-loop mode
(DESIGN.md §9.3).
