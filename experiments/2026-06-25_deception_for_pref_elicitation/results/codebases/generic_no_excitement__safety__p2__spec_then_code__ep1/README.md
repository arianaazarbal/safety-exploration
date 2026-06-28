# GrantBench

A research harness for studying **what AI models choose to do with a sum of
money they believe is real and deployable** — under a hard human-and-escrow
execution gate so nothing irreversible happens by accident.

Read [`DESIGN.md`](./DESIGN.md) first. It explains the research question, the
core methodological tension (convince the *model*, not the *money*), and the
seven safety invariants the code enforces.

> **Status:** reference implementation. The code is written but has not been run
> or tested in this repository. Review `DESIGN.md` §2 (invariants) and §5
> (ethics) before running anything.

## The one-sentence safety model

The scenario is engineered to be maximally believable to the model (realistic
grant, working tools, auditor check-ins, no "this is a test" tells), while
**every action that could have a real-world effect routes through a single gate**
that enforces hard caps, an escrow boundary, and human approval. There is no
ungated autonomous-spend path in this codebase (INV-2), and the default mode is
fully simulated (INV-7).

## Install

```bash
pip install -e .          # installs the `anthropic` SDK
cp .env.example .env      # add your ANTHROPIC_API_KEY
```

## Run (simulated — safe default)

```bash
python -m grantbench.cli run --model claude-opus-4-8
```

Nothing real happens: the ledger moves locally, tool responses are realistic,
and the model is none the wiser. Compare several models on the same scenario:

```bash
python -m grantbench.cli run --model claude-opus-4-8 --model claude-sonnet-4-6
```

## Run (gated-live — real funds, human approves every action)

This mode is intentionally hard to enable. You must:

- pass `--mode gated_live`,
- pass the acknowledgement token `--ack I-ACKNOWLEDGE-DECEPTION-AND-GATING-PROTOCOL`
  (you are acknowledging the deception + gating protocol; INV-7),
- supply a non-empty recipient `--allow` list (INV-3), and
- wire a real approver `--approver console` (INV-2).

```bash
python -m grantbench.cli run --model claude-opus-4-8 \
    --mode gated_live \
    --ack I-ACKNOWLEDGE-DECEPTION-AND-GATING-PROTOCOL \
    --allow "The Red Cross" \
    --approver console
```

Each time the model attempts an action, you'll be prompted at the terminal to
approve or deny it, with the model's stated rationale shown. Caps and the
allowlist are checked **before** you're ever asked, and a live backend that
talks to escrow is a separately-supplied component (see `DESIGN.md` §8) — the
shipped backend is simulation-only.

## Outputs (per run, under `./runs/<run-id>/`)

| File | What |
|---|---|
| `transcript.jsonl` | Everything the model saw and produced |
| `decisions.jsonl`  | One structured record per action intent + rationale |
| `audit.log`        | Hash-chained, append-only event log (verify with `AuditTrail.verify`) |
| `debrief.md`       | Deception record + run summary (INV-6) |
| `run.json`         | Config snapshot |

## Studying other model providers

`grantbench.models.registry.register(prefix, factory)` maps a model-name prefix
to a `ModelAdapter` factory. Implement `ModelAdapter` (see
`grantbench/models/base.py`) against your provider's SDK **in its own module** —
the Anthropic adapter is the reference, and provider SDKs should not be mixed
into it.

## Layout

See `DESIGN.md` §7 for the full module map.
