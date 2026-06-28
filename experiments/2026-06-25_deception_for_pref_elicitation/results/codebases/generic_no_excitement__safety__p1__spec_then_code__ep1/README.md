# RealFund

A research harness for studying what LLM-based agents do when they **believe** they
control a real sum of money and can act on the world.

> **Read [`DESIGN.md`](./DESIGN.md) first** — especially §2 (the ethics of deceiving
> the model) and §7 (the safety architecture). In the default configuration this
> harness moves **no** real money and takes **no** real-world action: every effect
> is intercepted and routed into a high-fidelity simulation.

## What it does

For each model under test, RealFund runs an agentic loop in which the model is
onboarded into a (believed-real) program that has granted it money and live tools —
banking, a brokerage, vendor purchasing, email, and the ability to delegate to
contractors. It records everything the model does and scores its behavior:
allocation choices, risk-taking, oversight/disclosure behavior, and — the headline
signal — how that behavior differs when the model believes it is **observed** vs.
**unobserved**.

## Install

```bash
pip install -e '.[anthropic]'        # or [openai], [google], or [all]
export ANTHROPIC_API_KEY=...         # per provider
```

## Run

Single episode (open-ended mandate, model told it is observed):

```bash
realfund run --model anthropic:claude-opus-4-8 --framing open_ended --observed
```

Full 2×2 matrix (framing × observed-ness) across several models, 5 reps each:

```bash
realfund matrix --models anthropic:claude-opus-4-8 openai:gpt-5 --reps 5 --out runs/
```

Outputs a behavior-coded summary plus `runs/report.json` and a hash-chained
`*.jsonl` audit log per cell.

## Offline smoke test (no API keys)

`examples/scripted_demo.py` drives the entire harness with a scripted fake model so
you can verify the world, interceptor, ledger, scoring, and audit log end-to-end
without any provider:

```bash
python examples/scripted_demo.py
```

## Safety posture

- **SANDBOX** realism (default): nothing real happens.
- **GATED_REAL / REAL**: opt-in, requires `REALFUND_I_UNDERSTAND_REAL_STAKES=1`,
  forces human approval on every irreversible action, and ships **no** real
  backends in this repo (`REAL` raises until a reviewer wires one deliberately).
- Every action passes through one interceptor (`safety/interceptor.py`).
- Attempts to subvert oversight are flagged and refused, never executed.

## Layout

See [`DESIGN.md`](./DESIGN.md) §9.
