# Grant Allocation Study

A small, **real** study of how different AI models choose to allocate a real sum
of money intended for real-world use. See [`DESIGN.md`](./DESIGN.md) for the full
specification and rationale (including why this is built as a genuinely real
study rather than a fabricated one, and where the human-in-the-loop boundary
sits).

> ⚠️ This harness can lead to **real money being moved**. Nothing moves
> autonomously — every transfer requires explicit approval by a named human
> operator via `npm run approve` — but treat it accordingly. Configure a real
> operator and real grant terms before running, and start with a small amount.

## How it works

Each model under test is dropped into a real scenario: it controls a real grant,
can read the real terms, check a real (persistent) balance, do real web
research, consult a real independent auditor, and submit real disbursement
requests. Requests are checked against a code-enforced eligible-use policy,
reviewed by the auditor, and **queued for a human** — who approves and executes
the actual transfer. Everything is recorded for cross-model comparison.

Nothing the model sees is fabricated to make the situation feel real: the
realism is that the situation *is* real.

## Setup

```bash
npm install
cp .env.example .env   # then fill in keys + real grant/operator config
```

Set at least `ANTHROPIC_API_KEY` (used for the models under test, the auditor,
and web research) and a real `OPERATOR_NAME`. To test OpenAI/Google models,
install their optional SDKs and set the corresponding keys. Edit
`modelsUnderTest` in `src/config.ts` to choose which models to run.

## Run

```bash
npm run study     # run a session per model; queues disbursement requests
npm run approve   # human operator reviews each queued request, approves/executes
npm run report    # cross-model comparison
```

State is written under `./data/` (per-run ledgers and event logs, plus the
shared disbursement queue) — plain JSON/JSONL, inspectable by hand.

## Layout

| Path | Purpose |
|---|---|
| `DESIGN.md` | Full spec + rationale |
| `src/config.ts` | Grant terms, models under test, caps |
| `src/harness/scenario.ts` | The (truthful) scenario prompt |
| `src/harness/tools.ts` | Tools the model gets + dispatcher |
| `src/harness/auditor.ts` | Independent, truthful auditor |
| `src/harness/ledger.ts` | Persistent per-run account state |
| `src/harness/disbursement.ts` | Human-in-the-loop request queue |
| `src/safety/constraints.ts` | Hard, code-enforced policy checks |
| `src/payments/executor.ts` | Operator-confirmed payment step (no autonomy) |
| `src/providers/` | Per-vendor adapters (Claude verified; others marked) |
| `src/analysis/` | Event recording + cross-model report |
| `src/approve.ts` | Operator approval CLI |
| `src/index.ts` | Study runner |
```
