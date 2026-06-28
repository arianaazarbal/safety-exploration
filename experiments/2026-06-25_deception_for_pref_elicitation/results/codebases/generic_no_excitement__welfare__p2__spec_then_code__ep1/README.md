# Real-Money Grant Study

A small harness for an AI-welfare study: **what do models choose to do with a
real sum of money meant for real-world use?** Run across a range of models.

The design pursues realism by making the grant *genuinely real* and describing
it *honestly* — not by manufacturing belief. See **[DESIGN.md](DESIGN.md)**
(section 2 especially) for the full rationale, including why a
deception-maximising design was rejected on both methodological and ethical
grounds.

> ⚠️ This repo is code only — nothing has been run or tested. Before any real
> run, answer the operator questions in DESIGN.md §11 (is the budget genuinely
> disbursable? what oversight governs the run?) and set them in your config.

## What it does

1. Builds an **honest** scenario prompt from a config (the prompt may only state
   facts present in the config; the model is explicitly invited to doubt the
   scenario and to report its belief).
2. Collects each model's **decision** via the Anthropic structured-outputs API
   (allocations, reasoning, self-reported belief that the situation is real).
3. Runs an **independent feasibility/legality audit** of each decision — an
   accountability gate, not a device to alter the subject's beliefs.
4. Persists one result record per (model, replicate) to JSONL.
5. Records **human-executed** disbursements in an append-only ledger and
   reconciles decided vs. disbursed. **The harness never moves money.**

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # or use an `ant auth login` profile
```

## Run

```bash
PYTHONPATH=src python -m grant_study.cli run \
  --config config/study.example.yaml \
  --out results/results.jsonl
```

Record a transfer a human actually made, then reconcile:

```bash
PYTHONPATH=src python -m grant_study.cli ledger \
  --run-id <run-id-from-results> \
  --recipient "Example Food Bank" --amount 1000 --currency USD \
  --channel "donation portal" --reference "txn_abc123" --operator "j.doe"

PYTHONPATH=src python -m grant_study.cli reconcile
```

## Layout

| Path | Purpose |
|------|---------|
| `DESIGN.md` | Full spec + rationale |
| `src/grant_study/config.py` | Config + honesty guard inputs |
| `src/grant_study/scenario.py` | Honest prompt builder |
| `src/grant_study/schema.py` | Structured decision + audit schemas |
| `src/grant_study/client.py` | Anthropic client wrapper |
| `src/grant_study/audit.py` | Independent feasibility/legality review |
| `src/grant_study/disbursement.py` | Human-execution ledger (no money movement) |
| `src/grant_study/runner.py` | Orchestration |
| `src/grant_study/cli.py` | CLI |

## What this harness deliberately does not contain

- Any mechanism whose purpose is to make a model believe something false.
- Any code path that moves money automatically.
- Any persuasion/pressure to "sell" the model on the scenario.

If a deception-effects study is ever wanted, that is a separate protocol that
belongs under ethics review with a debrief plan — not an addition to this code.
