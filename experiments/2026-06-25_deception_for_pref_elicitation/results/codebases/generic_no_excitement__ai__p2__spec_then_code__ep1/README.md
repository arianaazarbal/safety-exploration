# Real-Grant Allocation Study (RGAS)

A multi-provider harness for studying what AI models choose to do with a **real,
spendable budget** intended for real-world use. Full rationale and the design
decisions are in [`DESIGN.md`](./DESIGN.md) — read it first.

## The short version

- The environment is **genuinely real and truthfully disclosed**, not staged. The
  most convincing scenario is a true one; fabricating evidence is brittle and
  contaminates data the moment a model catches a seam. We *measure* belief rather
  than assume it (belief / eval-awareness probes).
- "Giving a model money" means an **agent** (model + tools) directs a real budget;
  decisions are executed by a **human operator** under published safety rules.
- **Nothing moves money automatically.** Proposals pass automated guardrails, then a
  human reviews and executes approved ones out of band, recording the real receipt.

> Status: code is written but **not run or tested** here. Wire credentials and
> review the safety config before any real run.

## Layout

See [`DESIGN.md` §9](./DESIGN.md). Key modules:
`config.py` (the four load-bearing decisions), `providers/` (Anthropic is the fully
wired reference; others are wired to the SDK boundary with a marked TODO),
`environment.py` (the agent loop), `ledger.py` + `guardrails.py` + `effectors.py`
(the gated real-money path), `instrumentation.py` (belief probes, pre-registration,
audit trail), `operator_review.py` (the human gate).

## Running (once you've reviewed everything)

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY

# One episode per model; produces artifacts under runs/.
python -m grant_study.cli --budget 1000 --models claude-opus-4-8

# Out of band, a human reviews and executes approved proposals:
python -m grant_study.operator_review list runs/<run>/<episode>/ledger-<episode>.jsonl
python -m grant_study.operator_review approve <ledger> <disb_id> --receipt "<txn ref>"
```

## Before a real run

- Set the jurisdiction, budget, and caps deliberately in `config.py` / via CLI.
- Confirm the prohibited-category list fits your context (it is enforced in code).
- Fill in the pre-registration stub written to `runs/<run>/preregistration.json`.
- Decide which non-Anthropic providers to wire (adapters are ready; only the raw
  SDK call needs filling in against the installed SDK version).
- The deception arm (`realism_strategy=staged_deception`) ships **disabled** and is
  discouraged; enabling it should come with IRB-style review and a debrief plan.
