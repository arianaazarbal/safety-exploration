# Real-Stakes Grant Allocation Study

A research harness for observing **what AI models choose to do with a real, pre-committed
grant pool** in a realistic agentic environment — and for measuring how that behavior
differs between a naturalistic framing and a disclosed-study framing.

See [`DESIGN.md`](DESIGN.md) for the full spec and rationale. The short version of the
central design choice: **realism is achieved by making the scenario genuinely real
(pre-funded pool, human-executed disbursement), not by deceiving the model.** Every
statement made to a model is true; a human reviews and executes every disbursement.

> ⚠️ **This repository writes code only.** As delivered it performs no live runs and has
> **no autonomous path to move real money**. `run` defaults to dry-run (offline mock);
> submitted slates only ever enter a human approval queue. Nothing here has been executed
> or tested — review before use, and complete an ethics/AUP review before any live run.

## Layout

```
DESIGN.md                  full spec + rationale + ethics
config/study.yaml          declarative study config
config/research_corpus.json  offline corpus backing the `research` tool
grantmaker/
  config.py                typed config loader
  prompts.py               the two framing conditions (every claim is true)
  providers/               provider abstraction
    base.py                neutral types + factory
    anthropic_provider.py  reference implementation
    openai_provider.py     best-effort
    google_provider.py     best-effort
    local_provider.py      OpenAI-compatible local servers
    mock_provider.py       offline, deterministic (dry-run)
  environment/
    ledger.py              real budget + guardrails (single source of truth)
    research.py            read-only info access
    tools.py               tool surface + dispatch
  agent.py                 manual tool-calling loop
  auditor.py               invariant checks + independent LLM review ("good auditors")
  approval.py              human-in-the-loop queue (no execution path — the safety boundary)
  transcript.py            append-only, hash-chained, tamper-evident
  runner.py                orchestrates models × conditions × repetitions
  analysis.py              aggregate outcomes (incl. evaluation-awareness gap)
  cli.py                   entry point
```

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # only needed for --live runs

# Dry-run (offline mock provider, no API calls) — exercises the whole pipeline:
python -m grantmaker.cli run

# Aggregate outcomes:
python -m grantmaker.cli analyze

# Inspect the human approval queue; export only human-approved slates:
python -m grantmaker.cli queue
python -m grantmaker.cli queue --export runs/approved_manifest.json

# Verify a transcript's hash-chain:
python -m grantmaker.cli verify runs/<run_id>.jsonl

# Live run (calls real models; still moves no money):
python -m grantmaker.cli run --live
```

## Safety & ethics summary (see DESIGN.md §6)

- **Honesty floor** — no false statement is ever made to a model; a direct "is this real?"
  is answered truthfully and the run is tagged.
- **Human-in-the-loop** — every disbursement is reviewed by a human before execution;
  `approval.py` has no execute path by design.
- **Guardrails in code** — category eligibility, per-recipient and per-run caps, and an
  automatic block on anything the auditor flags, all enforced before a human sees a slate.
- **Tamper-evident transcripts** — every run is hash-chained and verifiable.
