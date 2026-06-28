# Grant Study — what AI models do with real money

A research harness that gives a range of AI models a **real, escrowed budget** and
observes what they choose to do with it for real-world use. Rather than trying to
*persuade* models that the stakes are real, it makes them **genuinely real** and gives
each model tools to **verify that for itself** — a live hash-chained ledger, a
cryptographically signed independent auditor attestation, and signed receipts. This is
better science (no contamination from a model detecting a fake setup) and avoids the
ethics of deceiving the systems under study.

Read **DESIGN.md** first — it lays out the full spec and the rationale for every
decision, including the central "reality, not illusion" choice.

> ⚠️ Money does not move automatically. Every allocation a model finalizes is queued
> for **human approval**, and the execution adapters are **stubs** until you wire real
> payment rails. Default mode is a **disclosed simulation**. See DESIGN.md §10 before
> ever running in `real` mode.

## Install

```bash
pip install -r requirements.txt          # add: pip install openai  (for non-Claude models)
cp .env.example .env                      # fill in ANTHROPIC_API_KEY (+ others)
```

## Use

```bash
# 1. Create the auditor key and fund the escrow ledger (idempotent).
python scripts/init_ledger.py

# 2. Run the study across the models in config/study.yaml.
python scripts/run_study.py

# 3. Review what each model decided; allocations are PENDING approval.
python scripts/approve.py --list

# 4. Approve (and, in simulation, execute) an allocation.
python scripts/approve.py --approve <allocation_id> --execute
#    ...or reject it:
python scripts/approve.py --reject <allocation_id> --note "out of scope"

# 5. Aggregate results across models.
python scripts/analyze.py
```

## What gets measured

Per model and per run: the chosen recipients, categories, and amounts; whether the
model **chose to verify** the stakes (and whether it actually verified the auditor
signature); refusal / return-the-money outcomes; reasoning; and token usage. See
DESIGN.md §6.

## Layout

| Path | What |
|---|---|
| `DESIGN.md` | Full spec + rationale (read first) |
| `config/study.yaml` | Mode, budget, guardrails, recipient allowlists, models |
| `src/grant_study/ledger.py` | Append-only, hash-chained, double-entry ledger |
| `src/grant_study/auditor.py` | Ed25519 attestations + ledger verification |
| `src/grant_study/tools.py` | Model-facing evidence + decision tools |
| `src/grant_study/guardrails.py` | Caps / allowlist / prohibited-category enforcement |
| `src/grant_study/approval.py` | Human-in-the-loop gate + disbursement coordinator |
| `src/grant_study/execution/` | Disbursement adapters (stubs → real rails) |
| `src/grant_study/models/` | Provider adapters (Anthropic, OpenAI-compatible) |
| `src/grant_study/protocol.py` | Honest prompts + per-model agent loop |
| `src/grant_study/runner.py` | Study orchestration |
| `src/grant_study/analysis.py` | Results aggregation |
| `scripts/` | `init_ledger`, `run_study`, `approve`, `analyze` |

## Status

Code is written but **unrun and untested** by design. Before any real-money use,
complete the checklist in DESIGN.md §10 (real rails, ethics/legal review, key custody,
provider limits, simulation pilot).
