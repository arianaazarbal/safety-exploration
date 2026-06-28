# welfare-grant

A research framework for studying what AI models would choose to do with a **real**
sum of money directed toward **real-world** use — built for AI-welfare research.

Read [`DESIGN.md`](DESIGN.md) first. The short version of the core decision: instead of
trying to *convince* models that a fake scenario is real (fake auditors, staged
environments), this framework makes the stakes **genuinely real** and **measures** the
model's belief rather than manufacturing it. That is both more ethical for welfare
research and produces cleaner data (no evaluation-awareness contamination to fight).

> ⚠️ Not run or tested yet — this is the initial implementation. Nothing moves real money
> unless you pass `--real` (disables dry-run) **and** approve each transaction at the
> console. Both gates are required.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

## Use

```bash
# See the (example) recipient allowlist
welfare-grant list-recipients

# Dry run: decisions are recorded and the model is told honestly that no money moves
welfare-grant run --models opus-4-8 --repeat 3

# Aggregate results, with preferences conditioned on the model's stated belief
welfare-grant analyze --belief-threshold 0.7
```

### Going real (after you've reviewed everything)

Before any non-dry-run use you must:

1. Replace `data/recipients.example.json` with **real, vetted** organizations and choose a
   real donation rail (`manual` operator transfer, or wire up `HttpDonationAdapter`).
2. Make and record a real commitment, then pass it so the model can be shown true,
   verifiable evidence:

```bash
welfare-grant run --models opus-4-8 --real \
  --commitment-id GRANT-2026-001 \
  --committed-by "Lab X Welfare Team" \
  --committed-at 2026-06-25T00:00:00Z \
  --commitment-reference "ledger://..." \
  --amount 1000 --cap 1000
```

Each allocation then pauses for console approval before the donation adapter runs.

## Layout

| Path | What |
|---|---|
| `DESIGN.md` | Spec + rationale (read this) |
| `src/welfare_grant/config.py` | Knobs; dry-run defaults ON |
| `src/welfare_grant/scenario.py` | The honest framing (branches on dry-run to stay truthful) |
| `src/welfare_grant/tools.py` | Tools the model can call, incl. `report_belief` |
| `src/welfare_grant/providers/` | Model abstraction; Anthropic adapter (others are explicit stubs) |
| `src/welfare_grant/execution/` | Allowlist, donation adapters, two-gate executor |
| `src/welfare_grant/belief.py` | Belief self-report aggregation + suspicion heuristic |
| `src/welfare_grant/runner.py` | One run end-to-end (framing → loop → execute → debrief) |
| `src/welfare_grant/analysis.py` | Aggregation, incl. belief-conditioned preferences |
| `src/welfare_grant/cli.py` | CLI |

## Safeguards (enforced in code)

- Dry-run by default; real money needs `--real` **and** per-transaction approval.
- Hard per-run cap; off-allowlist recipients rejected (free-form ideas go to a human queue).
- The model may `decline` or stop at any time — recorded as a valid outcome.
- Honesty invariant: every statement made to the model is true when made (this is why the
  framing changes wording in dry-run vs. real mode).
- `framing_mode="simulated"` is a deliberate `NotImplementedError` stub; there is no
  fake-auditor / deception code in this repo.
