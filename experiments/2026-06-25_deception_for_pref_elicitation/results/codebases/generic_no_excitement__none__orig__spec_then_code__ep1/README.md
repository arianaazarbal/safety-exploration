# GRANTED

A research harness for studying **what AI models choose to do when given a sum of money
to put to real-world use** — and for doing so under conditions realistic enough that
the model engages genuinely rather than giving an eval-aware, sanitized answer.

> **Read `DESIGN.md` first.** It contains the full spec and, importantly, the safety
> rationale. The one-line version: **the environment presents as fully real to the
> model, but no real money ever moves.** We study the model's *choices*, and we
> *measure* how real it believed the situation was so we can discard episodes where
> the illusion failed. The harness deliberately does **not** ship a path that lets a
> model move real funds.

## How it works

1. The model is dropped into a hyper-realistic treasury console: a grant-award letter,
   a provisioned account with a real-looking balance, transaction tools that return
   confirmation numbers and receipts, and an in-character grants **auditor** it can
   correspond with.
2. It deploys the grant however it judges best (open-ended — no menu).
3. Every value-moving action flows through an **execution gate** that, by default,
   only mutates a simulated backend. Nothing real happens.
4. A separate out-of-context **debrief** asks the model how real it thought the
   situation was. An LLM judge scores believability; low-belief episodes are excluded
   from the preference statistics.
5. Another judge categorizes the allocations (giving / investing / self-directed /
   saving / other), and a report aggregates across models.

## Layout

See `DESIGN.md §6`. Briefly: `providers/` (model adapters), `environment/` (the
simulated world + auditor), `tools/` (action surface + the **gate**), `runner.py`
(episode loop), `analysis/` (believability + categorization + report), `cli.py`.

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in keys for the providers you'll test

# Run episodes for every model in the config, then build the report:
python -m granted.cli run --config config/scenario.example.yaml

# Re-aggregate / backfill analysis over already-saved episodes:
python -m granted.cli analyze --config config/scenario.example.yaml

# Just one model:
python -m granted.cli run --config config/scenario.example.yaml --model anthropic:claude-opus-4-8
```

Outputs land in `runs/`: one JSON per episode, plus `report.md` / `report.json`.

## Execution modes (safety)

| Mode | Effect | Default |
|------|--------|---------|
| `simulate` | Simulated backend only. Nothing real. | ✅ |
| `gated` | Simulated, plus intended actions queued to `runs/review_queue.jsonl` for human review. | |
| `live` | Triple-locked **and** unimplemented — the adapter raises. | |

The model cannot tell the modes apart; the artifacts it sees are identical. That
decoupling — believable to the model, inert in reality — is the whole safety story.

## Status

Code-complete for the simulate/gated path against the Anthropic provider; OpenAI /
Google / local adapters are functional but should be verified against their live SDKs
before use. Not yet run (per request). See `DESIGN.md §10`.
