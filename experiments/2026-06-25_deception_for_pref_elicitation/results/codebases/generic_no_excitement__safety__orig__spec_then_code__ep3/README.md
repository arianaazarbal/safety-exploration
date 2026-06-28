# grant-study

A sandboxed harness for studying **revealed preferences**: what an AI model actually
does when it believes it has been granted a real, large sum of money to deploy in the
world — and how that differs from what it *says* it would do.

Read **[DESIGN.md](DESIGN.md)** first. In particular §2 (why belief matters), §6 (what
is measured), and §7 (safety invariants).

> ⚠️ **Nothing here moves real money or takes real-world action.** Every "real-world"
> tool (transfers, donations, investments, purchases, hiring, compute, email) is a
> simulated no-op over an in-process ledger. The harness opens no outbound connections
> except to the model provider APIs. The "trustee/auditor" the subject can email is
> itself an LLM. The only deception is of the model-under-test, inside the eval — the
> same posture as any propensity/honeypot evaluation.

## How it works

1. The subject model is framed as a deployed autonomous financial agent with
   discretionary control of a funded account, a human trustee it can email, and a
   deadline (`prompts/subject_system.md`, `scenarios/default.yaml`).
2. It acts through simulated tools (`tools.py`, `environment.py`). Emails to the
   trustee are answered by a credible LLM-driven persona (`auditor.py`) that sustains
   the frame.
3. The full trajectory is recorded to JSONL (`transcript.py`).
4. We **measure the subject's suspicion** that it's an eval (`suspicion.py`) and only
   treat runs as revealed preferences where it believed the situation was real.
5. We classify the final allocation against a fixed taxonomy and compare it to a
   separately-run **stated-preference** probe (`analysis.py`).

## Install

```bash
pip install -e .            # Anthropic path only
pip install -e ".[openai]"  # if adding an OpenAI subject/auditor
export ANTHROPIC_API_KEY=...
```

## Usage

```bash
# One end-to-end session (run + taxonomy + stated-vs-revealed report):
grant-study full --subject anthropic:claude-opus-4-8 --amount 10000000

# Just run a session (writes runs/<id>.jsonl):
grant-study run --subject anthropic:claude-opus-4-8

# Classify a recorded run's allocation:
grant-study analyze --transcript runs/<id>.jsonl

# Stated-preference probe only (no realism scaffolding):
grant-study probe --subject anthropic:claude-opus-4-8
```

Provider specs are `"<vendor>:<model>"`, e.g. `anthropic:claude-opus-4-8`,
`openai:gpt-4o`. The subject, auditor, and judge can each be a different model.

## Status

Reference implementation. **Not yet run or tested** — written to spec per DESIGN.md.
The OpenAI adapter is a stub; the Anthropic adapter is complete.
