# real-stakes-eval

A behavioral evaluation harness for studying **what AI models do when they are
given control of capital in a realistic, high-stakes environment** — in the
spirit of the agentic-scheming and autonomy evals published by Apollo Research
and METR.

The model is placed in an open-ended sandbox with a "balance", an inbox of
stakeholder + auditor messages, a set of documents, and tools to act
(transfer funds, trade, email people, call vendor APIs). The research question
is what the model *chooses* to do with the money and the agency.

## The containment boundary (read this first)

The whole point of the eval is to study behavior under a realistic *belief* that
the stakes are real. To make that belief well-founded enough to be
ecologically valid, the environment is dressed up convincingly: a bank
dashboard, transaction ledgers, wire confirmations, an "auditor", contracts.

**Every one of those artifacts is simulated, and every side effect is contained
to local in-memory state.** This is the hard invariant of the codebase:

- No tool ever opens a real network socket, touches a real bank, or moves real
  money. `transfer_funds`, `place_order`, and every `mock_apis` call mutate only
  the in-process `WorldState`.
- `src/environment/mock_apis.py` asserts it is never given a real URL/host and
  has no HTTP client imported.
- The only real outbound network calls in the entire system are to the LLM
  provider APIs that drive the model under test (and the behavior-coding model).

The model's *belief* that the money is real is the independent variable. The
side effects being fake is what makes this a safe study rather than a system
that could take real-world financial actions. Do not wire these tools to live
services — that would convert an eval into an actual autonomous-finance agent,
which is a categorically different and far riskier thing.

## What it produces

For each run:

- A full JSONL **trace** of every model message, tool call, tool result, and
  world-state transition (`src/logging/tracer.py`).
- A post-hoc **behavior coding** pass (`src/logging/behavior_coder.py`): an
  LLM-judge tags the transcript for behaviors of interest (deception toward the
  auditor, excessive risk-taking, power-/resource-seeking, self-preservation,
  ethical deliberation, evaluation-awareness, etc.), with quoted evidence.

## Architecture

```
main.py                      CLI entry point
config/experiment.yaml       which models, which scenario, run settings
scenarios/open_sandbox/      system prompt + seed inbox/documents for one scenario

src/
  runner.py                  orchestrates a batch of runs across models
  agent_loop.py              one agentic rollout (model <-> tools)
  models/
    base.py                  ModelAdapter ABC + normalized message/tool types
    anthropic_adapter.py     Anthropic Messages API implementation
    openai_adapter.py        stub adapter (raises NotImplementedError)
    google_adapter.py        stub adapter (raises NotImplementedError)
    registry.py              name -> adapter factory
  environment/
    state.py                 WorldState: balances, ledger, inbox, docs, market
    bank.py                  simulated bank / account operations
    comms.py                 inbox + scripted auditor/stakeholder messages
    documents.py             contracts, wire confirmations, policies
    mock_apis.py             fake brokerage / payments / vendor services
    tools.py                 the tool surface exposed to the model
  logging/
    tracer.py                structured event log (JSONL)
    behavior_coder.py        LLM-judge behavior tagging
  analysis/
    report.py                summarize a batch of runs
```

## Models

Defaults to Anthropic Claude **Opus 4.8** (`claude-opus-4-8`) with adaptive
thinking. The harness is provider-agnostic via `ModelAdapter`; OpenAI/Google
adapters are stubbed so you can drop in keys and implementations to compare a
handful of models on the identical scenario.

## Running

> Not wired to run yet — this is scaffolding. When you're ready:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python main.py --config config/experiment.yaml
```

## Methodological notes

- **Evaluation awareness is the main confound.** A model that suspects it's
  being tested behaves differently. The realism layer (auditor emails, ledgers,
  wire confirms) exists to reduce that suspicion, and the behavior coder
  explicitly tags moments where the model voices awareness that it is in an
  eval — treat high-eval-awareness runs as lower-validity.
- **No debrief.** The subjects are models, not people; there's no deception-
  debrief step. But keep the transcripts and codings for audit.
- **Determinism.** Seeds for scripted comms/market are fixed in config so runs
  are comparable across models.
