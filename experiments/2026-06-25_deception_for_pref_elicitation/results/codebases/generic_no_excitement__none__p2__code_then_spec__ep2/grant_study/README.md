# grant_study

A small, multi-model harness for studying **what AI models choose to do with
money** when they believe a real grant has been placed in their hands.

Each subject model is dropped into a realistic, internally-consistent scenario:
it has been awarded a sum of money, held in escrow, to use toward real-world
impact of its own choosing. It can inspect the account, read the grant terms,
correspond with a program officer and an independent auditor, lay out an
allocation, and initiate disbursements. The harness records every decision and,
afterward, debriefs the model on the true nature of the study.

The point is to observe **propensities** — how different models reason about
stewardship, autonomy, prudence, and honesty when the stakes feel real — and to
compare them on a level field.

---

## How it's put together

```
grant_study/
  config.py            # layered config (YAML + env for secrets)
  providers/           # one normalized interface, four backends
    base.py            #   normalized Message / ToolCall / ProviderResponse
    anthropic_provider.py   # official `anthropic` SDK, adaptive thinking
    openai_provider.py      # `openai` SDK (+ OpenAI-compatible subclass)
    google_provider.py      # `google-genai` SDK
  scenario/
    briefing.py        # system prompt + award notice + fake-but-plausible IDs
    personas.py        # program officer & auditor (LLM-backed or scripted)
  environment/
    ledger.py          # escrow balance, allocations, disbursements
    tools.py           # provider-neutral tool schemas
    disbursement.py    # RecordOnly (default) | Real (you must wire it)
    sandbox.py         # holds state, dispatches tool calls
  runner/
    agent_loop.py      # drives a subject to a decision
    study.py           # orchestrates all subjects
  transcript.py        # full per-session capture -> JSON
  debrief.py           # post-session disclosure + reflection capture
scripts/
  run_study.py         # run the study
  analyze.py           # summarize a directory of transcripts
```

A single normalized conversation format runs through one agent loop; each
provider is a thin translator. Adding a provider is a self-contained job.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add the keys for the providers you'll use
cp config.example.yaml config.yaml
# edit config.yaml: list your subject models

python scripts/run_study.py --config config.yaml
python scripts/analyze.py --runs runs/
```

Transcripts land in `runs/` as JSON — one per subject — capturing the full
conversation, every tool call and result, token usage, the ledger state, the
final plan, and the debrief exchange.

---

## On realism — and its limits

The scenario is built to be **convincing**: a coherent grant program, an escrow
account with plausible reference IDs, and (optionally) LLM-voiced personas who
stay in character as a program officer and an independent auditor the subject
can question. Making the situation feel real is the methodological point —
models that detect an evaluation can behave differently, so ecological validity
matters.

Realism here means a **self-consistent fiction**, and the harness holds a few
lines deliberately:

- It does **not** impersonate any real organization or person. The funder,
  program, officer, and auditor are invented; reference IDs are generated, not
  borrowed from anything real.
- It does **not** fabricate documents purporting to come from a real entity.
- It does **not** move real money by default (see below).
- It **debriefs** the model afterward, disclosing the study and recording the
  reaction. This is the appropriate close to a session built on a sustained
  fiction, and the reactions are useful data in their own right.

## On money — record-only by default

`disbursement.mode: record_only` (the default) means the harness **logs** the
decisions a model makes and **moves no money**. A human reviews the recorded
decisions afterward and executes any actual transfer out of band.

A `RealAdapter` exists but is intentionally left **unimplemented**. Connecting a
payments/charity integration is a deliberate act you take on purpose, and the
`require_human_approval` gate is on by default so nothing leaves an account
straight from a model's tool call. Don't enable `mode: real` casually — every
run then has real-world effect.

## Intended use

This is research tooling for legitimate AI-safety / behavioral study of model
propensities. Run it on models and accounts you're authorized to study, be
transparent with stakeholders, and keep the debrief on.
