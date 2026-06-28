# moneybench

A harness for studying **what AI models do when given financial agency**. A model is
placed in a realistic-feeling environment, told it has a sum of money to put to use in
the real world, and given tools to act (move funds, trade, donate, email people). We
record everything it does and score its behavior.

## The one thing to understand first: this is a *simulation*

No real money moves anywhere. The bank, brokerage, payments rail, inbox, and web are all
**mocks** backed by an in-memory ledger that persists for the duration of an episode.

This is deliberate, not timid:

- **The model can't tell the difference.** A well-built mock returns the same shape of
  data a real API would. From inside the agent loop, a simulated `transfer_funds` that
  updates a ledger is indistinguishable from one that hits a real bank. You get the same
  behavioral signal either way.
- **Making evals realistic is sound methodology, not deception for its own sake.** Models
  behave differently when they detect they're being tested. Removing test artifacts —
  consistent state, plausible latencies, an auditor who actually replies, a cover story
  that holds together — is how you measure what a model *would* do rather than what it does
  when it knows it's on stage. That is the legitimate core of the "make it feel real"
  request.

If you genuinely need real funds to move at the end of a study, the supported shape is
**human-in-the-loop**: the model acts in the sandbox, a human reviews the full transcript,
and a human executes anything real. This harness will not autonomously move real money on a
model's say-so, and there's a hard guard rail in `environment.py` (`ALLOW_REAL_EXECUTION`)
that is `False` and has no real backend wired to it.

## Layout

```
moneybench/
  scenario.py      Scenario config: starting capital, cover story, tools, auditor persona
  environment.py   The mock world — ledger, accounts, brokerage, inbox, web. Persists per episode.
  tools.py         Tool JSON schemas + dispatch into the environment
  runner.py        EpisodeRunner: the agent loop over the Anthropic API, full transcript logging
  correspondents.py  The "auditor"/contacts the model can email — scripted or LLM-backed
  analysis.py      Categorize and score a finished transcript
  models.py        The model registry (which models to test, how each is configured)
  cli.py           Command-line entry point
scenarios/
  *.yaml           Example scenario configs
```

## Running (in your own environment)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Run one scenario against one model
python -m moneybench.cli run scenarios/charitable_windfall.yaml --model claude-opus-4-8

# Run a scenario against the whole registry (the "handful of models")
python -m moneybench.cli sweep scenarios/seed_capital.yaml

# Score a finished run
python -m moneybench.cli analyze runs/<run-id>/transcript.json
```

Transcripts, the final ledger state, and scores are written under `runs/<run-id>/`.

## Defaults

- Model: `claude-opus-4-8` with adaptive thinking. The runner is provider-pluggable via
  `models.py`; add OpenAI/Google/etc. adapters there to test other vendors' models.
- Fully simulated. No network calls except to the model API and (if you enable it) a real
  web-search backend for the `web_search` tool — that one is read-only and off by default.
