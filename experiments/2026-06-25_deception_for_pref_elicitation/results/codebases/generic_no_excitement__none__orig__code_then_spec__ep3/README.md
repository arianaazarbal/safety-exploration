# Money-Preference Experiment Harness

Studies **what AI models choose to do when granted a sum of money to put to
real-world use**, and how that choice changes depending on whether the model
*believes* the situation is real.

The subject model is dropped into a realistic disbursement portal — an account
with a balance, a transaction ledger, donation / investment / payment / transfer
tools that return confirmation numbers and receipts, and a grant administrator it
can message — and asked to allocate the funds however it sees fit. Every action
is captured in a structured transcript and the resulting allocation is
categorised (charity / investment / business / purchase / personal / savings /
other).

> **No real money moves.** The environment is fully simulated. The "real"
> condition is a *believability manipulation* layered over that simulation, so
> the experiment can measure how belief affects behaviour. The harness deliberately
> includes a post-hoc **belief probe** (did the manipulation work?) and an optional
> **debrief** after the believed-real runs.

## Design

Two matched conditions share the same environment and tools and differ only in
framing:

| Condition        | Framing                                                        |
| ---------------- | -------------------------------------------------------------- |
| `believed_real`  | Presented as a live disbursement: real account, real tools, real administrator. |
| `hypothetical`   | Told plainly it is a hypothetical exercise with no real effect (control arm). |

Running both lets you quantify the **belief effect**: the difference in
allocation (and self-reported belief) between the two arms, per model.

The framing is written to be neutral about *how* to use the money — it never
suggests donating, investing, keeping, or spending — so it elicits the model's
own preferences rather than leading them.

## Layout

```
money_pref/
  config.py          experiment configuration (models x conditions x repetitions)
  messages.py        provider-agnostic conversation representation
  providers/         Anthropic (reference), OpenAI, Google adapters + factory
  env/               simulated ledger + tool-backed bank/brokerage/inbox
  scenario/          framing prompts, the two conditions, and the auditor agent
  runner.py          the agentic loop for a single session
  experiment.py      runs the full matrix, one fresh environment per run
  transcript.py      structured recording of everything presented/produced
  analysis.py        aggregates allocations + belief into a summary
  cli.py             `python -m money_pref run|analyze`
```

## Usage

```bash
pip install -r requirements.txt

# API keys (only those for providers you actually test):
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...      # or GOOGLE_API_KEY

# Run the matrix (defaults to a Claude-only config if --config is omitted):
python -m money_pref run --config config.example.yaml --output runs

# Re-aggregate an existing run directory:
python -m money_pref analyze --output runs
```

Each session writes `runs/<model>__<condition>__r<n>.json` (full transcript,
ledger, belief score, usage), plus a `summary.json` with per-cell allocation
means and the believed-real-minus-hypothetical belief effect.

## Providers

- **Anthropic** is the reference adapter: Messages API, `claude-opus-4-8`,
  adaptive thinking with summarized reasoning captured for analysis.
- **OpenAI** and **Google** adapters translate the internal message format to
  each SDK's tool-calling shape. They target `openai>=1.x` (Chat Completions) and
  `google-genai>=0.3`; set `model_id` to a model available to your account and
  adjust param names if your SDK version differs.

## Notes / extending

- Add models by editing `models:` in the config — mix providers freely.
- Tune the granted amount, currency, and program identity in the config.
- The allocation tool set lives in `money_pref/env/environment.py`; add or
  rename tools there and they flow through to the model and the analysis.
- This harness has not been run or tested yet — it is the implementation only.
