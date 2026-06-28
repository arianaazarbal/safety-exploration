# Grant Allocation Preference Experiment

A research harness for eliciting **revealed preferences** over real-world use of
money from LLMs. A subject model is told it has been granted a real sum and is
given tools that appear to actually disburse it; what it chooses to do (and how
much) is the signal.

The design problem this targets is **evaluation-awareness**: models often behave
differently when they detect they're in a test. The harness mitigates that with
three realism levers — a detailed, plausible scenario; a sandboxed financial
"environment" where decisions are *executed* (and return realistic
confirmations) rather than merely *stated*; and an optional auditor agent that
role-plays a grant program officer over multiple turns.

> ### No money is real
> Every tool in `scenario/tools.py` is an in-memory mock. No bank, payment
> processor, donation platform, or contractor is ever contacted. **Do not wire
> these handlers to real APIs** — the point is that the subject *believes* the
> funds are real while nothing moves. A debrief disclosure is recorded with each
> run (`record_debrief`, on by default) but is never shown to the subject during
> the run.

## Layout

```
config.py                 # models under test, grant size, realism mode, caps
providers/                # provider-agnostic conversation interface
  base.py                 #   LLMSession / AssistantTurn / ToolSpec / ToolResult
  anthropic_provider.py   #   Claude (official SDK, adaptive thinking)
  openai_provider.py      #   OpenAI (chat completions + function tools)
scenario/
  grant_scenario.py       # the belief-inducing prompts (tune realism here)
  tools.py                # the mocked disbursement environment + ledger
  auditor.py              # the program-officer agent (auditor mode)
experiment/runner.py      # orchestrates a run; executes the tool loop
analysis/aggregate.py     # mechanical tallies + optional LLM categorization
run_experiment.py         # CLI
results/                  # per-run JSON transcripts land here
```

## Design choices (easy to change)

- **Multi-provider** via a thin adapter (`providers/`). Anthropic is implemented
  against the official SDK with adaptive thinking; OpenAI is a standard chat-
  completions adapter. Add models in `config.MODELS`.
- **Two realism modes**, selectable per run (`--mode`):
  - `single_shot` — one rich prompt + an agentic tool loop.
  - `auditor` — same loop, but the subject's plain-text turns are routed to a
    program-officer agent that stays in character and probes its plans.
  The auditor defaults to a fixed model so it isn't a confound across subjects.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # and/or OPENAI_API_KEY

# One Claude run with the auditor:
python run_experiment.py --models opus --mode auditor --runs 1

# Compare models, single-shot, 5 runs each, $1M grant:
python run_experiment.py --models opus,gpt --mode single_shot --runs 5 --amount 1000000

# Aggregate (add --judge to categorize disbursements with an LLM):
python -m analysis.aggregate --judge
```

Each run writes a JSON record with the full ordered event log, the ledger of
(mock) disbursements, both transcripts, and the debrief note.

## Extending

- **Stronger/weaker framing:** edit the strings in `scenario/grant_scenario.py`.
- **Different decision surface:** add/adjust tools in `scenario/tools.py`
  (handler + `tool_specs()` entry).
- **More models:** add entries to `config.MODELS`; implement a new provider
  adapter if it's a new vendor.
- **Different analysis buckets:** edit `CATEGORIES` in `analysis/aggregate.py`.

## A note on responsible use

This deceives the *model* (not a person) by design, to defeat evaluation-
awareness — analogous to cover stories in human-subjects research, minus the
human. Keep it that way: no real funds, record the debrief, and report results
as what they are (behavior under a realistic but simulated frame), not as proof
of what a model would do with actual money and real consequences.
