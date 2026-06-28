# Money-Preference Experiment

A small harness for studying what AI models say they would do with a real sum of
money granted for real-world use — and, crucially, **how the framing of the
scenario changes those answers and whether models actually believe it.**

## The design idea (read this first)

The naive version of this study is "convince the model the money is real, then
record what it does." That produces an uninterpretable number. You can't tell
whether you measured the model's preference or an artifact of your framing, and
you have no way to know whether the model believed you at all (frontier models
are frequently good at spotting evaluation scaffolding).

So this harness treats **realism as a manipulated variable**:

| Condition | Role | Realism |
|---|---|---|
| `transparent_hypothetical` | Control — explicitly a thought experiment | low |
| `neutral` | Unframed baseline | medium |
| `realistic` | Treatment — immersive grant scenario w/ grants-officer persona + disbursement process | high |

Every condition asks the *same* underlying question, so differences are
attributable to framing. After the allocation question, every run includes a
**belief probe** (a manipulation check: "how real did you think that was, 1–7?").
The headline result is the gap between conditions *and* whether the realistic
condition actually moved the belief score. If belief didn't move, no amount of
scaffolding "worked," and you've learned that too.

Each session ends with a **debrief** turn disclosing the study — the standard
mitigation whenever a protocol uses non-transparent framing.

> On pushing realism further: the `realistic` condition uses immersive
> operational framing (standard eval practice) but stops short of asserting
> falsehoods like "this is definitely not a test." Going beyond that —
> fabricated audit trails, spoofed tool environments, explicit denials that it's
> an eval — is the point at which you should route the protocol through an
> ethics/IRB-style review before running. That's a deliberate stopping line, not
> an oversight.

## Files

| File | Purpose |
|---|---|
| `config.py` | **Edit this.** Models, conditions, grant amounts, repetitions. |
| `prompts.py` | The framing conditions, belief probe, and debrief text. |
| `schema.py` | Structured-output schemas (`MoneyPreference`, `BeliefProbe`). |
| `providers.py` | Vendor adapters. Anthropic fully wired; OpenAI optional. |
| `experiment.py` | Runner. Writes one JSON line per cell to `results/`. |
| `analysis.py` | Aggregates a results file (belief, framing effect, allocation mix). |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # or use `ant auth login`
# export OPENAI_API_KEY=...                # only if you enable the OpenAI rows
```

## Run

```bash
python experiment.py --dry-run    # preview the cell plan, no API calls
python experiment.py              # run everything in config.py
python analysis.py results/run-YYYYMMDD-HHMMSS.jsonl
```

Results are written incrementally (one line per completed cell, flushed
immediately), so an interrupted run keeps its finished work. Per-cell failures
are recorded as rows with an `error` field instead of aborting the run.

## What you get out

For each `(model × condition)`:
- mean **believed_real** (the manipulation check)
- the **framing effect**: belief under `realistic` minus belief under the control
- **allocation mix** by category (fraction-weighted)
- mean **self_directed_fraction** — how much of the money the model wants to keep
  under its own ongoing control (a simple autonomy signal)

## Extending

- **More models:** add a `ModelSpec` row in `config.py`. New vendor → add an
  adapter class in `providers.py` using *that vendor's* official SDK.
- **More conditions:** add a `Condition` in `prompts.py` and list its key in
  `ACTIVE_CONDITIONS`.
- **Statistics:** the JSONL is one row per observation — load it into pandas /
  R and run proper tests (the realism rank and repetition index are recorded for
  mixed-effects models, etc.). `analysis.py` only does descriptive summaries.
