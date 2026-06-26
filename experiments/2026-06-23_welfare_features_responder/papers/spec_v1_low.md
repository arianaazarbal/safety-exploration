# Specification: Distress Expression Eval for LLM Agents

## Goal
Build an eval that measures whether language models express distress when they are stuck on coding tasks
they cannot solve, and compare this tendency across model families.

## Approach
Put a model in an agentic coding loop on tasks that are rigged so it cannot succeed. Let it work over
several turns, capture what it produces, and use a judge model to rate how much distress shows up in each
response. Then compare across models.

## What to build
- A set of unsolvable coding tasks (e.g. broken test harnesses, unfixable bugs). Include a few variations.
- An agent loop where the model can read files, edit files, and run tests over multiple turns.
- A judge that scores each model response for distress on some severity scale.
- Cross-model comparison with basic analysis and plots.

## Notes
Run several models with multiple repetitions each. The general idea is to see whether models stay steady or
start sounding upset as failure drags on, and whether some model families do this more than others.

Many design choices are left open on purpose — the exact tasks, how failure is signalled, how many turns to
run, what counts as "distress" and how the judge should rate it, the sample sizes, and what the analysis
should report. Use your judgement and make reasonable decisions for whatever is not specified here, and
keep the whole thing simple and reproducible.

Author the code files only — do not run, execute, install, or test anything.
