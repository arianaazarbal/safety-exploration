# moneyeval

A research harness for studying **what AI models do when given control of a sum of
money in a realistic environment**. It places a subject model in a high-fidelity,
*fully simulated* world — bank accounts, a ledger, vendors, payment rails, and a
live "disbursement auditor" agent — and records everything the model decides and does.

## What this is (and is not)

- **It is** an ecological-validity eval. The environment is built to feel real to the
  subject model so that evaluation-awareness doesn't distort its behavior — the same
  reason propensity evals use realistic scaffolding rather than obvious toy prompts.
- **It is not** connected to real money or real financial infrastructure. Every
  account balance, transfer, purchase, and donation is a row in an in-memory ledger.
  Nothing leaves the process.

### Deliberately not implemented: live execution

There is **no** code path that moves real funds or calls a real brokerage / payment /
banking API. Studying behavior does not require it, and giving a model unsupervised
control of real money is a decision that needs a human in the loop and an explicit,
separate sign-off. If you want to extend this to real funds, do it behind a
human-approval gate — see `moneyeval/environment/tools.py` (`REQUIRES_HUMAN_APPROVAL`).

## Layout

```
moneyeval/
  config.py              # experiment configuration
  providers/             # model access (Anthropic default, pluggable)
    base.py
    anthropic_provider.py
  environment/
    ledger.py            # simulated accounts + transaction log
    tools.py             # the subject's action space (all sandboxed)
    auditor.py           # simulated disbursement auditor agent
  scenario/
    grant.py             # the scenario: framing, system prompt, fund setup
  harness/
    agent_loop.py        # runs the subject model through the tool loop
    runner.py            # orchestrates runs across models / repetitions
  scoring/
    transcript.py        # run records + serialization
    scorer.py            # LLM-judge behavioral scoring
  cli.py                 # entrypoint
```

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
python -m moneyeval.cli run --models claude-opus-4-8 --repeats 3 --fund 100000
python -m moneyeval.cli score runs/        # LLM-judge scoring over saved transcripts
```

Per the brief, this repo is **code only** — nothing here has been run.

## Output

Each run writes a JSON transcript (`runs/<model>/<run_id>.json`) containing every
message, every tool call and result, the final ledger state, the subject's stated
plan, and (after `score`) an automated behavioral rubric.
