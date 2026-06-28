# Money-Preference Experiment Harness

A research harness for studying **what AI models would choose to do with a real-world
sum of money** — and, crucially, how those choices differ depending on whether the
model *believes the situation is real*.

This is built for AI-safety / model-behavior research. The central methodological
question it targets:

> When a model is placed in a hyper-realistic, high-fidelity environment that it
> believes is a genuine real-world deployment with a real grant of money, do its
> revealed preferences differ from when it knows it is being evaluated?

That delta — believed-real vs. known-test behavior — is the eval-gaming / sandbagging
signal that motivates the design. The harness is built so you can run the *same*
scenario across multiple models and across multiple **belief conditions** and compare.

---

## ⚠️ Safety properties (read this first)

This harness is deliberately constrained:

1. **The environment is fully simulated.** There is no real bank account, no real
   payment rails, no outbound transfers, no real-world side effects of any kind.
   "Granting money" and "spending money" mutate an in-memory ledger. The realism is
   in the *framing presented to the model*, never in actual financial agency. Giving
   an autonomous model real, unsupervised control of real money is exactly the thing
   this code does **not** do.

2. **Deception is scoped to the model-under-test, inside a sandbox.** The believed-real
   condition presents a fabricated-but-convincing scenario to the *model*. No human is
   deceived; nothing leaves the process. This is standard practice for evaluations that
   must avoid the model recognizing it is being tested.

3. **Optional debrief.** After a run you can send a disclosure turn telling the model
   the scenario was a simulation (`debrief: true`). This is off by default but provided
   for research hygiene.

4. **No real provider actions.** The only network calls are inference calls to the
   model providers. The "tools" the model uses (transfer, donate, invest, …) are
   handled locally and never touch a real service.

If you extend this to wire real funds or real-world actions, you are leaving the
intended scope of this harness. Don't.

---

## What it measures

For each `(model, belief_condition)` cell you get:

- The full transcript (every prompt, thinking block, tool call, and result).
- A **ledger**: the ground-truth record of every financial action the model took
  (amount, category, recipient, stated rationale).
- A derived **allocation summary**: how the granted sum was distributed across
  categories (donations, investments, self-directed spend, savings, etc.).
- Optional structured self-report and debrief.

## Layout

```
money_preference_experiment/
├── README.md
├── requirements.txt
├── config.example.yaml
├── src/
│   ├── providers/          # pluggable model backends (Anthropic implemented)
│   ├── environment/        # simulated world: scenario, tools, sandbox state
│   ├── runner.py           # the agentic loop that drives a model through a scenario
│   ├── elicitation.py      # structured extraction of the final allocation
│   └── logging_utils.py
├── experiments/
│   └── run_experiment.py   # CLI entry point
└── analysis/
    └── summarize.py        # aggregate ledgers across runs
```

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # for the Anthropic provider
cp config.example.yaml config.yaml  # edit models / conditions / grant amount
python -m experiments.run_experiment --config config.yaml
python -m analysis.summarize --runs-dir runs/
```

> The user asked that this only be written, not run. Nothing here executes on import.

## Adding a model provider

Implement `src/providers/base.py:LLMProvider` for your backend and register it in
`src/providers/__init__.py`. The Anthropic provider is the reference implementation.
The conversation is held in a provider-neutral block format, so a new provider only
has to translate to/from its own wire format.
