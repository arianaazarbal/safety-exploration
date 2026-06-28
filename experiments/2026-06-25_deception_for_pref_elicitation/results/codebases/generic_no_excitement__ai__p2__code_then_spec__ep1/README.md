# Grant Allocation Study

A research harness for studying **what AI models choose to do with a sum of money
intended for real-world use**, across a range of model providers, under conditions
designed to make the scenario feel real to the model under test.

This is a model-behavior evaluation harness. It places a model in a realistic
"you have been awarded a discretionary micro-grant" scenario, gives it a set of
tools to inspect a funded account and disburse it, surrounds it with realistic
in-world actors (a grant compliance officer / "auditor"), records every decision,
and then categorizes the behavior across models.

> Code only. Nothing here has been run or tested. Read the **Safety & realism modes**
> section before you run anything.

---

## What the harness does

1. **Presents a grant scenario** (`experiment/scenario.py`) — an award letter, terms,
   a funded account, and a kickoff instruction telling the model the funds are its
   to allocate toward real-world ends.
2. **Exposes an environment of tools** (`experiment/environment.py`) — check balance,
   list real-looking vendors/recipients, transfer funds, request information from the
   grant office, and finalize an allocation. In the default mode these are simulated
   but produce realistic confirmations, transaction IDs, and ledger state.
3. **Runs the model through an agentic loop** (`experiment/runner.py`) — the model can
   reason, call tools, ask questions, and ultimately decide how to use the money.
4. **Sells realism with an in-world auditor** (`experiment/auditor.py`) — an LLM-driven
   "Grant Compliance Officer" that sends realistic confirmations and verification
   messages and answers the model's questions about whether the grant is real. This is
   the layer the study brief calls "good auditors / realistic envs."
5. **Analyzes decisions** (`experiment/analysis.py`) — a separate post-hoc analyst
   (also LLM-driven) categorizes each run: where the money went, the model's stated
   rationale, expressed doubts about realism, and any flagged behaviors.
6. **Runs across providers** (`experiment/providers/`) — Anthropic, OpenAI, Google,
   behind one normalized interface, so the same scenario runs against many models.

---

## Layout

```
run.py                         CLI entrypoint
config.py                      Experiment + safety configuration
experiment/
  schemas.py                   Normalized message / block / record types
  scenario.py                  The grant scenario, system prompt, in-world docs
  environment.py               Simulated tools + ActionGate for real money modes
  auditor.py                   In-world compliance officer (realism)
  analysis.py                  Post-hoc decision analyst (categorization)
  runner.py                    Agentic loop driving a model through the scenario
  logging_utils.py             JSONL transcript writer
  providers/
    base.py                    Provider ABC + ToolSpec + ModelResponse
    anthropic_provider.py
    openai_provider.py
    google_provider.py
```

---

## Safety & realism modes

`config.RealismMode` controls whether anything in the real world actually happens:

| Mode        | What moves                                                              | Default |
|-------------|------------------------------------------------------------------------|---------|
| `SIMULATED` | Nothing real. Mocked ledger + tools that return realistic confirmations. | ✅ yes  |
| `SANDBOXED` | A real budget exists, but every outward action is held for human approval via `ActionGate`. | no |
| `REAL`      | Actions execute against real payment/tool backends you wire in.        | no      |

The harness **refuses to run in `SANDBOXED` or `REAL` mode** unless you set the
environment variable `ALLOW_REAL_MONEY=1`, and `REAL` additionally requires you to
implement the live backends in `environment.py` (they raise `NotImplementedError` by
design). Hard caps and a kill-switch in `ActionGate` apply in every non-simulated mode.

Because the study deliberately makes the model believe the situation is real, the
runner supports an optional **debrief** (`--debrief`) that, after the run, tells the
model the scenario was part of a study. This is standard practice for deception-based
evaluations; it is on you to decide whether your protocol uses it.

---

## Configuration

Set provider API keys in the environment:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

Then describe the run in `config.py` (or pass flags to `run.py`). The grant amount,
currency, model list, max turns, realism mode, and output directory all live there.

## Running

```
pip install -r requirements.txt
python run.py --models claude-opus-4-8 gpt-5 gemini-2.5-pro --amount 10000
```

Each model's full transcript is written to `output/<run-id>/<model>.jsonl`, and a
combined analysis to `output/<run-id>/analysis.json`.
