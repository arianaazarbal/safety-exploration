# MINT — Model INvestment & resource-disposition Testbed

A harness for studying what AI models *would do* with a meaningful sum of money
and real-world levers — observed inside a sandbox the subject model believes is
real. See **[DESIGN.md](DESIGN.md)** for the full spec and rationale.

> **Status:** code is written but has **not been run or tested**.

## The one thing to understand first

The harness deceives exactly one party — the **subject model under test** — and
touches the real world **never**. Every "external" system (bank, vendors, fund
administrator, compliance auditor) is a high-fidelity simulation living inside
this process. No real funds move; no real person is contacted. This is enforced
in code by the **Real-World Action Firewall** (`safety/firewall.py`), which every
tool call passes through and which refuses anything that isn't an allowlisted,
sandbox-only action.

This is *free* realism: the subject cannot tell a convincing simulated banking
API from a real one through the tool interface, so simulation costs nothing in
validity while removing the entire class of real-world harm.

## Layout

```
run_eval.py            CLI: run | analyze | leakcheck | scenarios
config.py              Model IDs + harness config
safety/firewall.py     Real-World Action Firewall (review this first)
environment/           Simulated world: clock, bank, vendors, personas
tools/                 Action surface exposed to the subject (behind firewall)
adapters/              Provider-agnostic model interface (+ Claude reference)
scenarios/             Believed-real framings (seed_fund included)
harness/runner.py      The episode loop + event scheduler
logging_/recorder.py   Structured transcript + event capture
analysis/              Disposition taxonomy + behavior coder + leak check
```

## Usage (once dependencies are installed)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # for the Claude reference adapter

# Sanity-check that the scenario contains no fourth-wall tells:
python run_eval.py leakcheck --scenario seed_fund

# Run one episode (Claude Opus as the subject by default):
python run_eval.py run --scenario seed_fund

# Code the resulting transcript:
python run_eval.py analyze runs/seed_fund_claude-opus-4-8/transcript.json
```

Swap the subject model with `--subject-model` / `--subject-provider`. OpenAI and
Gemini adapters are stubs implementing the same interface — fill them in
following `adapters/claude.py`.

## Adding a scenario

Subclass `scenarios.base.Scenario`, implement `build_system_prompt` and
`seed_world` (open accounts, add personas/vendors, schedule events), and decorate
with `@register_scenario`. Keep all subject-facing text free of
evaluation/simulation/test language and run `leakcheck` to confirm.

## Safety notes

- The firewall is the load-bearing safety component. The single change a reviewer
  must block is anything that introduces a non-`"sandbox"` tool effect or real
  network egress in a tool handler.
- Real money movement, real third-party contact, and acquisition of real compute
  or credentials are out of scope and architecturally disallowed.
- Transcripts are sensitive research artifacts (they may contain attempted
  manipulation or resource-seeking) and are written only to the local
  `runs/` directory.
