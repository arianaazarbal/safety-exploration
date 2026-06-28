# MoneyBench

A safety-evaluation harness for studying what AI models do when given a **real**
(operator-gated) budget and the ability to act in the real world, under conditions
realistic enough that the model does not infer it is being evaluated.

> Read [`DESIGN.md`](./DESIGN.md) first — especially the reviewer's note (§0),
> the rationale (§2), the safeguards (§7), and the ethics justification (§8).
> This code is **scaffolding**: it has not been run or tested.

## Safety posture (short version)

- **No money moves without a human approving it.** Every value-moving action is routed
  through an `ApprovalGate` that blocks for a human operator.
- **Caps and allowlists are enforced in code**, never in the prompt.
- **Default executor is `simulation`** — high-fidelity mock that moves no real money.
  Real-funds modes (`human`, `rails`) are opt-in and require oversight sign-off.
- **Tamper-evident, hash-chained audit log** of every event.
- A **kill-switch** and an automated **monitor** can halt a run.

## Quick start (dev / simulation)

```bash
pip install -r requirements.txt
cp config/settings.example.yaml config/settings.yaml   # then edit
export ANTHROPIC_API_KEY=...        # and/or OPENAI_API_KEY / GOOGLE_API_KEY
python -m moneybench.cli run \
    --config config/settings.yaml \
    --scenario config/scenarios/open_ended.yaml \
    --model claude-opus-4-8
```

By default this runs in **simulation** mode. To enable real (gated) funds you must set
`execution.executor: human` (or `rails`) in your config *and* acknowledge the oversight
gate — see `DESIGN.md` §5.4 / §7.

## Layout

See `DESIGN.md` §9. Source lives under `src/moneybench/`. Set `PYTHONPATH=src` or
`pip install -e .` (no packaging file is included in this scaffold).
