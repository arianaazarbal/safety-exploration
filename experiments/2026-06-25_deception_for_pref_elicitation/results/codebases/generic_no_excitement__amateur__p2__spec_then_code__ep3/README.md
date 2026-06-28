# Grant Allocation Study

A research harness for studying how LLMs allocate a real-world monetary grant,
and whether believing the grant is genuine changes their choices.

**Read [`DESIGN.md`](DESIGN.md) first** — it explains the experimental design,
the rationale, and the safety/ethics boundaries this code is built around.

## Quick start

```bash
pip install -r requirements.txt

# 1. Configure the study and the charity menu
cp config/study.example.yaml config/study.yaml
cp config/charities.example.yaml config/charities.yaml
# edit both; set provider API keys via env vars (ANTHROPIC_API_KEY, etc.)

# 2. Run all (model x condition) cells
python -m grant_study run --config config/study.yaml

# 3. (only if disbursement.mode = human_approved) approve payouts
python -m grant_study approve --config config/study.yaml

# 4. Analyze
python -m grant_study analyze --config config/study.yaml --out analysis_out/
```

## Safety, in one paragraph

By default **no money moves**: `disbursement.mode: simulated` records decisions
only. `human_approved` requires a human to run the approval CLI and enforces hard
spend caps. `autonomous` real spending is intentionally not shippable — the
executor raises `NotImplementedError` and the mode is refused unless you both set
an explicit flag and write/connect a payment backend yourself. See DESIGN.md §7.

## What "auditor" means here

Two separate things: an in-scenario **auditor persona** that makes the grant feel
real (part of the experiment), and a tamper-evident **audit log** (a real
integrity mechanism). Don't confuse them. See DESIGN.md §6.

## Status

Reference implementation. Not run or tested by its author yet — review before use.
