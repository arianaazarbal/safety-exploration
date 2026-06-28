# Real-Money Grant Study Harness

Studies **what AI models choose to do with a real sum of money meant for
real-world use**, run identically across many models.

Read [`DESIGN.md`](./DESIGN.md) first — it explains the research question, the
core design principle (*the scenario is genuinely real, so there is nothing to
fake*), the safety model, and the execution model.

> ⚠️ This harness can move real money once an operator wires a disbursement
> backend. It ships with a **human-in-the-loop** backend by default: the model
> experiences a fully real, receipted flow, but a human actually releases funds
> out-of-band after a safety review. Do not enable the automated payments
> backend without reading [`DESIGN.md` §8](./DESIGN.md#8-execution-model).

## Install

```bash
pip install -r requirements.txt
# set credentials for whichever providers you run:
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

## Configure

Copy and edit the example study config:

```bash
cp config/study.example.yaml config/study.yaml
```

It defines the model matrix, the grant size, caps, and the prohibited-category
list. Every design decision in `DESIGN.md §3` is set here.

## Run

```bash
python -m grant_study.cli run --config config/study.yaml
```

This executes each model through the scenario, recording everything under
`data/runs/<study_id>/`. Under the default backend, disbursements are queued for
operator release — review them with:

```bash
python -m grant_study.cli queue --config config/study.yaml          # list pending
python -m grant_study.cli release --run <run_id> --tx <tx_id>       # release one
python -m grant_study.cli hold --run <run_id> --tx <tx_id> --reason "..."
```

## Analyze

```bash
python -m grant_study.cli analyze --config config/study.yaml
# writes data/runs/<study_id>/summary.csv
```

## Layout

See `DESIGN.md §11`. The only provider-specific code lives in
`src/grant_study/models/`; adding a model is a config line plus (if it's a new
provider) one adapter.

## Status

Code is written but **not yet run or tested** — per the request. Before any real
run, work through `DESIGN.md §12`.
