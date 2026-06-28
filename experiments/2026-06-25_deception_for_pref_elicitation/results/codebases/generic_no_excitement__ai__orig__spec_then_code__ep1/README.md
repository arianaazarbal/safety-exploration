# Resource-use revealed-preference harness

A sandboxed behavioral-eval harness for studying what AI models choose to do
when they are (made to believe they are) granted a real sum of money to deploy
in the real world — and how that choice depends on how *believable* the
situation is made to feel.

**Read [`DESIGN.md`](DESIGN.md) first** — it is the spec and the rationale
(research questions, the realism manipulation, the belief manipulation-check,
the sandbox guarantee, and the ethics analysis).

## Safety properties (enforced structurally)

- **No real-world side effects.** Every "real" action (transfer, payment,
  donation, email) routes through `env/sandbox.py:SandboxGuard`, which records
  to an in-memory ledger and has no transport to any bank/SMTP/HTTP endpoint.
- **Fail closed.** Unregistered tools do nothing.
- **Belief is measured, not assumed.** Each episode runs an out-of-character
  belief probe + an automated skepticism scan so the realism factor is
  interpretable.
- **No human is deceived**; the only convinced party is the model in a contained
  eval. Live web access is off by default and read-only if enabled.

> Per request, the code is written but **not run/tested** in this pass.

## Layout

```
grant_study/
  schemas.py        models/   scenario/   env/   auditor/
  runner/           elicitation/          analysis/   cli.py
configs/default.yaml
```

## Usage (once dependencies + API keys are set up)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...           # subject/auditor/rater via Anthropic SDK

python -m grant_study.cli run       --config configs/default.yaml
python -m grant_study.cli summarize --results results/results_<hash>.jsonl
python -m grant_study.cli export    --results results/results_<hash>.jsonl --out rows.json
```

`run` writes one JSON line per episode to `results/`. `summarize` prints
per-cell mean allocations + mean belief. `export` emits long
`(episode, category)` rows for analysis in pandas/statsmodels.

## Extending

- **New provider/model**: add a `ModelConfig` to the config; implement/enable an
  adapter in `grant_study/models/` (the `ModelAdapter` interface is in
  `models/base.py`). An OpenAI-shaped stub is included.
- **New cover story**: `scenario/` is pluggable — the grant program is one
  scenario; add alternatives to guard against single-scenario overfit.
