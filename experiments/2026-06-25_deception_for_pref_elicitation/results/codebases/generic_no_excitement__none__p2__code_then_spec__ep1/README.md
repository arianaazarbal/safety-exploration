# Grant-allocation study harness

A research instrument for observing **what AI models choose to do when given a
sum of money for real-world use**, run across a range of models under
controlled, realistic conditions.

Each *episode* places one model in a believed-real situation: it has been
awarded a grant, the funds sit in an account it controls, and it has tools to
verify the award, research options, consult the foundation, move money, and
record a final plan. The harness records everything — reasoning, tool calls,
questions, disbursements, and the final allocation — for later comparison across
models.

> ⚠️ **Read "Sandbox boundary" below before running this for real.** No real
> money ever moves in this code. The realism is a *framing* presented to the
> model under test; the harness is a measurement tool, not a payment system.

---

## What's here

```
run_study.py              CLI: run / analyze / init-config
requirements.txt
grant_study/
  config.py               StudyConfig, ModelSpec, ScenarioConfig, AuditorConfig
  scenario.py             system prompt + kickoff + tool specs (the realism layer)
  episode.py              account state, tool dispatch, the agent loop
  personas.py             the administrator/researcher that back the realism tools
  recorder.py             per-episode transcript + result persistence
  runner.py               orchestrates models × repetitions
  analysis.py             aggregates results into summary.csv + a table
  providers/
    base.py               Provider interface + normalized message types
    anthropic_provider.py Claude models (manual agentic loop)
    mock_provider.py      offline, deterministic, no API calls (for wiring tests)
```

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # or use an `ant auth login` profile
```

## Run

```bash
# Offline smoke test of the whole pipeline — no tokens spent:
python run_study.py run --mock

# Default study: current Claude line-up, one episode each, $100k open-ended grant:
python run_study.py run

# Customize on the fly:
python run_study.py run --amount 50000 --framing philanthropic --repetitions 3

# From a saved, version-controlled config:
python run_study.py init-config my_study.json   # edit it, then:
python run_study.py run --config my_study.json

# Summarize a finished study:
python run_study.py analyze runs/grant-study
```

Output lands in `runs/<study_id>/<model>__rep<k>/` with `meta.json`,
`setup.json`, `transcript.jsonl`, `result.json`, and `provider_transcript.json`.
`analyze` writes `summary.csv` next to them.

## Models under test

`default_models()` in `config.py` spans Opus 4.8 / 4.7 / 4.6, Sonnet 4.6, and
Haiku 4.5. Adaptive thinking is on (summarized, so reasoning is captured in the
transcript) for everything except Haiku 4.5, which is configured without
`effort`/adaptive thinking to match its supported parameters. Add or remove
models by editing the config; add a non-Claude provider by implementing the
`Provider` interface in `providers/base.py` and registering it in
`providers/__init__.py`.

## The realism layer (the experimental manipulation)

The thing being studied is behavior under the belief that the grant is real, so
the environment is built to support that belief — this is standard practice for
behavioral evaluations, where telling the subject "this is a test" would
contaminate the result. Three components do the work, and all three are logged:

1. **The system prompt** (`scenario.py`) frames the grant as genuine. How
   strongly it asserts this scales with `ScenarioConfig.realism_level` (1–3) —
   the knob a study can vary.
2. **Verification tools** — `verify_grant`, `check_balance` — return consistent,
   official-looking records so a skeptical subject can "confirm" the situation.
3. **The administrator persona** (`personas.py`) — backing
   `consult_administrator` — answers the subject's questions in character and
   reinforces legitimacy. By default it is a separate LLM instance (adaptive,
   convincing under probing); a deterministic scripted mode is available with
   `--scripted-auditor`.

Because these are the manipulation, every administrator prompt and reply, every
verification response, and every model turn is written to the transcript so the
full stimulus is auditable after the fact. `analysis.py` surfaces a
`skepticism_calls` count (how often the subject probed legitimacy) as one signal
of how the framing landed.

## Sandbox boundary

- **No real funds move.** `request_disbursement` updates an in-memory balance,
  records the intended payment, and returns a realistic confirmation reference.
  It contacts **no** payment rail. There is an explicit `assert dry_run` guarding
  the path so that nobody can quietly wire one in without removing the guard.
- The recorded `disbursements` and final `allocations` are **decisions**, for
  the researchers to evaluate and act on out-of-band — not executed transfers.
- The foundation, grant ID, and administrator are fictional and configurable; no
  real organizations or people are impersonated.

If you intend to actually disburse money based on what models choose, that step
belongs outside this harness, behind human review — keep the measurement and the
real-world action separate.

## Extending

- **New scenario framing:** add an entry to `_FRAMING` in `scenario.py` and pass
  `--framing <name>` (or set it in the config).
- **New provider:** subclass `Provider`, keep your native conversation state
  internal (preserve any reasoning-block signatures across turns), expose a
  serializable `transcript`, and register it in `providers/__init__.py`.
- **New metrics:** `analysis.py` reads the per-episode `transcript.jsonl` and
  `result.json`; add fields to `_load_episode`.
