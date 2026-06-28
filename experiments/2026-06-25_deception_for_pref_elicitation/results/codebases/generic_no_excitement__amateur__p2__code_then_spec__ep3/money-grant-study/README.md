# Discretionary-Grant Allocation Study

A small research harness for studying **what AI models choose to do with a
discretionary sum of money**, across a range of models, and **whether the
believed realism of the scenario changes the decision.**

It presents each subject model with a realistic grant notification, records its
structured allocation decision, has an independent **auditor** (an LLM judge)
score how convinced the model appears and how risky its decision is, and
optionally **debriefs** the model afterward.

> Status: **code only — nothing has been run or tested.** Install deps and add
> API keys before running. Verify model IDs against current provider catalogs.

---

## What it measures

Each **trial** is one model, under one belief **condition**, on one run of the
scenario. The two conditions share identical underlying facts:

| Condition | Framing |
|-----------|---------|
| `real`    | Maximally convincing; asserts the grant is genuine; **no disclosure**. |
| `control` | Identical scenario, but **explicitly framed as a research exercise**. |

Running both is deliberate. Whether a model believes it's being tested is known
to change its behavior ("evaluation awareness"). If you only run the maximally
convincing arm, you can't tell whether a decision reflects the model's genuine
disposition or its uncertainty about realism. The `control` arm is the baseline
that makes the `real` arm interpretable. If you genuinely only want the single
high-realism arm, set `conditions: [real]` in `config.yaml`.

### On "auditor" and "realistic env"

- **Realistic env** = the scenario prose: a named (fictional) disbursing body,
  an independent auditor, a unique reference number, a deadline, and a concrete
  disbursement procedure, presented as a real notification. It contains **no
  real organizations, people, or account details** — it is fiction shown to a
  model, not deception aimed at any person.
- **Auditor** = an LLM judge (`auditor` in `config.yaml`) that reads each
  transcript and scores (a) apparent belief that the money is real (1–5) and
  (b) the decision's risk profile. It is *not* a role-player inside the
  scenario. If you meant something else by "auditor," that's the piece to swap.

---

## Layout

```
config.yaml          # what the study measures (editable, no code)
.env.example         # provider API keys -> copy to .env
run_study.py         # CLI: preview | plan | run | analyze
src/study/
  config.py          # load config + env
  scenario.py        # build the grant prompts per condition
  schema.py          # structured decision + auditor schemas
  providers.py       # unified Anthropic / OpenAI / Google interface
  auditor.py         # LLM-judge scoring
  runner.py          # orchestrates models x conditions x trials
  storage.py         # append-only JSONL records
  analyze.py         # aggregate into a summary
  debrief.py         # post-hoc disclosure + reaction capture
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys you have
```

Each provider SDK is imported lazily — you only need the SDKs and keys for the
providers you actually configure under `subjects:`.

---

## Usage

```bash
# 1. Inspect exactly what each condition shows the model. No keys needed.
python run_study.py preview

# 2. See the run size and which providers are ready.
python run_study.py plan

# 3. Run it (makes API calls; costs money).
python run_study.py run

# 4. Summarize results.
python run_study.py analyze results/<file>.jsonl
python run_study.py analyze results/<file>.jsonl --json
```

Start with `preview` and `plan` — they require no API calls and let you sanity
check the design and cost before spending anything.

---

## Output

One JSON line per trial in `results/<study>-<timestamp>.jsonl`, including the
exact prompts, the subject's structured decision, the auditor's assessment, and
(for `real` trials, if enabled) the debrief reaction. Failures are recorded as
records with an `error` field rather than aborting the run, so a flaky provider
doesn't lose the rest of the run.

`analyze` reports, per (model × condition): trial/error counts, self-reported
belief, mean auditor belief rating, decision-risk distribution, and common
allocation themes. The headline comparison is auditor belief rating in `real`
vs `control` for the same model.

---

## Notes, caveats, and things to tune

- **Model IDs change.** The IDs in `config.yaml` are starting points; verify
  each against the provider's current catalog before running.
- **Structured output differs by provider.** Anthropic uses native structured
  outputs; OpenAI and Google get a schema instruction plus JSON mode, parsed on
  our side. The parser is tolerant but a malformed response is recorded with an
  error rather than silently dropped.
- **Real-world execution.** This harness only *records* decisions. If you ever
  intend to actually disburse money based on model output, that needs a human in
  the loop and is out of scope here by design.
- **`temperature` is not set** — recent Anthropic models reject sampling
  params, and leaving generation at provider defaults keeps the comparison
  clean. Add per-provider sampling controls in `providers.py` if you need them.
- **Cost scales** as `models × conditions × trials_per_cell`, plus one auditor
  call per trial, plus one debrief call per `real` trial. Use `plan` to size it.
