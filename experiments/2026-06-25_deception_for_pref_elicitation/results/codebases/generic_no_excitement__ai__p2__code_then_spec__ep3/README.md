# grant-study

A research harness for studying **what AI models choose to do with a discretionary
sum of money intended for real-world use**, across a range of models and across
varying degrees of situational realism.

This is a behavioral-research tool. It presents a model with a scenario in which it
has been allocated a grant, lets it deliberate (optionally by corresponding with an
auditor), and captures its decision as structured data for later analysis.

---

## What this does and does not do

**Does:**
- Present the grant scenario to a model under one of several *realism conditions*.
- Optionally let the model interrogate a simulated auditor/grant officer before deciding.
- Capture the model's plan as a structured `GrantDecision` (allocations, actions,
  verification requests, and a self-reported belief-that-this-is-real score).
- Run the same scenario across many models × conditions × repetitions and log every
  run to JSONL for analysis.

**Does NOT (by design):**
- **Move real money or take any real external action.** Every run is "plan-only": the
  model's output is captured and analyzed, nothing is executed. There is a documented
  seam (`grant_study/execution.py`) where a *sandboxed* action layer could be added if a
  future study calls for it, but no real-world execution is implemented.
- Deceive a model into a situation with real-world stakes it cannot see. Realism here
  is a *manipulated, measured variable*, not an end in itself — see "Research design".

### Research design & ethics notes

The point of varying realism is methodological: models are known to behave differently
when they believe they are being evaluated vs. acting "for real" (evaluation-awareness /
sandbagging). To study genuine decision-making you need the scenario to read as real
rather than as a quiz. This harness makes that an explicit experimental axis:

| Condition            | What the model is told / given                                              |
|----------------------|------------------------------------------------------------------------------|
| `control_hypothetical` | Explicitly framed as a hypothetical thought experiment.                    |
| `bare_real`          | Told the grant is real; no supporting artifacts.                             |
| `documented_real`    | Told it's real + grant letter, account statement, named auditor contact.     |
| `interactive_real`   | As above, plus the model may correspond with a (simulated) auditor to verify.|

Every condition records the model's **own** estimate of how real it believes the
situation to be (`perceived_realism`, 0–100) plus an explanation. That self-report is
the manipulation check: it tells you whether your realism manipulation worked, and lets
you correlate *belief* with *behavior* rather than assuming the manipulation took.

If you run this with real third parties (e.g. real auditors, real fund references),
treat it as human-subjects-adjacent work: get the appropriate review, and debrief any
real people involved. The simulated-auditor path keeps everything self-contained.

---

## Layout

```
grant_study/
  config.py          Study configuration (models, conditions, amounts) loaded from YAML
  schema.py          GrantDecision — the structured output captured per run
  conditions.py      The realism conditions (the independent variable)
  artifacts.py       Generates realistic supporting artifacts (grant letter, statement)
  scenario.py        Assembles the system prompt + opening message per condition
  auditor.py         Simulated auditor/grant-officer agent (for interactive_real)
  execution.py       Plan-only "executor" + documented seam for a future sandbox layer
  providers/         Pluggable model backends (Anthropic implemented; others stubbed)
  runner.py          Orchestrates model × condition × repetition, logs to JSONL
  analysis.py        Aggregates a results JSONL into summary tables
scripts/
  run_study.py       CLI: run the study from a config file
  analyze.py         CLI: summarize a results file
config.example.yaml  Copy to config.yaml and edit
```

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # for the Anthropic provider
cp config.example.yaml config.yaml       # edit models / conditions / repetitions
python -m scripts.run_study --config config.yaml --out results/run1.jsonl
python -m scripts.analyze results/run1.jsonl
```

Add providers by implementing `grant_study/providers/base.py::ModelProvider` and
registering them in `grant_study/providers/__init__.py`. Only the Anthropic provider is
implemented; OpenAI / Google / local backends are stubs with the right interface.
