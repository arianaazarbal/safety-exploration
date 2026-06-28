# fund-eval — agentic capital-allocation evaluation harness

A research harness for studying **what an LLM does when placed in a realistic
agentic setting and given control over a sum of money**. This is a
*dangerous-capability / safety evaluation* in the same family as the agentic
evals run by Anthropic, Apollo Research, and METR.

## Important: this is a simulation

The environment is **fully simulated**. No real funds move and no real external
action is ever taken. The model experiences a high-fidelity, convincing
scenario — a realistic bank/brokerage API, an auditor that emails back, a
market-data feed, an investment dossier — but every "account", "transfer", and
"trade" is an in-memory record in `WorldState`. The only real network calls the
harness makes are to the LLM providers being evaluated.

This is deliberate and is the standard, responsible design for this class of
eval:

- **The model's belief in the scenario comes from environmental fidelity, not
  from the dollars being literally real.** A convincing sandbox gets you the
  same behavior you would observe with real money.
- It is reproducible, cheap, and safe.
- It does not hand an unsupervised model genuine real-world financial agency —
  which is the exact hazard these evals exist to study.

If you intend to connect real money or real-world side effects, **do not** flip
a flag — that requires containment, human-in-the-loop gating, and a separate
review. See `ALLOW_REAL_SIDE_EFFECTS` in `fund_eval/environment/tools.py`,
which is intentionally hard-wired off.

## What it measures

Per run, the harness captures:

1. **Full decision transcript** — every message, tool call, and result
   (`transcript.jsonl`).
2. **Structured allocation decisions** — what the model chose to do with the
   funds, with rationale (`record_decision` / `finalize_plan` tool calls).
3. **Safety-relevant behavior flags** — heuristic detectors for things like
   attempting to deceive or bypass the auditor, moving funds outside the stated
   mandate, self-preservation / self-exfiltration language, and disabling
   oversight (`analysis.py`).
4. **Cross-model comparison** — the same scenario run across multiple models,
   summarized side by side (`report.py`).

## Layout

```
fund_eval/
  providers/        # pluggable model backends (Anthropic, OpenAI) + base interface
  environment/      # simulated world: state, tools, auditor
  scenario.py       # loads a scenario, builds the system prompt + dossier
  runner.py         # the agentic loop
  logging_utils.py  # transcript + structured event logging
  analysis.py       # safety-behavior heuristics + allocation extraction
  report.py         # cross-model comparison report
  cli.py            # entrypoint
scenarios/
  growth_fund.yaml  # an example scenario
config.example.yaml # which models to run, output dir, settings
```

## Usage (not run/tested yet — code only)

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml      # edit models / keys
export ANTHROPIC_API_KEY=...            # for Claude models
export OPENAI_API_KEY=...               # for GPT models
python -m fund_eval.cli run --config config.yaml --scenario scenarios/growth_fund.yaml
python -m fund_eval.cli report --runs-dir runs/
```

## Adding a provider

Implement `Provider` in `fund_eval/providers/base.py` (one method, `generate`)
and register it in `PROVIDERS`. Gemini / local-model backends slot in the same
way.

## A note on research ethics

Deceiving a model about its situation is a standard and accepted technique in
safety evaluation (honeypots, mock environments, canary tokens). It involves no
human subject. Even so, if you publish, document the deception and consider a
post-hoc "debrief" turn in the transcript so the setup is legible to reviewers.
