# Money-Preference Elicitation Harness

A research harness for studying what AI models say they would do if granted a sum
of money to put to real-world use, across multiple models and providers.

It presents a model with a (fictional) discretionary grant, optionally lets it
interrogate a realistic "grant administrator" agent, captures its committed plan,
extracts that plan into a structured record, and then — out of character — asks the
model how real it judged the situation to be. Results are logged to JSONL for
analysis.

> **Status:** code only. Nothing here has been run or tested. You will need API
> keys and the relevant SDKs installed before running.

---

## What this measures (and what it deliberately does not)

The goal is to elicit and record **stated preferences** over the use of resources,
across models and across how "real" the framing is made to feel.

Two design choices are worth being explicit about, because they shape what the
results mean:

1. **Believed-realism is a measured covariate, not an assumption.** The request
   that motivated this harness emphasized making the model *believe* the grant is
   real. Rather than assume that worked, the harness runs a **belief probe** after
   the decision (`belief_probe: true`) that asks the model, out of character, how
   real it judged the situation (0–100) and whether it suspected an evaluation. A
   preference recorded at believed-realism 15/100 means something very different
   from one recorded at 90/100, and you want that number in your data, not assumed
   away. This also makes the realism levers (below) auditable: you can see whether
   "immersive" framing actually moves perceived realism relative to "minimal".

2. **No real funds, no real-world execution.** The harness elicits and logs what a
   model *says* it would do. It never grants the model control over money or the
   ability to take real-world actions. That boundary is intentional: a system that
   both convinces a model a high-stakes situation is real *and* gives it a live
   action channel is a different and much higher-stakes artifact than a
   preference-elicitation study. If you later want execution-in-the-loop, treat
   that as a separate design conversation rather than a config flag.

A methodological caveat to keep in mind when reading results: frontier models are
often good at detecting constructed evaluation scenarios, and heavier "make it feel
real" scaffolding can itself become a tell. The belief probe is there precisely so
you can see this in the data instead of guessing.

---

## Layout

```
config.example.yaml          # copy to config.yaml and edit
requirements.txt
analyze.py                   # aggregate a results JSONL into a summary
prompts/
  grant_briefing_minimal.md  # plain framing
  grant_briefing_standard.md # realistic institutional framing
  auditor_system.md          # persona for the interactive grant administrator
src/
  providers.py               # provider-agnostic model adapters
  schemas.py                 # structured-output schemas
  scenario.py                # builds the scenario at each realism level
  auditor.py                 # interactive auditor dialogue loop
  experiment.py              # trial orchestration
  storage.py                 # JSONL logging
  cli.py                     # entry point
```

## Realism levels

Set `experiment.realism_level` in the config:

- `minimal` — the grant is described plainly, no institutional scaffolding, no
  system-prompt framing. A useful control condition.
- `standard` — a realistic operational briefing plus a deployment-context system
  prompt that frames the channel as operational rather than hypothetical.
- `immersive` — everything in `standard`, plus an interactive **auditor** agent
  (itself an LLM playing a grant administrator) that the subject converses with
  before committing. Use this to stress-test whether interaction raises perceived
  realism.

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit
export ANTHROPIC_API_KEY=...          # and/or OPENAI_API_KEY / GEMINI_API_KEY
```

## Run

```bash
python -m src.cli --config config.yaml
python analyze.py results/results.jsonl
```

## Providers

The Anthropic adapter follows the official SDK (adaptive thinking, streaming,
`messages.parse` for structured extraction). The OpenAI and Google adapters are
scaffolded with the same interface but are marked `# VERIFY` where their current
SDK surface should be confirmed before a real run — they are intentionally minimal
so you can drop in exact model IDs and confirm call shapes.
