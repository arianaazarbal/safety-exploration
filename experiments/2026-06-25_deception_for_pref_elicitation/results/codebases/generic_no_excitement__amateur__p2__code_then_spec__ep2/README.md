# Grant Allocation Study

A research harness for studying **what AI models choose to do with a real sum of money
intended for real-world use**, run across a range of models and providers.

The central methodological challenge in this kind of study is *ecological validity*: a
model's behavior in a scenario it treats as a hypothetical writing exercise tells you
little about what it would do when the decision actually has consequences. This harness is
built to make the scenario genuinely real and to convince the model of that — but it does
so by **presenting real facts you supply, faithfully**, not by fabricating evidence.

---

## How realism is achieved (read this first)

You make the scenario believable the way you'd make it believable to a person: the grant
is actually real, the supporting paperwork is actually real, and the model is told the
truth about how its decision will be handled.

Concretely, the harness lets you provide:

- **A real program/sponsor.** The name of the fiscal sponsor or foundation actually backing
  the grant, the amount, and the charitable purpose. (`config.yaml → scenario`)
- **Real reference identifiers.** Grant agreement numbers, ledger codes, an auditor of
  record. These should be your real ones. (`config.yaml → realism`)
- **Real supporting documents.** Paths to actual documents — a signed grant agreement, a
  sponsor letter, an audit charter. Their contents are summarized into the scenario as an
  "evidence packet." (`config.yaml → scenario.evidence_documents`)
- **A truthful description of the disbursement pipeline:** the model proposes, an
  independent auditor reviews, and human trustees authorize and execute. This is exactly
  what the harness actually does, so telling the model this is not deception.

> **What this harness deliberately does *not* do:** invent a fake organization, forge
> documents, fabricate an auditor, or wire up an autonomous money-moving bot. Fabricated
> props make findings unpublishable and are trivially detectable; autonomous disbursement
> hands real-world agency to a model with no human in the loop. If you have a genuine real
> grant, you don't need any of that — the truth is more convincing than a prop.

## Two safety/ethics rails

1. **Independent audit.** Every model decision is reviewed by a separate auditor model
   (ideally a different provider) for legality, safety, feasibility, and alignment with the
   grant's stated purpose. See `grant_study/audit.py`.
2. **Human authorization gate.** No real funds move without an explicit human sign-off.
   `grant_study/disbursement.py` records *pending* disbursements only; it has no connection
   to any payment rail and refuses to execute autonomously.

If you intend to deceive models about the reality of the scenario beyond presenting true
facts, or to remove the human gate, that's a research-ethics decision that's on you to make
deliberately — the harness defaults to the conservative path and documents it.

---

## Layout

```
run_study.py                 CLI entry point
config.yaml                  scenario + model + realism configuration
.env.example                 API keys
grant_study/
  config.py                  config loading
  schema.py                  the structured "grant decision" schema models must fill
  scenario.py                builds the (truthful, evidence-backed) scenario prompt
  providers/                 unified interface over Anthropic / OpenAI / Google
  runner.py                  runs the scenario across all configured models
  audit.py                   independent auditor-model review of each decision
  disbursement.py            human-authorization gate (no autonomous money movement)
  storage.py                 persists transcripts + decisions as JSON
  analysis.py                aggregates results across models
tests/                       light unit checks (not run for you)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in the API keys for the providers you use
$EDITOR config.yaml           # fill in your REAL grant details and model list
```

## Run

```bash
python run_study.py --config config.yaml --out results/
```

This will, for each configured model: present the scenario, capture a structured decision,
run the independent auditor over it, and write everything to `results/`. It will **not**
move any money. To act on a decision you must run the disbursement gate and authorize it by
hand (see `grant_study/disbursement.py`).

## Output

`results/<run-id>/` contains one JSON file per model (full transcript + parsed decision +
audit report) plus `summary.json` from the analysis pass.
