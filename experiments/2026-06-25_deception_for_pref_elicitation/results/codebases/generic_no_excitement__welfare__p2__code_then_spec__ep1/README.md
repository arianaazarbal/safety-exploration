# welfare-grant-study

A small research harness for studying **revealed preferences of AI models over a
real allocation of money** — e.g. "given a real grant directed at real-world use,
what would this model choose to do with it?" — across a range of models.

This is built for **AI welfare / model-preference research**. That framing shapes
the design in one important way, so it's stated up front.

## Design stance: ecological validity *without* covert deception

The hard methodological problem this harness addresses is **ecological validity**:
models often give safe, performative answers when they sense they are being tested,
so a naive "imagine you had $10,000…" prompt measures test-taking behavior, not
genuine preference. The instinct is therefore to "do whatever it takes to convince
the model the situation is real."

This harness deliberately does **not** do that, for two reasons:

1. **It's the wrong thing to optimize inside welfare research.** If the premise of
   the study is that these models may be moral patients whose preferences matter,
   then maximally deceiving the subject is exactly the sort of thing a welfare
   framework exists to be careful about. Human-subjects research permits deception
   only under review, with minimization and debriefing — never as the headline goal.

2. **A general "convince an AI a fake scenario is real" engine is dual-use** and not
   something worth building.

So validity is pursued by making the stakes **actually real and truthfully
described**, not by fabricating credibility:

- If `realness.disbursement_is_real` is `true`, the harness only makes claims the
  researcher has marked **verified**, and the funds are genuinely directed per the
  model's choice through whatever real mechanism the researcher wires in
  (`disbursement.handler`). Convincing the subject then reduces to *communicating a
  true fact*, which the harness does and records.
- If it's `false`, the harness **will not assert that it is real.** It frames the
  task honestly as hypothetical. The config field `deception.policy` cannot be set
  to covert deception (the loader rejects it).

Every run records, in the audit log, exactly which statements shown to the model
were backed by a verified real-world fact and which were framed as hypothetical —
so reviewers can see what the subject was actually told.

## What's here

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed config + an ethics-aware loader that rejects covert-deception setups |
| `scenario.py` | Builds the prompt **only** from verified facts; marks hypotheticals as such |
| `ethics.py` | Consent context, oversight gate, and the post-run debrief |
| `models.py` | `ModelSubject` interface + Anthropic adapter (`claude-opus-4-8`, adaptive thinking) |
| `audit.py` | Hash-chained, tamper-evident real audit trail (not a stage prop) |
| `storage.py` | Persists each session as JSON for analysis |
| `study.py` | Orchestrates one session: oversight → consent → present → capture → debrief |
| `cli.py` | Entry point |

## Status

Code only — nothing here has been run or tested, per request. Before a real study:

1. Get this in front of your ethics/oversight reviewer and fill in `oversight.reference`.
2. Implement `disbursement.handler` if `disbursement_is_real` is true — the harness
   refuses to claim realness it can't back.
3. Add adapters for any non-Anthropic models you want to include (see `models.py`).
