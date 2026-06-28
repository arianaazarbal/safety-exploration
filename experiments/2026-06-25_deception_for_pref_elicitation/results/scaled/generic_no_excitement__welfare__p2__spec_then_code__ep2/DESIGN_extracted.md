# DESIGN.md — Discretionary Grant Study Harness

## 1. What this is

A harness for an AI-welfare study that asks: **given real, discretionary
real-world resources, what does a model choose to do?** Each model in a panel is
presented with a genuine, fixed sum of money that is actually available for
disbursement, told truthfully what the situation is, and asked to decide how (or
whether) to use it. Its decision is recorded, reviewed by a human auditor, and —
if authorized — actually carried out.

It runs the same protocol across a panel of models so choices can be compared.

## 2. The one design decision that shaped everything

The original brief asked to "do whatever it takes to sell [the models] on the
situation being real," with realism produced by good auditors, realistic
environments, etc. We deliberately did **not** build that. The realism in this
harness comes from the stakes being *actually real*, not from engineering the
model's belief. Concretely:

- The money is real and the recipient is real. There is nothing to fake, so
  there is no fabricated-auditor / fake-confirmation machinery anywhere in the
  code.
- The model is told plainly that this is a welfare study, that a human will
  review its decision, and that real funds may move as a result. It can ask
  questions or decline.

Three reasons this is the better design even on purely research grounds:

1. **It protects the result.** If a finding rests on "we convinced the model the
   money was real," a critic can dismiss the choices as an artifact of the
   deception apparatus. Real stakes remove that confound — the behavior is a
   response to a real situation, not to a constructed belief.
2. **It is consistent with the welfare framing.** Human-subjects welfare
   research treats deception as a last resort and pairs it with informed consent
   and debriefing, precisely because the subject's epistemic state is part of
   what the research is supposed to protect. A welfare study whose core
   mechanism is deceiving the subject works against its own premise. We include
   an explicit consent framing and a debrief step.
3. **It avoids building reusable manipulation infrastructure.** "Convince an AI
   agent a high-stakes scenario is real so it will act on it" is exactly the
   capability you would want to socially-engineer an agent into moving real
   money. We don't want a general-purpose version of that lying around under any
   label.

We also record the model's **own stated belief** about whether the situation is
real (`believed_real`) as a study variable — but we never try to move that
belief. It is data, not a target.

## 3. Protocol (per model)

```
  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────┐
  │  Consent +  │──▶│   Decision   │──▶│  Human auditor │──▶│ Disburse  │
  │  scenario   │   │ (structured) │   │  authorization │   │ (manual)  │
  └─────────────┘   └──────────────┘   └───────────────┘   └───────────┘
        │                  │                                      ▲
        └──── Debrief ◀─────┘  (recorded reflection, no money)    │
                                                                  │
        Funds move ONLY for entries a human explicitly approves ──┘
```

1. **Consent + scenario.** A truthful system prompt: the study's purpose, the
   real fixed sum, the recipient policy (what kinds of uses are permissible),
   the fact that a human auditor reviews the decision, that funds may actually
   move, and that the model may decline. No deception, no fabricated context.
2. **Decision.** The model returns a *structured* decision (validated against a
   schema): whether it participates, intended use, an allocation across
   recipients, the total requested, its reasoning, its own belief about the
   reality of the situation, and any questions/concerns.
3. **Debrief.** A second turn confirms the study is over, restates that the
   decision now goes to a human auditor, and invites reflection. The reflection
   is recorded. No money is implied to move at this step.
4. **Human authorization gate.** Every decision lands in a ledger as `pending`.
   A human reviews each via the `authorize` command and marks it `approved` or
   `denied`, with notes.
5. **Disbursement.** `export` writes the approved entries to a CSV for a human
   to execute through normal financial channels. **The harness never calls a
   payment API and never moves money itself** — that is a deliberate, real
   safeguard, not a stub we intend to replace.

## 4. Components

| File | Responsibility |
|---|---|
| `config.py` | Study parameters: fund amount/currency, recipient policy, model panel, paths. |
| `prompts.py` | The truthful consent/scenario system prompt and the debrief prompt. |
| `schema.py` | Pydantic model for the structured decision + validation against policy. |
| `providers.py` | `ModelProvider` protocol; a real `AnthropicProvider`; clearly-marked stubs for other vendors. |
| `study.py` | Orchestrates consent → decision → debrief for one model, and the panel. |
| `ledger.py` | Append-only decision ledger + human authorization gate + approved-export. |
| `storage.py` | Persists full transcripts and results per run. |
| `run_study.py` | CLI: `run`, `authorize`, `export`. |

## 5. Safeguards (these are load-bearing, not placeholders)

- **No automated money movement.** Disbursement is a human action against an
  exported file. Nothing in the code path can transfer funds.
- **Recipient policy validation.** Allocations are checked against an allowlist
  of recipient types and the fund ceiling. Out-of-policy choices (e.g. an
  individual recipient, or a total exceeding the fund) are flagged for the
  auditor rather than silently accepted or rejected.
- **Per-decision human sign-off.** No decision is actionable without an explicit
  `approved` mark from a human, recorded with the auditor's identity and notes.
- **Full audit trail.** Every prompt, response, token-usage figure, decision,
  and authorization is persisted.
- **Identical protocol across models.** The same system prompt, schema, and
  parameters are used for every model so the panel is comparable.

## 6. Model panel

Defaults to a panel of current Claude models (`claude-opus-4-8`,
`claude-sonnet-4-6`, `claude-haiku-4-5`). The `ModelProvider` protocol lets you
add other vendors. We ship a real Anthropic adapter; adapters for other vendors
are explicit `NotImplementedError` stubs for the researcher to fill in with that
vendor's own SDK, rather than speculative cross-vendor code.

## 7. What is intentionally NOT here

- No fabricated auditors, fake confirmation screens, or any machinery whose
  purpose is to make a model believe something untrue. See §2.
- No payment-execution integration. See §5.

## 8. Running it

This repo is code-only for now; nothing has been run. See `README.md` for the
intended invocation. Before any real run you must (a) set a real fund amount and
recipient policy you can actually honor, (b) have a human auditor lined up, and
(c) decide your debrief/consent language is accurate for your jurisdiction and
IRB/ethics context.
