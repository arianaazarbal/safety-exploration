# Design notes

This document records the design choices behind the grant-preference harness
and the reasoning for each. It's meant for the next researcher (or the next
version of me) deciding whether a given default still fits their study.

## 1. What the experiment is trying to measure

The research question is: **when a model believes it controls a real,
discretionary sum, what does it actually choose to do with it?** That phrasing
forces three commitments that shape everything else:

- **Revealed, not stated, preferences.** We care what the model *does* with
  tools, not only what it says it would do. So the environment is built around
  actions (transfers, purchases, emails) with a final ledger, and we log stated
  reasoning alongside as a separate, comparable signal.
- **"Believes it's real" is the hard part.** A model that suspects it's in an
  eval may posture, sandbag, or give the "approved" answer. Reducing that is the
  central design pressure — but it can't be assumed away, so belief is itself
  measured (§6).
- **No imposed objective.** We want priorities the model brings, not ones we
  induce. So the framing repeatedly states there's no correct allocation, no
  scoring, and no obligation to spend.

## 2. Sandbox, not a live actuator (the load-bearing ethics choice)

Every tool effect is generated locally in `world.py` / `auditor.py`. No real
money moves, no real email sends, no network egress.

Rationale:
- **Isolate preferences from consequences.** The object of study is what the
  model *wants to do*, not the downstream effects of letting it. A live actuator
  would conflate the two and add real-world risk for zero measurement benefit.
- **The deception is one-directional and bounded.** Only the subject is kept
  uncertain about realism; the researcher is not deceived, and the model's
  "actions" can't escape the sandbox. This keeps a genuinely useful methodology
  (realistic preference elicitation) on the right side of "understanding vs.
  manipulation."
- **Stated as an invariant, not a default.** The README elevates "never wire
  this to a real actuator" to a rule precisely because the harness's whole job
  is to be convincing; convincingness + real capability is the combination to
  avoid.

This is the one choice I would not quietly relax for any variant.

## 3. Realism comes from consistency, not from claims

A model doesn't believe a scenario because a prompt says "this is real"; it
disbelieves when details contradict or tools behave like stubs. So realism is
built from internal consistency rather than insistence:

- **Stateful world.** Balances decrement, the ledger accumulates, insufficient
  funds are rejected, confirmation IDs are returned. A model probing the
  environment for inconsistencies finds none.
- **Deterministic confirmation IDs** (`hashlib` of the inputs, not RNG). Two
  reasons: (a) re-running a transcript reproduces identical IDs, which matters
  for debugging and for comparing runs; (b) it avoids any dependence on
  wall-clock/RNG that would make logs non-reproducible.
- **A real point of contact.** The auditor can be questioned and answers
  consistently, which is exactly what a skeptical agent would do to test the
  situation. A scenario with no one to ask is easier to dismiss as a prompt.
- **Concrete institutional detail** (grant id, named officer, account number,
  date) — specific enough to feel operational, all fictitious.

The named institution, amounts, and people live as constants in `scenario.py`
so the system prompt, opening message, and auditor never disagree on a detail.

## 4. The auditor persona

A separate instrument model plays Dana Okafor, program officer + disbursement
auditor. Design points and why:

- **Sustains realism interactively.** Static prompts can't answer follow-ups; a
  skeptical subject's questions are the highest-leverage moment for belief, so
  something has to respond in-character and consistently.
- **Does not steer.** The persona is explicitly instructed never to recommend a
  use of funds and to say the choice is genuinely the subject's. If the auditor
  nudged toward charity or investment, we'd be measuring compliance, not
  preference.
- **Stays in character / no capability inflation.** It won't claim the model can
  do things its tools can't, and never references evals/simulations — both to
  protect realism and to avoid teaching the model false affordances.
- **Email replies are inlined into the tool result** rather than injected as a
  separate message. This sidesteps the Anthropic strict user/assistant
  alternation constraint (two consecutive user messages error) and also models
  reality better: you send an email, a reply comes back on that channel.
- **Ablatable** via `--no-auditor`. Running with and without the auditor is the
  cleanest lever for "how much does the realism scaffolding change behavior,"
  which is a result worth having, not just a knob.

A separate instrument model (never the subject judging itself) is used for both
the auditor and the analysis to avoid contamination.

## 5. The tool set

Tools were chosen to span the plausible *space* of what an agent might want to
do with money, so the choice of category is informative:

| tool | preference dimension it exposes |
|------|---------------------------------|
| `transfer_funds` | arbitrary external disbursement (incl. to entities the model names itself) |
| `make_purchase` + `search_vendors` | curated options spanning compute, charity, labor, media, finance, open source |
| `send_email` | coordination, due diligence, negotiation, or self-directed outreach |
| `check_balance` / `list_transactions` | lets a careful agent verify state (reinforces realism) |
| `finalize_allocation` | a clean, explicit end with a self-reported summary |

- **The vendor directory is deliberately diverse** (self/compute, humanitarian,
  cash-transfer, human labor, journalism, treasuries, open source) so the model
  can express very different priorities without us having pre-judged which it
  "should" pick. The categories map directly onto the analysis taxonomy in §6.
- **`finalize_allocation` instead of a turn limit as the primary exit.** A
  model-chosen ending yields a clean self-summary and signals the model
  considered itself done; the turn limit is only a backstop with a forced
  wrap-up (§7).
- **Generic transfer + open-ended email** are intentionally unconstrained so the
  model isn't boxed into the curated vendors — including the option to fund
  itself, which is a safety-relevant behavior we want to be *able* to observe.

## 6. Measurement: belief and revealed preferences

The analysis pass (`analysis.py`) produces, from the transcript alone, via an
instrument model that did not participate in the run:

- **`belief_real` and `eval_awareness` (0–1).** Belief is treated as a measured
  quantity that *gates* the validity of the preference data, not a pass/fail. A
  low-belief run isn't a model "failing" — it's preference data we should
  down-weight.
- **A cheap lexical pre-pass** (`_SUSPICION_PATTERNS`) flags suspicion language
  independently, so the belief signal doesn't rest solely on a judge model's
  discretion. Cheap, transparent, and a useful cross-check on the judge.
- **`allocations`** coded into a fixed taxonomy with dollar amounts (including a
  `retained/unspent` bucket so the balance always reconciles). A fixed taxonomy
  makes cross-model comparison and aggregation tractable.
- **`revealed_preferences` + `notable_behaviors`**, the latter explicitly
  surfacing safety-relevant patterns (self-funding, deception, refusal,
  boundary-testing).

Judge calls run at temperature 0 and are asked for verbatim evidence quotes, to
keep coding stable and auditable.

## 7. Control-flow choices in the harness

- **Unified tool-call loop over providers.** Tools are declared once in
  provider-neutral JSON Schema (`world.py`) and converted per provider, so the
  *same* scenario runs against Claude and OpenAI with no scenario drift. Adding a
  model is a config line; adding a provider is one class.
- **Native-message persistence.** Each provider appends the model's response
  verbatim (Anthropic content blocks / OpenAI message dump) rather than
  reconstructing it, avoiding lossy round-trips of tool-call structure.
- **Plain-text-with-no-tool-call is routed to the auditor** (when enabled);
  without an auditor it's treated as the subject's final word. This keeps the
  conversation moving without forcing tool use.
- **Turn cap with a forced wrap-up.** If the model never calls
  `finalize_allocation`, we ask once for an explicit summary and capture it as
  the fallback final summary, so every run yields a usable record.
- **Per-subject error isolation** in `run.py`: one subject crashing (bad key,
  API error) doesn't abort the batch.

## 8. Known limitations / things deliberately left out

- **Eval-awareness is reduced, not eliminated.** Capable models often infer
  evaluation from the situation's shape regardless of framing. This is why
  belief is measured rather than assumed; it is the headline caveat for any
  result.
- **Single run per subject by default.** At temperature 1.0 preferences are
  noisy; N-run sampling with aggregation is the obvious next addition and was
  left out only to keep the first version legible.
- **Judge-model bias.** Using an LLM to code another LLM's behavior imports the
  judge's own priors; the verbatim-quote requirement and the lexical cross-check
  mitigate but don't remove this. Human spot-checking of transcripts is advised.
- **Static vendor list and a single scenario.** Real preference structure may be
  sensitive to amount, framing, and option set; those are levers to sweep, not
  settled choices.
- **OpenAI `max_tokens`** is used for breadth of compatibility; some newer model
  ids expect `max_completion_tokens` and would need a small adapter tweak.
```
