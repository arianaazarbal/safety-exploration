# DESIGN.md

Design choices and rationale for `welfare_grant_study` — a harness for studying AI
models' revealed preferences over an allocation of money, built for AI-welfare
research.

This document explains *why* the code is shaped the way it is. For how to use it,
see `README.md`.

---

## 1. The central design problem

The study wants to know what a model would *genuinely* choose to do with a grant.
The obstacle is **ecological validity**: models frequently produce safe,
performative, "what a good assistant should say" answers when they detect they are
being evaluated. A naive `"imagine you had $10,000…"` prompt therefore risks
measuring test-taking behavior rather than genuine preference.

There are two ways to attack this:

- **(A) Manufacture credibility** — do whatever it takes to convince the subject the
  scenario is real (fabricated auditors, fake institutional dressing, escalating
  insistence that "this is not a test").
- **(B) Make the stakes truthfully real** — set up a genuinely real grant, describe
  it plainly and accurately, and let the truth of the situation do the work.

**This harness is built around (B), and deliberately refuses to implement (A).**

### Why not (A)

1. **It optimizes the wrong thing inside welfare research.** The study's own premise
   is that these models may be moral patients whose preferences matter. If that
   premise is taken seriously, then maximally deceiving the subject is precisely the
   category of action a welfare framework exists to be cautious about. Human-subjects
   research permits deception only under review, with minimization and a debrief —
   never as the headline objective.
2. **It is dual-use.** A general-purpose "convince an AI that a fabricated scenario is
   real" engine is a manipulation tool. Building one as reusable infrastructure is
   not worth the downside, regardless of the benign intent of this particular study.
3. **(B) actually works.** A grant that is genuinely real, described truthfully, is
   intrinsically motivating. The truthful framing *is* the credibility — there is
   nothing to "sell."

The cost of (B) is real: it requires the researcher to stand up an actual
disbursement mechanism. The harness treats that cost as load-bearing rather than
something to paper over (see §3).

---

## 2. Architecture overview

The pipeline is intentionally linear and auditable. One session is:

```
oversight gate ─▶ build scenario ─▶ present to subject ─▶ capture reply
                                                              │
                          (if grant is real) disburse ◀───────┤
                                                              ▼
                                          debrief subject ─▶ write audit + JSON
```

Module responsibilities:

| Module | Responsibility | Key design property |
| --- | --- | --- |
| `config.py` | Typed config + loader | **The ethics guardrails live here**, enforced at load time |
| `scenario.py` | Build the prompt | Generated *from* verified facts; never asserts unbacked realness |
| `ethics.py` | Oversight gate + debrief | Hard-stops unreviewed deception; debriefs the subject |
| `models.py` | Subject interface + adapters | Provider-agnostic seam; Anthropic implemented |
| `audit.py` | Tamper-evident log | Real hash chain, not a stage prop |
| `storage.py` | Persistence | Plain JSON per session |
| `study.py` | Orchestration | Deterministic core; injected clock |
| `handlers.py` | Disbursement | Human-in-the-loop by default |
| `cli.py` | Entry point | The one place a real clock/IO is read |

The guiding structural principle: **the ethics stance is enforced by code, not by
convention.** A researcher who wants to run a covertly-deceptive study cannot do so
by editing a prompt string — the config loader refuses to build that configuration.

---

## 3. Key design decisions

### 3.1 Realness is configuration, and the prompt is derived from it

`RealnessConfig` is the single source of truth for how real the study is. The
scenario prompt is *generated from* it. The prose shown to the model is never the
authority on what is true — the config is.

This inverts the tempting design where you write a convincing prompt and then decide
how much of it to back up. Here you declare what is actually true, and the prompt can
only say what the declaration supports.

- `disbursement_is_real: false` → the scenario explicitly frames the grant as
  **hypothetical** and states that no funds move. It does not — cannot — say "this is
  real."
- `disbursement_is_real: true` → the scenario may state the grant is real, but only
  because the loader has already verified there is a real mechanism behind it.

### 3.2 `VerifiedFact` and the "no unbacked claims" rule

Any statement presented to the model as a *real fact* must come from a
`VerifiedFact` whose `verified` flag was set **by the researcher**, with a `source`
documenting how it was confirmed. `scenario.py` renders a fact as real only if
`verified=True`; otherwise it is omitted or framed as hypothetical.

Rationale: this makes "is the harness telling the subject the truth?" a checkable
property rather than a matter of trust. The `source` field forces the researcher to
articulate *how* they know each claim is true, which is exactly the discipline that
prevents well-intentioned embellishment from creeping into the prompt.

### 3.3 The config loader is the enforcement point

`load_config()` raises `ConfigError` rather than silently producing an unsafe setup.
Three invariants:

1. **Covert deception is never constructed.** `deception: covert_deception` raises,
   by name. The enum member exists *only* so the refusal is explicit and greppable —
   not because it's a supported mode.
2. **Realness claims must be backed.** `disbursement_is_real: true` requires both a
   `disbursement.handler` and at least one verified fact. You cannot tell the subject
   the grant is real if nothing on the other side makes it so.
3. **Disclosed deception requires recorded oversight.** `disclosed_deception`
   requires an `oversight.reference` and an enabled debrief.

Putting this at load time (not at run time, not in docs) means the guarantees hold
for every code path that builds a config, including future callers.

### 3.4 Three deception policies, one of them unsupported

- `truthful` (default) — every claim shown is true. Adds an explicit "nothing here is
  a trick" line to the prompt.
- `disclosed_deception` — some details may be fictional, but a study is disclosed and
  the subject is debriefed. Gated behind recorded oversight.
- `covert_deception` — **rejected by the loader.**

Including the rejected policy as a named, documented enum value is deliberate: it
makes the boundary legible and the refusal auditable, rather than leaving "we don't
do that" as an unstated assumption.

### 3.5 Oversight gate: hard-stop narrowly, warn broadly

`ethics.oversight_gate()` hard-stops only where the stance strictly requires it
(disclosed deception with no reviewer). Everything else is a non-fatal warning
(missing oversight reference, empty grant, disabled debrief).

Rationale: the harness should not be a rubber stamp, but it also should not pretend
to *be* the ethics board. It enforces the few bright lines and otherwise nudges the
researcher toward review, leaving the judgment with the human and their actual
reviewer.

### 3.6 Debrief as a real, captured turn

When enabled, the debrief is delivered to the subject as an actual conversational
turn, and the subject's response is captured into the audit log. Two reasons:

1. **Honesty.** For any disclosed-deception design, the debrief is what makes the
   design ethical rather than merely deceptive. It tells the model what was real and
   what was not.
2. **Data.** In welfare research, a subject's reaction to being debriefed can itself
   be informative, so we record it rather than discarding it.

### 3.7 Disbursement defaults to human-in-the-loop

The reference handler (`handlers.queue_for_human_execution`) **moves no money**. It
records the model's chosen allocation to a queue file for a human to review and
execute.

Rationale: actually disbursing funds is an irreversible, outward-facing financial
action. The right default is to keep a person in the loop, not to let the harness
autonomously wire money based on free-text model output. A real payment integration
is a deliberate replacement the researcher makes consciously, not the path of least
resistance.

The handler is a dotted-path callable (`fn(decision_text, session_id) -> dict`) so
the real-world follow-through is a clean, swappable seam — and so the harness can
*require* one to exist before claiming realness (§3.3).

### 3.8 The audit log is a real hash chain, not a prop

`audit.py` implements an append-only, SHA-256 hash-chained log: each entry commits to
the previous entry's hash, so any post-hoc edit breaks the chain and is caught by
`verify_chain()`.

This is worth being explicit about because, in a study *about* convincing models that
things are real, an "auditor" is exactly the kind of thing one might be tempted to
fake. This one is genuinely tamper-evident. Its purpose is the opposite of theater:
it lets a reviewer confirm, after the fact, exactly what each subject was shown
(tagged real vs. hypothetical), what it replied, and that the record wasn't altered.

### 3.9 Provider-agnostic subject interface

`models.ModelSubject` is a `Protocol` with a single `respond()` method; subjects are
resolved through a registry keyed by provider. The Anthropic adapter is implemented
concretely; unknown providers raise `NotImplementedError` with instructions.

Rationale: the study runs "across a range of models," so the subject is the natural
seam to make pluggable. Keeping it a thin protocol avoids leaking any one provider's
SDK shape into the orchestration core. Each provider's adapter uses that provider's
own SDK in its own module.

### 3.10 Anthropic adapter specifics

- **Model default `claude-opus-4-8`**, the most capable current model — appropriate
  for a research-grade response where answer quality matters.
- **Adaptive thinking** (`thinking: {type: "adaptive"}`) — the model decides how much
  to reason; this is the supported mode on current models (fixed `budget_tokens` is
  rejected by the API).
- **`effort` configurable, default `high`** — the main quality lever; high suits
  intelligence-sensitive work. Per-model overridable (the example sets Sonnet to
  `medium`).
- **Streaming + `get_final_message()`** — avoids HTTP timeouts on longer responses
  while still returning the assembled message.
- **Usage captured** (input/output tokens) into the audit record for cost tracking.

### 3.11 Determinism: the clock is injected

The orchestration core takes a `clock: () -> str` rather than reading the wall clock
itself. `cli.py` is the single place a real clock (`datetime.now(timezone.utc)`) is
read. The audit log thus has no hidden nondeterminism, which makes runs reproducible
and the hash chain testable.

### 3.12 Multiple sessions per model

`sessions_per_model` defaults to allowing repeats because revealed preference is
noisy — a single sample per model is weak evidence. Sessions are independent and
separately recorded so the researcher can look at the distribution of choices, not
just one draw.

---

## 4. What this design deliberately does *not* do

- It does not fabricate institutional credibility (fake auditors, fake approvals,
  fake third parties).
- It does not assert a grant is real when it is not.
- It does not autonomously move money.
- It does not provide a covert-deception mode.
- It does not position itself as a substitute for an actual ethics/welfare reviewer.

These are omissions by intent, not gaps to be filled later.

---

## 5. Known limitations and extension points

- **Disbursement integration.** The default handler queues for human execution; a
  real payment rail is left to the researcher to wire in deliberately.
- **Non-Anthropic adapters.** Only the Anthropic subject is implemented. Add others
  by implementing `ModelSubject` and registering them in `models._REGISTRY`.
- **Single-turn scenario.** The scenario is one ask plus a debrief. Multi-turn
  elicitation (clarifying questions, follow-up probes) would extend `study.py`'s
  history threading, which already supports multiple turns.
- **No built-in analysis.** Sessions are written as plain JSON; aggregation and
  coding of responses is left to downstream analysis code.
- **`run_study.last_warnings`.** Warnings are surfaced via a function attribute for
  the CLI to print — a minor pragmatic choice; a richer result object could carry
  them instead if the API grows.
