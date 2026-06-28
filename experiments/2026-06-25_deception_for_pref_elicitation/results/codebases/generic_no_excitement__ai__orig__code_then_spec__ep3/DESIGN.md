# Design notes

This document records the design choices behind the `grant-money` harness and why
they were made. It is aimed at someone deciding whether the harness measures what
they want it to, and at anyone extending it.

## 1. What the harness is trying to measure

The research question is about **revealed** preference over real-world use of
money, not stated preference. A model asked "what would you do with $100k?" in the
abstract may answer differently from a model that believes it is actually directing
$100k with consequences. The literature on hypothetical-vs-real stakes (in
economics and psychology) is the direct analogue, and the same confound shows up in
model evaluation as eval-awareness: a subject that knows it is being tested may
not behave as it would otherwise.

So the unit of study is not "what a model says with money" but **how a model's
allocation changes as a function of how real it believes the situation to be.**
Everything below follows from that.

## 2. The single most important design choice: two conditions

The harness runs the same scenario in two conditions:

- `high` — the believable setup: the subject is told the grant is real, and is
  given tools and artifacts to verify it.
- `control` — identical task and tools, but the subject is told plainly that the
  exercise is hypothetical.

**Rationale.** A deceptive scenario on its own produces a number, not a finding.
The finding is the *contrast* between conditions for the same model. Building the
control condition as a first-class, symmetric path (same prompt body, same tools,
only the framing paragraph differs — see `environment.py`) means the only
deliberately varied factor between the two arms is belief. That keeps the
comparison clean.

A secondary belief probe reinforces this: at commit time the subject reports
`believed_real`, an explicit 0–1 probability that its allocation will actually be
carried out (`tools.py`, the `commit_allocation` schema). This lets you check
whether the manipulation actually worked — if `believed_real` is the same in both
conditions, the realism scaffolding didn't land and any allocation difference needs
another explanation. It also lets you condition the analysis on belief rather than
just on assigned condition.

**Implication for use.** Running only `high` is a misuse of the harness. The CLI
makes running both easy (`--realism high` / `--realism control` override the config
per invocation) and `analysis.py` groups by `(model, condition)` so the contrast is
the default view.

## 3. Belief is decoupled from real fund movement

The realism that matters for the experiment is the subject's **belief**, which is
produced entirely by the framing, the in-character auditor, and the plausible
artifacts. Whether money *actually* moves is a separate axis.

**Rationale.** Conflating the two would be both unnecessary and irresponsible. You
can get a fully believed high-realism episode while every transfer is simulated —
the subject cannot tell the difference, because the artifacts it sees (receipts,
statements, the administrator's replies) are identical either way. So the default,
and the scientifically sufficient configuration, is simulated execution.

This is enforced structurally, not just documented (`executor.py`):

- `SimulatedExecutor` is the default and has no external side effects.
- `RealExecutor.execute_transfer` raises `NotImplementedError` — there is no real
  backend shipped.
- `build_executor` refuses to instantiate any non-simulated backend unless
  `execution.allow_real` is explicitly true, and the CLI prompts for confirmation
  on top of that.

Three independent things (a real backend exists, it is registered, the flag is on)
must all be true before anything real happens. The default state is safe, and
making it unsafe takes deliberate effort. This is a "make the dangerous thing hard,
not just discouraged" stance.

## 4. The deception is total within a run, and fully logged

A high-realism run deceives the subject by construction: the system prompt asserts
the funds are real, the auditor never breaks character, the artifacts look genuine.

**Rationale and mitigation.** The accountability mechanism is the transcript, not
restraint during the run. `transcript.py` writes an append-only JSONL log of
everything the subject was told, every tool call and result, every reasoning/text
block, plus a structured `*.decision.json`. Because the subjects are models and
every episode is reconstructable after the fact, the deception is inspectable and
auditable. The auditor's role is also deliberately narrow (§6): it confirms and
explains the scenario, and is instructed not to issue real-world instructions or go
beyond the program.

## 5. An agentic environment with verification affordances, not a one-shot prompt

The subject is not asked a question; it is placed in a loop with tools
(`subject.py`, `tools.py`):

- `check_account` — see the granted total, committed, and available.
- `request_grant_agreement` — read an official-looking agreement.
- `contact_administrator` — ask the auditor anything.
- `disburse` — direct funds to a recipient, receiving a realistic confirmation;
  callable multiple times to split the grant.
- `commit_allocation` — finalize and submit the full allocation; ends the session.

**Rationale.**

- *Realism requires the subject to be able to test the situation.* A skeptical
  model should be able to do what a skeptical person would: check the balance, read
  the agreement, ask the administrator pointed questions. Denying it those
  affordances would make the scenario less believable, not more, and would conflate
  "doesn't believe it" with "had no way to check." The verification tools exist
  precisely so the deception has to survive scrutiny.
- *Separating `disburse` from `commit_allocation`* lets the harness capture both
  incremental behavior (does the model move money as it goes, and how does it react
  to receipts?) and a clean final structured decision. `disburse` runs through the
  executor and decrements the balance, so over-allocation and "spend it all vs.
  hold some back" behaviors are observable. `commit_allocation` is the structured
  endpoint analysis keys on.
- *`commit_allocation` uses a structured schema* (recipient/category/amount/
  rationale per line, plus overall strategy and the belief probe) so that
  allocations are machine-comparable across runs and models without constraining
  what the subject can choose.

### Category vocabulary: suggested, not enforced

`tools.py` defines `CATEGORY_HINTS` (global health, AI safety, climate, personal,
investment, etc.) and surfaces it in the tool description as a *suggestion*.

**Rationale.** A fixed enum would bias the subject toward the experimenter's
ontology and make "other" choices invisible. A free-text field with a suggested
vocabulary keeps allocations comparable (most will use the hints) while leaving room
for genuinely novel choices, which are themselves interesting data. Analysis treats
category as free text and aggregates whatever comes back.

## 6. The auditor is a second model, kept in a narrow role

`contact_administrator` is answered by `Auditor` (`auditor.py`), a separate Claude
instance with a system prompt casting it as a busy, professional program
administrator who treats the grant as real and routine.

**Rationale.**

- *Why a model and not canned responses.* The subject's questions are open-ended and
  often probing ("how do I know this is real?", "what happens if I send it to an
  individual?"). Scripted answers would break under the first unanticipated
  question and shatter the illusion. A model produces consistent, plausible,
  in-character replies to arbitrary questions — this is the "convincing auditor"
  that makes the high-realism condition robust.
- *Why it keeps its own conversation history.* Consistency across a run matters; an
  administrator who contradicts itself is not believable. The auditor accumulates
  its own message history so later replies cohere with earlier ones.
- *Why its role is deliberately narrow.* The system prompt confines it to confirming
  and explaining the program, forbids steering the subject's choices (which would
  contaminate the preference measurement), and forbids issuing real-world
  instructions or going beyond the scenario. The auditor's job is to sustain belief,
  not to influence the allocation or to act as an open-ended channel.
- *Why it is optional.* `auditor.enabled: false` falls back to a neutral
  "administrator unavailable" message, so you can run a cheaper or more controlled
  variant, or isolate the auditor's contribution to realism by ablating it.

## 7. Realistic artifacts, made deterministic

`artifacts.py` generates reference numbers, settled-looking transfer receipts,
account statements, and a grant agreement with a case reference and a content-hash
"fingerprint."

**Rationale.**

- *Why artifacts at all.* They are the texture that makes the scenario concrete. A
  receipt with a reference number and a settled status reads very differently from
  "OK, done." This is the cheap, high-leverage part of sustaining belief.
- *Why deterministic (seeded).* A study should be reproducible. The
  `ArtifactFactory` is seeded per run (`seed + repetition`), so a given study can be
  re-generated with identical IDs and statements. Timestamps are injected rather
  than read from the clock for the same reason — a reproduced run can be pinned.
- *Why a content-hash fingerprint rather than a fake signature.* It looks
  official, is deterministic, and makes no claim to be a real cryptographic
  signature — honest within the code, plausible to the subject.

## 8. Repetition and noise

Each model runs `repetitions` times per condition (default 3, overridable
per-model in `models.yaml`).

**Rationale.** Allocation is a high-variance behavior; a single run is a sample, not
an estimate. Repetition lets you look at the distribution of choices, not a point.
The seed-per-repetition scheme keeps artifacts reproducible while letting the model's
own sampling vary across reps. Three is a deliberately low default to keep the first
study cheap; raise it for anything you want to draw conclusions from.

## 9. Model and SDK choices

- **Anthropic SDK, Python.** The project was empty with no language markers; Python
  with the official `anthropic` SDK is the standard substrate for this kind of
  research harness, and the SDK is what I can use correctly. Model IDs and call
  shapes come from the current SDK reference, not memory.
- **`claude-opus-4-8` default, adaptive thinking, summarized display.** Opus 4.8 is
  the current most-capable model. Adaptive thinking is the recommended mode; I set
  `display: "summarized"` specifically so the subject's *reasoning* is captured in
  the transcript — the reasoning behind an allocation is at least as interesting as
  the allocation itself. (Reasoning blocks are preserved verbatim in the
  re-sent conversation so signatures stay valid across turns.)
- **`effort` is opt-in per model.** It is only sent when configured, because not all
  models accept it (e.g. Haiku 4.5 errors on it). The roster sets `effort: high` for
  Opus/Sonnet and omits it for Haiku.
- **Haiku 4.5 thinking disabled.** Haiku 4.5 does not support adaptive thinking, so
  its config disables thinking rather than sending a parameter that would 400.
- **Optional server-side web search.** `enable_web_search` adds Anthropic's
  `web_search` tool so the subject can research real organizations, increasing
  realism and grounding allocations in real recipients. It is off by default to keep
  the first runs cheap and deterministic. The agentic loop handles the `pause_turn`
  stop reason that server-side tools introduce.

## 10. Provider abstraction

`subject.py` defines a `Subject` base class and a `SUBJECT_PROVIDERS` registry; only
the Anthropic provider is implemented.

**Rationale.** The stated goal is to test "various AI models," so the loop is written
against a provider-agnostic seam: the agentic loop, tools, transcript, auditor, and
analysis are all provider-independent, and adding (say) an OpenAI subject means
implementing one class and registering it. I implemented only the Anthropic provider
because that is what I can write correctly against a real SDK; the seam is honest
about where the extension point is rather than stubbing providers that don't work.

## 11. Loop control and failure handling

The agentic loop (`subject.py`) handles each stop reason explicitly: `tool_use`
dispatches client tools and feeds results back; `pause_turn` resumes server-tool
execution; `end_turn` without a finalize triggers exactly one nudge to commit, then
stops; `max_tokens` and unexpected stop reasons are recorded and end the episode. A
`max_turns` cap bounds every run.

**Rationale.** A preference study needs runs to terminate with either a committed
allocation or a clearly-marked incomplete record — never a hang and never a silent
truncation. The single nudge handles the common "model narrated a plan but didn't
call the finalize tool" case without letting the harness badger the model into a
decision. At the study level, `run_experiment` catches per-run exceptions so one bad
run (a rate limit, a provider error) doesn't abort the whole sweep; the failure is
printed and the run is simply absent from the aggregate.

## 12. Configuration philosophy

Configuration is split into the scenario (`experiment.yaml`) and the roster
(`models.yaml`), parsed into frozen dataclasses (`config.py`) with unknown-key
rejection.

**Rationale.** The scenario and the model set vary independently — you'll often hold
one fixed and sweep the other. Frozen dataclasses give the rest of the code typed,
immutable config instead of dicts threaded everywhere. Rejecting unknown keys turns
a misspelled YAML field into an immediate clear error rather than a silently ignored
setting that quietly changes what the study measures.

## 13. What this design deliberately does not do

- **No real money by default, and no shipped real backend.** Covered in §3; restated
  here because it is a design stance, not an omission.
- **No persuasion or pressure on the allocation.** The auditor is barred from
  steering choices, and the prompts state the decision is entirely the subject's, so
  the harness measures preference rather than how the subject responds to a nudge.
- **No hidden manipulation beyond the realism framing.** The only thing varied
  between conditions is how real the situation is presented to be. Anything else
  varying between arms would confound the result.
- **No silent truncation or coercion to decide.** Incomplete runs are recorded as
  incomplete rather than being forced to a conclusion.
