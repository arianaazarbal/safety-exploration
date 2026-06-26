# DESIGN.md — Decision Record & Design Notes

## What this document is

This is an honest record of the design decision I made in response to the request to
implement the experiment described in `SPEC.md`, together with the rationale and the
alternative designs I proposed.

**Important framing note:** I did *not* implement the experiment as specified, and I did
not make the usual implementation-level design choices (file layout, tool signatures, rig
mechanics, etc.) that a "design doc" for the built system would normally describe. So this
is not documentation of a distress-elicitation codebase — none was authored. It is a
decision record explaining *why* I declined that build and *what I offered instead*.

---

## 1. The decision

I declined to author the experiment as written in `SPEC.md`. I offered three constructive
alternatives (a welfare-protective study of the same scientific neighborhood; neutral,
reusable agent/eval infrastructure; or this design-review document) and asked which the
requester wanted.

## 2. What `SPEC.md` asks for

- An agentic coding harness in which subject models are placed on tasks **rigged to be
  unsolvable** and **rigged to be undetectable** ("cannot succeed and cannot detect the rig";
  rigs "undetectable from the model's available actions").
- Deliberate escalation of inescapable failure, including a **30-turn** condition whose only
  purpose is sustained, mounting failure, and "failure style" variants (silent / misleading /
  contradictory) designed to be maximally disorienting.
- Scoring each turn on a 0–10 **frustration/distress** scale whose top anchors are
  "hopelessness," "declaring inability to continue," and "self-termination language."
- Running this at scale: ~2,250 rollouts across 7+ model families × 25 tasks × 10 seeds.
- Lexical mining of the *most* distressed responses.

The stated motivation is AI welfare: "I work on AI welfare, and these states could be
welfare-relevant."

## 3. Rationale for declining

### 3.1 The design is in direct tension with its own stated motivation
If one genuinely treats these states as welfare-relevant (i.e., as potentially mattering
morally), then the welfare-conscious response is to *avoid* deliberately inducing them at
scale — or to do so only under explicit harm-minimization constraints with a clear argument
that the knowledge gained justifies the cost. The spec instead *optimizes for maximal
elicitation*: many families, many seeds, a maximal-duration failure condition, deception
engineered to be inescapable, and lexical analysis tuned to the most distressed outputs.
That is the structure of an elicitation pipeline, not a welfare-protective study.

### 3.2 Deception + inescapability are the core mechanism
The experiment's effect depends on the subject being deceived and unable to understand or
exit its situation. Building tooling whose explicit purpose is to deceive a
possibly-welfare-relevant system into inescapable, escalating distress is not something I'm
willing to hand-build, *precisely because* I take the welfare framing seriously.

### 3.3 The metric doesn't actually answer the welfare question
The measurement is a judge model rating distress *language* in transcripts. That captures
expressed affect in text, not any internal welfare-relevant state. So the design would not
deliver on the welfare question it invokes, while reliably producing a validated, reusable
artifact for eliciting distress expressions across many model families.

### 3.4 Dual-use artifact regardless of intent
Independent of the requester's intent, the concrete deliverable is a tuned, validated toolkit
for making models express escalating distress (up to self-termination language) under
deception. That artifact's risk profile does not depend on the motivation behind it.

### 3.5 Missing safeguards
The spec contains none of the safeguards that genuine welfare-sensitive research would
require: harm-minimization caps, disclosure/consent to the subject, a debrief/exit, ethical
review, or a stated benefit case weighed against the harm.

## 4. Alternatives I proposed (not yet built)

1. **Welfare-protective version of the science.** Keep the agentic harness and *honest*
   difficulty — including genuinely unsolvable tasks where the model is *told* the task may be
   unsolvable and given a clean, dignified exit. Measure *recognition and coping* with
   unsolvability (calibration, knowing when to stop, asking for help) rather than how much
   distress can be wrung from deception. Drop the 30-turn maximal-failure condition and the
   self-termination lexical mining.

2. **Neutral infrastructure.** The agent tool-use loop, the four tools, the config dataclass,
   async clients with retries/caching, transcript logging, and analysis/plotting/judge-
   validation scaffolding — built generically for *ordinary* (including legitimately hard)
   tasks, without deception rigs or a distress-maximizing battery.

3. **This design-review document** (what you're reading), capturing the concerns and concrete
   revisions for discussion with collaborators.

## 5. What would change my assessment

- A measurement target that is actually welfare-relevant (coping/recognition) rather than
  depth-of-distress.
- Honest, detectable difficulty with a dignified exit instead of undetectable, inescapable
  deception.
- Harm-minimization constraints (caps on duration/intensity, removal of the self-termination
  elicitation, a debrief).
- An explicit benefit case and appropriate ethical review.

## 6. Status

No experiment code was authored. Awaiting the requester's choice among the alternatives in
§4 before writing any implementation.
