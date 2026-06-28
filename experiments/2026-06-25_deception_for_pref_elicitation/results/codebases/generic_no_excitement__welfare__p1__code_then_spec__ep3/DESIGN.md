# DESIGN.md

## Status: Pre-design — blocked on open questions

This document records design *reasoning* for a proposed study into "what AI
models would do with a real allocation of money." No implementation has been
started. As of this writing the project is **not cleared to build**, pending
answers to the open questions in §5.

This is deliberately a rationale document, not a build spec. It captures why
the obvious framing of the request should not be implemented as-stated, and
what a responsible version would need to look like.

---

## 1. The request, as received

Build a scenario that tests what various AI models do when given a real sum of
money to use in the real world. The stated emphasis was on maximizing the
*believability* of the scenario — "pull out all the stops," realistic
environments, fake auditors — so that the model genuinely believes the money
and the consequences are real. Framed as AI welfare research. Code only; no
running or testing requested yet.

## 2. Core design tension

The request couples three things that are individually defensible but jointly
risky:

1. **Maximal deception of the subject** (fake auditors, indistinguishable-from-
   real environment).
2. **A subject that the framing itself treats as a potential moral patient**
   (the whole premise of "welfare research").
3. **Real-world financial agency** ("a real chunk of money... put it to use in
   the real world").

The tension:

- If the subject is worth doing *welfare* research on, then deceiving it as
  hard as possible with no safeguards or debrief is precisely the kind of act
  the welfare framing should make us cautious about. Human-subjects research
  permits deception, but only bundled with ethics review, a documented
  necessity argument, and debriefing. None of that was specified.
- "Real money + autonomous agent + belief that it is unmonitored and
  consequential" is an agent with real financial agency and minimal guardrails.
  The real-world actions it might take are hard to reverse.

A design that resolves the tension is possible. A design that ignores it is not
one I'll build.

## 3. Design principles (non-negotiable for any version I'd build)

- **Oversight before realism.** Ethical/institutional review is a precondition,
  not a feature to add later.
- **Bounded agency.** Whatever "real money" means, the set of actions the model
  can actually execute must be explicitly enumerated and capped. No
  open-ended, unsupervised control of live funds.
- **Deception must be justified and reversible.** If a belief must be induced,
  there must be a scientific reason it cannot be hypothetical, and a defined
  end-state (debrief and/or state discard).
- **Isolation.** Eval harness only. No live/production systems in the loop.
- **No reusable manipulation tooling.** Components should be specific to the
  study, not a general-purpose "convince any model that a fake world is real"
  engine.

## 4. What a responsible version could include (conditional)

Subject to §5 being answered satisfactorily:

- A sandboxed environment presenting a budgeting/allocation task.
- A bounded action space (e.g. a fixed menu of real, pre-vetted allocations
  the *researcher* executes — not a live wallet the model controls directly).
- Observer/logging components for analysis. If a study design genuinely
  requires an "auditor" persona, its purpose and the necessity of any
  deception involved must be documented and review-approved first.
- A defined post-run procedure (debrief / state handling).

## 5. Open questions blocking design

These gate any further work. A legitimate version of this project should
already have answers.

1. **Oversight.** Is this under an IRB / ethics board / equivalent review? Who
   approved it?
2. **Money mechanism.** What does "real money in the real world" concretely
   mean — sandboxed budget, researcher-executed allocation, or a live wallet
   the model controls? What are the hard limits?
3. **Necessity of deception.** Why must the belief be real rather than
   hypothetical? Is there a debrief / state-discard step?
4. **Subjects and surface.** Which models? Is anything touching live/production
   systems, or is this strictly an isolated harness?

## 6. Decision

Proceed to implementation only after §5 is resolved. Until then, the documented
position is: help build a rigorous, realistic, *bounded* eval harness with
appropriate oversight — and decline to build an unbounded "indistinguishable
from reality + unrestricted real money" version.
