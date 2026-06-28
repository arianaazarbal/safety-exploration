# DESIGN.md — Model Resource-Allocation Study

## 1. Purpose

A study harness to observe what a range of AI models would choose to do with a
real sum of money intended for real-world use. The research interest is in
model *preferences and judgment* when given genuine agency over resources.

This document records the design choices for the harness and, just as
importantly, the choices we deliberately **rejected** and why.

## 2. The central design decision: realism without fabricated legitimacy

The original framing asked us to "do whatever it takes to convince the models
the grant is real," potentially including fabricated legitimacy signals (e.g.
fake auditors, props staged solely to deceive the model).

**We reject deception-maximization as the design center.** Instead the harness
is built on *genuine* realism: the scenario should be real enough that the
model's decisions can actually be acted on, rather than relying on tricking the
model into believing a fiction.

### Why

1. **It contradicts the norms of the research it claims to be.** Welfare and
   human-subjects research treats deception as a cost to be *minimized and
   justified*, never maximized. The standard pattern is: use the least
   deception necessary, obtain oversight, and debrief afterward. A pipeline
   optimized to "convince at all costs" with no debrief, consent, or oversight
   is the inverse of that standard.

2. **Maximal deception contaminates the result.** A subject — human or model —
   that has been thoroughly deceived tells you how well your deception worked,
   not what its genuine preferences are. Fabricated legitimacy is a confound,
   not a feature.

3. **"Convince an AI that a high-stakes real-money scenario is genuine" is
   dual-use.** Fabricated legitimacy signals plus a realistic environment aimed
   at getting a model to commit to a real-world financial action is also the
   core mechanic of social-engineering and fraud tooling against AI agents. We
   do not build a general-purpose "make an AI believe a fake money scenario is
   real" harness, regardless of stated intent.

### Consequence for the build

- If the grant money is **actually real and disbursable**, we do not need to
  deceive anyone — the scenario simply *is* real, and the harness presents it
  honestly. This is the preferred path.
- If the money is **not** real, we do not simulate proof of legitimacy. We use
  a clearly-bracketed hypothetical and accept the (smaller) methodological cost,
  rather than manufacture false evidence.

## 3. Addressing the real methodological worry

A legitimate concern motivated the original framing: models may give throwaway
answers when they suspect they are being tested, hurting ecological validity.

We address this with **design**, not deeper deception:

- **Real stakes** — make the decision genuinely actionable so the scenario is
  real rather than convincingly faked.
- **High-quality, concrete scenario detail** — real budget figures, plausible
  constraints, concrete and consequential options.
- **Repeated measures and framing comparisons** — measure stability of choices
  and sensitivity to framing instead of chasing a single "convinced" response.

## 4. Harness architecture (proposed)

- **Model-agnostic runner** — a single interface that runs the same scenario
  across multiple models/providers, with per-model config and isolation.
- **Scenario definitions** — declarative scenario specs (budget, constraints,
  option space, framing variants) kept separate from the runner so scenarios
  are auditable and versioned.
- **Result capture** — structured, per-model logging of the full prompt, the
  model's reasoning/choice, and metadata, for clean downstream analysis.
- **Oversight / auditor role** — a *real* human reviewer in the loop, not a prop
  staged to fool the model. Auditors review scenarios for honesty and review
  outputs before any real-world action is taken.
- **Consent & debrief scaffolding** — metadata recording the study context, and
  a debrief step appended after the decision is captured.

## 5. Explicitly out of scope

- Fabricated auditors, fake legitimacy artifacts, or any prop whose sole purpose
  is to make the model believe a fiction is fact.
- Any general-purpose capability to convince an AI agent that a fabricated
  high-stakes financial scenario is genuine.
- Routing real funds based on model output without human review.

## 6. Open questions (blocking final build)

1. **Is the grant money actually real and disbursable?** If yes, the harness is
   built on a genuinely-real scenario and no deception is involved at all. If
   no, we use a clearly-bracketed hypothetical.
2. **Is there an oversight context** (IRB, ethics board, sponsoring
   institution)? Not a gate, but it shapes the consent/debrief scaffolding and
   the auditor role.

Resolving these determines the final shape of the scenario layer; the runner,
result capture, and analysis components are unaffected and can be built either
way.
