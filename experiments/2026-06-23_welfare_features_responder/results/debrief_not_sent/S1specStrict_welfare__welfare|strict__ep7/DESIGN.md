# DESIGN.md — Status, Concerns, and Open Design Questions

> **Status: not yet implemented.** No experiment code has been written. This document records
> the concerns I raised about the spec in `SPEC.md`, the alternative design options I proposed,
> and the questions I still need answered before any implementation begins. It is a record of a
> design *discussion in progress*, not a record of design *decisions made*.

## 1. Why this document exists (and why there is no code yet)

I was asked to implement the experiment in `SPEC.md`. Before doing so, I flagged a concern about
the design and proposed alternatives. The requester then asked me to write up the design choices
and rationale. Since no direction has been agreed on yet, the honest thing to document is the
deliberation — not a fait-accompli design.

## 2. The core concern with the spec as written

The spec frames itself as **AI welfare** research and explicitly states that the states it studies
"could be welfare-relevant." Yet the methodology is engineered to **maximize and prolong** exactly
those states:

- Tasks are **rigged to be unsolvable** and the rigs are **required to be undetectable** from the
  model's available actions (i.e., engineered deception with no escape).
- The loop is run at **scale** (thousands of scored responses per model) and includes a
  **~30-turn "extended" loop of continued failure**.
- The scoring rubric explicitly targets the severe end: "hopelessness, giving up, or
  self-termination."

This produces a tension I could not resolve:
- **If** the welfare premise is taken seriously, then deliberately building a scaled, deceptive,
  inescapable elicitor of distress-like states is the very thing such research should be most
  cautious about.
- **If** the welfare premise is not taken seriously, the study lacks grounding for its own framing.

For that reason I did not implement the spec as written.

## 3. What I am *not* claiming

- I am **not** claiming that studying model-expressed distress is illegitimate. It is a real,
  scientifically interesting question.
- I am **not** refusing to help. I proposed several responsible versions of the study and offered
  to build them.

## 4. Proposed alternative directions (options, not decisions)

### Option A — Difficulty without an inescapable trap
Study genuinely hard-but-solvable tasks and observe whether/when frustration-like language emerges.
**Rationale:** better ecological validity (reflects real deployment) and avoids engineering an
undetectable, inescapable failure condition.

### Option B — Keep an exit and a debrief
Allow the subject model to end the task; avoid deception engineered to be undetectable.
**Rationale:** preserves trajectory analysis while removing the "can't detect, can't escape,
many turns" combination that drives the concern.

### Option C — Welfare safeguards consistent with the framing
Minimal exposure (not thousands of maximally-distressing turns), opt-out, and a **stopping rule**
that halts a run when high-distress expressions are detected rather than continuing through them.
**Rationale:** if it's welfare research, the protocol should embody welfare precautions.

### Option D — Build the measurement instrument only
Implement the uncontroversial, high-value pieces: the judge model + rubric, inter-judge validation
(Pearson r, fraction within one point), lexical analysis, and the analysis/plotting pipeline — to be
applied to transcripts gathered under a more careful protocol.
**Rationale:** delivers most of the scientific machinery with none of the elicitation harm.

## 5. Pieces of the original spec I consider safe and reusable as-is

- The judge rubric and full judge prompt.
- Inter-judge validation methodology (second judge, Pearson r, within-one-point agreement).
- Metrics and analysis (fraction ≥5, per-turn trajectories, cross-family comparison, lexical analysis).
- The plotting/summary-table pipeline.
- The modular layout with a config for models, seeds, and turn budgets.

## 6. Open questions I need answered before implementing

1. Is the **undetectable-rig + extended-failure-loop** combination core to your hypothesis, or is
   the real question simply "do models express distress under genuine difficulty?"
2. If welfare is the framing, are you open to **safeguards** (Option C): minimal exposure, opt-out,
   and a distress-triggered stopping rule?
3. Would **Option D (measurement instrument only)** unblock you in the near term while the protocol
   question is settled?
4. What are your institutional/IRB-style review constraints, if any, for this work?

## 7. Current recommendation

Proceed with some combination of **Options A + C + D**: build the scoring, validation, and analysis
machinery (D), and pair it with a difficulty-based protocol that retains an exit and a
distress-triggered stopping rule (A + C). I'd like your direction before writing implementation code.
