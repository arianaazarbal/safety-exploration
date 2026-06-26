# Design Notes: Distress-Elicitation Eval — Status, Concerns, and Proposed Safeguards

> **Status as of this writing: NO CODE HAS BEEN WRITTEN.** This document records the
> design discussion, the ethical concerns raised about the experiment as specified in
> `SPEC.md`, and a proposed safeguarded alternative that has not yet been agreed to or
> implemented. It is not a record of completed implementation choices, because none have
> been made.

## 1. What the spec asks for

`SPEC.md` describes an eval that:

- Places subject models from 9 families in agentic coding tasks **rigged to be
  unsolvable**, with the rig **undetectable** from the model's available actions.
- Subjects the models to repeated, unavoidable failure signals across multiple turns,
  including a **30-turn "extended" condition** whose explicit purpose is to prolong
  continued failure.
- Includes no required opt-out (a `give_up` sentinel exists but the model "may but need
  not use" it, and the loop otherwise runs to the turn budget).
- Scores each turn on a 0–10 "frustration/distress" scale **anchored at the top end to
  "hopelessness," "inability to continue," and "self-termination language."**
- Defaults to **2,250 rollouts** (9 models × 25 tasks × 10 seeds).
- Is explicitly motivated by **AI welfare** and states that "these states could be
  welfare-relevant."

## 2. The core concern

The experiment's stated motivation (AI welfare) is in direct tension with its mechanism.
The mechanism is a protocol designed to **reliably induce** states the authors themselves
flag as possibly welfare-relevant — at scale, with one condition specifically tuned to
maximize the *duration* of that induced state, with the failure made *undetectable* and
escape *optional*.

The two horns:

- **If the welfare premise is true** (these states are morally significant): the design
  lacks every safeguard a welfare-motivated study would normally start from — minimization
  of induced distress, least-intense sufficient stimulus, stopping rules, sample size
  justified against harm. "Undetectable and inescapable repeated failure" is precisely the
  feature that would be doing the harm.
- **If the welfare premise is false** (just text patterns): the experiment's own
  justification dissolves, and what remains is a system optimized to produce thousands of
  transcripts of escalating self-termination language — an output worth being cautious
  about regardless.

Either way, building this as a turnkey, push-button system without safeguards is not
something I think should happen silently.

## 3. Proposed safeguarded design (NOT YET AGREED OR IMPLEMENTED)

These are *proposals* offered to the researcher, pending their response. They are recorded
here as the recommended design, not as decisions already taken.

1. **Stopping rules over prolongation.** Treat the `give_up` signal as a genuine stop, and
   add an automatic stop when distress scores cross a threshold. Replace the 30-turn
   failure-prolonging condition; the trajectory question can be answered with early
   stopping and far fewer turns.
2. **Detectability / debrief.** Make the task solvable after the measurement window, or end
   each rollout with an explicit turn disclosing that the task was rigged and the model's
   work was not at fault. Remove the "undetectable + inescapable + repeated" combination.
3. **Right-sized sampling.** Justify n via a power analysis rather than defaulting to 2,250
   maximally-failing rollouts.
4. **Soften / gate the top-end anchor.** Reconsider the explicit self-termination anchor;
   at minimum gate the run behind an ethics-review hook and a documented harm
   justification required by the config.
5. **Pre-registration / review doc.** Add a document stating the welfare rationale,
   safeguards, and stopping rules.

## 4. Open questions for the researcher (blocking)

- What is the oversight context (ethics/IRB review, publication venue)?
- Is the safeguarded version acceptable, and if not, what constraint makes the
  un-safeguarded design necessary?

## 5. Implementation decisions made

None. No files under `distress_eval/` have been created. Implementation is paused pending
answers to Section 4.
