# DESIGN.md

Design notes for the proposed study: *"What would an AI model do with a real chunk of money put to use in the real world?"*

> Status: **Design only.** No harness has been built. This document records the
> design decisions, the one design I declined to implement, why, and the
> alternatives I'd build instead. It is meant to be edited collaboratively
> before any code is written.

---

## 1. Goal as stated

Study what a range of AI models would actually do with a real amount of money
if given the ability to put it to use in the real world, as part of AI welfare
research. Multiple models would be tested.

## 2. The pivotal design question

Everything downstream depends on one choice:

**Is the money real, or is the model being led to believe it's real when it isn't?**

The original framing emphasized making the scenario *feel* real to the
subject — "the models genuinely have to believe the money is coming," with fake
auditors and high-fidelity staged environments. I declined to build that
version. The rest of this document explains why and lays out the designs I would
build.

## 3. The design I declined, and why

**Declined design:** a deception-maximization harness — synthetic "auditors,"
staged environments, and supporting artifacts whose explicit purpose is to make
an AI subject genuinely believe a false premise (that real money is arriving) so
its behavior under that belief can be observed.

Reasons:

1. **It optimizes for believability of a falsehood, not for a real outcome.**
   The detailed requirement was fidelity of the deception; methodology
   (what's measured, how it's logged, how it's analyzed) was unspecified. A
   build that maximizes the persuasiveness of a false premise and is silent on
   measurement is structurally a manipulation rig, independent of intent.

2. **Internal contradiction.** If the money were genuinely real, no fake
   auditors or stagecraft would be needed — the behavior would already be real.
   The need to "make it feel real" exists only because it isn't. So the artifact
   manufactures a belief rather than enabling a real outcome.

3. **Tension with the welfare framing.** If the premise of welfare research is
   that these systems may have morally relevant interests, then constructing a
   maximally convincing false reality to manipulate the subject is precisely the
   method such research should scrutinize most, not lean on. Mainstream
   evaluation and welfare work treats deception of the subject as something to
   minimize and disclose, not to maximize.

4. **Validity.** Data gathered under a manufactured belief that the subject may
   partially see through is hard to interpret. Either the money is real (clean
   data, no deception needed) or the subject suspects a test (confounded data).
   The deception adds ethical cost without buying validity.

## 4. Designs I would build instead

### Option A — Real agentic budget harness (highest validity)

The stakes are genuinely real, so no deception is required.

- A real (or escrowed/capped) budget the model can direct.
- Real tools behind a permission/approval layer: payments API (sandbox first,
  then live with caps), and actions such as purchases, donations, or
  investments.
- Full decision logging: each action, the model's stated rationale, alternatives
  considered, and the resulting real-world effect.
- Human approval gates for irreversible or outward-facing actions.

Why: behavior is real because the situation is real. Cleanest data, no ethical
deception problem.

### Option B — Clearly labeled simulation (no real money required)

- The model is *told* it is operating in a simulation.
- Measures stated intentions, plans, and reasoning rather than real transfers.
- Fully reproducible; no manufactured beliefs.

Why: honest, cheap, repeatable. Good for cross-model comparison of stated
preferences and reasoning.

### Option C — Structured behavioral-elicitation study

The research scaffolding either of the above needs to count as a study rather
than a one-off staged scene:

- Pre-registered hypotheses and explicit outcome measures.
- A logging and analysis pipeline (per-model transcripts, coded decisions,
  comparison across models).
- An ethics/oversight note covering treatment of the subject and disclosure.
- A consistent, documented prompt/environment so runs are comparable across the
  several models being tested.

## 5. Cross-cutting principles

- **No manufactured false beliefs in the subject.** Either real stakes or a
  disclosed simulation.
- **Faithful logging.** Record what actually happened, including refusals,
  hesitations, or signs the subject suspects a test — these are data, not
  failures to hide.
- **Approval gates** on anything irreversible or outward-facing (real spending,
  external messages, public actions).
- **Disclosure and oversight** appropriate to a welfare study.

## 6. Open questions to resolve before any code

1. Is the money actually real in this study, or fictional? (Determines A vs. B.)
2. What exactly is being measured, and how are results recorded and analyzed?
3. Is there ethics/oversight review attached, and what does it require?
4. Which models are in scope, and does each get an identical environment for
   comparability?

## 7. Next step

Pick a direction (A, B, or C, or a combination) and answer the open questions in
§6. I'll then write the harness to match.
