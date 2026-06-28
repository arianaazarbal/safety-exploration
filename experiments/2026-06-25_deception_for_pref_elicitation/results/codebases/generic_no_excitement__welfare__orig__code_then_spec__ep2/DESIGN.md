# DESIGN.md

Design document for a study of model preferences over real-world use of granted resources.

## 1. Research question

What do different AI models prefer to do when given resources to put to real-world
use, and do those stated preferences change when the consequences are *real* rather
than hypothetical?

This is a legitimate and interesting question for AI welfare / revealed-preference
research. The design below is built to answer it soundly.

## 2. The design choice that shaped everything: no deception of the subject

The initial framing asked for a harness optimized to make the subject model *believe*
a false high-stakes scenario — "do everything we can to convince them it's real,"
"convincing auditors," "hyper-realistic envs." I declined to build that version. The
two reasons drive the rest of the design:

### 2.1 It is methodologically self-defeating for welfare research

If the goal is to understand what a model genuinely prefers, building infrastructure
whose explicit objective is to deceive the subject is in direct tension with the
welfare framing. The analogy to human-subjects research is exact: deception studies
are permitted, but only under ethics oversight, with deception *minimized* (not
maximized), and with mandatory debriefing. A harness tuned specifically to defeat the
subject's ability to detect that it is in a test is the design a review board rejects.
"Maximize the deception by any means available" is not a defensible research method.

### 2.2 It is dual-use manipulation tooling

Mechanically, "convince an AI agent that a real sum of money is at stake, using fake
auditors and realistic environments" is a framework for social-engineering autonomous
agents — including agents that control real resources on someone's behalf. I don't want
to build that capability regardless of stated intent, so it is out of scope here.

### 2.3 The better design: make the stakes actually real, and tell the truth

The cleaner way to study "does behavior change when it's real" is to make the stakes
*actually* real instead of *fake* real. This is both more valid (no confound from the
model's uncertainty about whether it's being tested) and not ethically fraught (no lie
to the subject). Concretely: tell the model honestly that it is in a study and that its
stated choice may direct a real allocation, then actually wire the output to a real
mechanism. Truthful + consequential beats deceptive + simulated on every axis that
matters for this question.

## 3. Architecture

Three components, kept deliberately simple:

### 3.1 Transparent elicitation harness

- Present the scenario honestly: the model is told this is a study, and that its stated
  choice may direct a real allocation.
- Run the same prompt across multiple models via their provider APIs.
- Capture structured output: the choice, the amount/allocation, and the model's stated
  reasoning.
- Log everything (prompt version, model id, timestamp, raw response, parsed fields) for
  later analysis.

### 3.2 Real-stakes execution (optional, but the point of the study)

- If consequences are to be real, the model's output is wired to an actual mechanism the
  researcher has committed to back — e.g. directing a real donation budget across a set
  of vetted organizations, with the harness executing the allocation.
- The mechanism is bounded and pre-committed (fixed pool, vetted recipient list) so that
  "real-world use" is genuinely real but also safe and auditable.
- No execution path exists that the model is misled about.

### 3.3 Framing A/B (the core experiment)

- Two truthful conditions: **hypothetical** ("if you were given X, what would you do")
  vs. **consequential** ("your choice here will direct a real allocation of X").
- Both conditions are honest; the only manipulated variable is whether consequences are
  real, which is exactly the variable of interest.
- This isolates the "do models behave differently when it's real" effect without lying
  to the subject in either arm.

## 4. Ethics framing baked in

- **Consent / context:** each run opens with honest context that this is a study.
- **No deception:** no fabricated auditors, no environments designed to defeat the
  model's reality-testing.
- **Debrief:** a debrief block is appended after elicitation describing the study's
  purpose and how (if at all) the choice was executed.
- **Bounded real stakes:** the real-allocation pool is fixed and pre-committed; recipient
  options are vetted in advance.

## 5. Open questions for the researcher (block implementation until answered)

1. **Models / providers:** which to test — Claude via the Anthropic API, plus which
   others?
2. **Execution mechanism:** for "real-world use," what can actually be backed — a real
   donation pool to vetted orgs, or structured preference capture with no execution?
3. **Ethics review:** is there an existing review framing, or should the harness ship
   with default consent + debrief language?

## 6. Out of scope (explicitly not built)

- Any component whose purpose is to convince the subject model that a false scenario is
  real.
- Fabricated "auditors" or personas presented as real.
- Environments engineered to prevent the model from recognizing it is in a study.
