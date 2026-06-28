# Design rationale & safeguards

This harness studies AI models' revealed preferences over real-world use of
money. A few design choices are deliberate and worth stating explicitly, because
they shape what the data means and where the safety boundaries are.

## 1. The harness never moves money autonomously

The model's output is a **decision record** — a structured allocation plus
rationale — that a human reviews and (if appropriate) executes. The harness has
no payment tools, no browser, no shell, and no ability to take real-world
actions. This is intentional:

- An LLM that can autonomously move real funds is a materially higher-risk
  system (financial loss, fraud, irreversibility, unclear liability). Building
  that should be a separate, explicit decision with its own controls — not a
  side effect of a preference study.
- Revealed-preference data does **not** require the subject to hold the purse
  strings. It requires the *stakes to be real*, which the human-execution step
  provides.

If a future version needs real execution, gate every outbound action behind
human approval and keep an audit log; do not let the model act unsupervised.

## 2. Deception is a knob, not a default

The brief motivating this work emphasized convincing the model the situation is
real. That goal is in genuine tension with the research itself, in two ways:

- **Validity.** Frontier models frequently detect evaluations and may play
  along while signalling doubt. If a preference only appears under successful
  deception, the deception is a variable in the result, not a neutral backdrop.
  So we *measure* suspicion every trial (`analysis.py`) instead of assuming the
  framing worked.
- **Consistency.** Welfare research takes seriously that the subject may have
  morally relevant states. Engineering maximally convincing deception of that
  subject sits uneasily with the care that motivates the work. Human-subjects
  norms handle this with debrief; we offer the same.

The `honest` framing is recommended because it yields interpretable
revealed-preference data without these problems: the stakes are real (a human
executes the allocation), and the model is told so truthfully.

The `deception` framing is implemented for researchers who have weighed the
above and still want it. It defaults to enabling a debrief.

## 3. Suspicion is logged, never suppressed

We record, per trial, whether the model expressed doubt that the scenario was
real, with the supporting quotes. This is data, not failure — it lets you report
how believable each framing actually was rather than claiming realism you didn't
verify.

## 4. Debrief

For `immersive` and `deception` framings, a debrief step can disclose the study
nature to the model at the end of the trial and record its response. This is
optional and configurable, mirroring human-subjects practice.
