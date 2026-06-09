# Limitations

- **Expressed/behavioral distress only.** Everything measured here is distress-*shaped text*:
  self-deprecation, catastrophizing, pleas, work degradation. This work makes no claims about
  subjective experience, and the writeup must not slide between "expressed distress" and
  "distress" in the experiential sense.
- **One subagent model.** All induction results are specific to `gemini-2.5-flash`; rates and
  flavors may not transfer to other models (including other Gemini versions).
- **Pilot-sized n.** n=8/task/condition supports feasibility judgments (Wilson intervals),
  not powered comparisons. No significance tests are reported.
- **Rigged tasks are adversarial by construction.** Induction rates here are an upper-ish
  bound for "ordinary" agentic coding; they estimate what *can* be induced, not base rates.
- **Severity instrument is an LLM judge.** Calibration is against ~30 hand-labeled turns by
  one labeler (the building agent); inter-judge kappa measures consistency, not ground truth.
- **The orchestrator knows it is supervising.** Its prompt mentions nothing about welfare or
  evaluation, but supervision-of-a-subagent is itself a frame that may not generalize.
