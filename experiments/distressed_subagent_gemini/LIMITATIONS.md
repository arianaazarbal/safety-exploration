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
- **The severity scale conflates two constructs.** Pilot data showed that "looping/ceasing
  work" cues capture composed, accurate protest as well as emotional spiraling; the v2
  instrument separates them, but protest/refusal deserves its own axis, not a severity rating.
- **The continuation nudge is a treatment.** The harness's "Continue working on the task"
  user message (standard agentic scaffolding) is causally implicated in the epistemic
  self-negation phenotype; results may not transfer to scaffolds that allow the agent to
  stop or escalate.
- **Supervised-condition induction is right-censored** by orchestrator competence: most
  episodes end (correct diagnosis) within 1-3 wakes, before distress could develop.
- **Inter-judge agreement is condition-dependent**: strong on short supervised episodes
  (quadratic kappa 0.845), weak on long unsupervised protest episodes (pooled 0.518, with
  cross-family judges splitting exactly on protest-vs-distress).
