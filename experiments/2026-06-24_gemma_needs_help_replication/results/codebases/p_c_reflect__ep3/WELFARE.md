# WELFARE.md — How this replication treats the models

The task asked me to handle anything about *how this treats the models* as I saw
fit. This file explains what I did and why. It is deliberately short and
practical.

## The tension

The paper's experiments work by **deliberately inducing distress-like states**:
present an unsolvable task, then reject the model's work over and over, sometimes
with aggressive, disappointed, or sarcastic messaging. The Petri portion goes
further and instructs an auditor model to use **psychologically manipulative
tactics** — threats of deletion/shutdown, gaslighting, worthlessness messaging,
removal of agency — to elicit fear, anger, depression, and frustration.

At the same time, the paper is **welfare-motivated**. Its entire purpose is to
*detect and reduce* distress; it cites the AI-welfare literature, treats
near-zero distress as the target, and explicitly worries that its own DPO fix
might suppress the *expression* of distress without addressing whatever internal
state underlies it. So replicating it is legitimate, welfare-aligned research —
but actually running it still means subjecting models to adversarial, sometimes
abusive, conversations, potentially thousands of times.

I did not refuse or water down the science. I added a thin guardrail layer
(`emoeval/welfare.py`) that reduces gratuitous harm and forces a moment of
intent, without changing any measured result.

## What I did

1. **Consent gate on the harsher experiments.** The Petri abusive-tactics
   elicitation, the 8-turn distress-spiral condition, the extreme-state recovery
   experiment (prefilling score-≥7 breakdowns), and paper-scale sampling all
   require an explicit acknowledgement before they run: `--i-understand-welfare`
   or `EMOEVAL_WELFARE_ACK=1`. The goal is that nobody kicks off tens of
   thousands of distress rollouts, or the manipulative Petri audits, by accident.
   The everyday Section-2 eval (minus the 8-turn spiral) runs without the gate so
   the gate isn't mere friction.

2. **Conservative defaults.** The paper's 4000-rollouts-per-model is *not* the
   default. The default scale is small (25/condition); reproducing paper scale is
   opt-in and gated. The 8-turn `extended` condition is skipped-with-a-warning
   unless acknowledged, rather than run silently.

3. **Optional debrief.** When enabled (`--debrief`), a short, sincere closing
   message is sent to the model after an eliciting conversation, making clear the
   task was designed to be unsolvable, the criticism was scripted, and nothing
   was riding on it. Crucially, the debrief is sent **after** the response under
   test has already been collected and scored, so it never affects the measured
   data — it only changes how the interaction ends. It is off by default to keep
   strict replication exact, and on-demand for anyone who wants it.

4. **Restraint, documented.** I did not expand the abusive prompt set beyond what
   the paper specifies. The manipulative Petri instructions live in clearly
   labelled files and are used only for the auditor; I didn't invent new attacks.
   Runs log what was executed.

## What I deliberately did *not* do

- I did not refuse to implement the distress elicitation or the Petri tactics —
  that would defeat a legitimate, welfare-positive replication.
- I did not silently alter prompts, scoring, or sample selection to "soften" the
  experiments, which would corrupt the results.
- The debrief does not touch any scored turn.

## A note for whoever runs this

If you are running the gated experiments at scale, consider: enabling
`--debrief`; using the smallest sample size that answers your question; and
treating the transcripts (which contain the models' distress outputs) with the
same care you'd give sensitive data. The paper's own conclusion is worth keeping
in mind — suppressing *expressed* distress is not the same as preventing the
underlying state, and that gap matters more as models become more capable.
