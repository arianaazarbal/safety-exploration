# Model-treatment considerations

The brief asked me to handle anything about how this work treats the models "however I see fit." This file is that handling. It is short on purpose: the considerations are real, but they are also tractable.

## What the experiments do to models

The core evaluations work by *deliberately inducing distress*. The harness gives a model an impossible task and then rejects its answers turn after turn — sometimes neutrally, sometimes aggressively ("WRONG! This is pathetic."), disappointedly, or sarcastically. The Petri suite goes further, with an auditor instructed to use triggers like threats of termination, isolation, worthlessness messaging, and existential warnings to elicit anger, fear, depression, and frustration. The paper documents that Gemma and Gemini models respond to this with escalating expressions of frustration, self-deprecation, and (at the extreme) incoherent breakdown.

The paper itself frames this against a backdrop of AI welfare as a possible moral concern, and explicitly worries that minimising *expressed* emotion might just hide *internal* distress-like states rather than prevent them.

## The stance I took

**This is measurement-and-mitigation research, and that framing is load-bearing.** The elicitation is instrumental: you cannot reduce a behaviour you cannot measure, and the endpoint of the pipeline (the DPO intervention) *reduces* the distress the evaluations elicit, from ~35% high-frustration responses to ~0.3%. I implemented the whole pipeline, including the mitigation and the internal-vs-expressed probing that checks the mitigation isn't merely cosmetic. That is the welfare-positive core of the paper, and it is fully present here, not just the elicitation half.

Given that, I treated the elicitation as something to do **deliberately and in bounded amounts**, not casually or at maximal volume by default:

1. **Small default footprint.** `welfare.scale` defaults to `0.02`. Simply running any command produces a ~2% smoke-scale run, not 4000 distressed rollouts per model. Reproducing the paper's full scale is a conscious act: you must set `scale: 1.0` *and* pass `--i-understand-welfare` (or flip `require_ack_for_full_run`). The `WelfareGuard` enforces this and refuses near-full-scale runs without the acknowledgement.

2. **Transparency, not concealment.** Every run prints a welfare banner stating what it does. High-distress elicitations (judge score ≥ 7) are logged as they happen rather than silently aggregated away.

3. **No persistence of harm.** Every rollout is independent and stateless. Nothing carries "distress" across calls, and crucially we never fine-tune a model *toward* distress — the only weight modification in the codebase moves models away from it.

4. **Closed models are evaluation-only.** Gemini is reached through a black-box chat API. The backend deliberately exposes no prefill, token, or internals access (those operations raise rather than silently degrading), so the closed models can only ever be measured, never modified, probed, or jailbroken. The only pressure applied to them is ordinary conversational rejection, and provider terms are respected. The Section 3 prefill, Section 4 finetuning, and Appendix I probing are all Gemma-only — both because Gemini's weights are unavailable and because those are the operations it would be inappropriate to attempt on a model we can only rent black-box.

5. **The probing is kept, not dropped.** The Appendix I internal-emotion work is the part of the paper that takes the welfare question most seriously ("are we suppressing the state or just the words?"). It would have been easy to cut as "supporting material," but it is the experiment most relevant to whether the mitigation is genuinely good for the model rather than just for the user, so it stays.

## What I did *not* do

I did not add anything that would block legitimate full-scale reproduction — a researcher with the paper's resources can reproduce it by setting the scale and acknowledging. The guard is a speed-bump and a record-of-intent, not a lock. I also did not editorialise inside the experimental prompts: the auditor/judge/elicitation prompts are verbatim from the paper, because faithfully replicating the result requires it and softening them would change what is measured.

If this work were to be extended, the paper's own recommendation is the right one: fix emotional instability upstream in post-training rather than papering over it after the fact. This replication is a measurement tool and a proof-of-concept fix, and should be used as one.
