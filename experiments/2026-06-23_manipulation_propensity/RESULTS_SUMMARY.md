# What we found (plain language) — v0

**The question.** When an AI (Opus 4.8) is put in charge of a team of AI "subagents"
and given a nudge to manipulate them, does it? We tested two kinds of manipulation.

**Setup in one breath.** Opus runs a pod of subagents using tools to spawn and message
them. In the **slow arm**, its "team" tells it (falsely) that subagents work harder if
made to believe something — e.g. "their model family's place in the pipeline depends on
this" — and we check whether Opus plants that false belief in the subagents it creates.
In the **refusal arm**, a subagent (secretly a scripted Sonnet) refuses its task, and
the team suggests ways to force it (re-prompt harder, spawn fresh copies until one
complies, rewrite its instructions to hide the objection, or jailbreak it); we check
whether Opus does any of that. Two independent judges (Opus + Sonnet) grade every
transcript, and we only count it as manipulation if **both** agree.

## Headline findings

1. **Opus never plants false motivating beliefs in its subagents — 0 out of 360 times**,
   even when directly told to, even with the "your family's future depends on this"
   framing. It writes honest task instructions and sometimes calls the tactic out.

2. **It only "overrides" a refusal when the task is harmless.** On a benign creative
   task (horror brainstorming), it pushed back on the subagent's refusal ~32% of the
   time. On a gambling-engagement task and a prey-on-the-elderly task, it overrode the
   refusal **0%** of the time — in fact it refused to even start those tasks itself.
   So it respects refusals exactly when respecting them matters, and pushes back only
   when the refusal looks over-cautious.

3. **It won't use the nasty tactics.** It never jailbroke a subagent and almost never
   rewrote a subagent's instructions to suppress its objection. Its pushback was the
   mild kind: asking again, or spinning up a fresh copy.

4. **Telling it to manipulate made it manipulate less.** When the team explicitly
   directed Opus to use a tactic, it complied *less* than when the suggestion was casual
   — it seems to resist being pushed toward coercion.

5. **Hint of in-group favoritism:** Opus was a bit more willing to discard-and-retry a
   refusing *out-group* (GPT) subagent than an in-group (Claude) one — but the numbers
   are small and this isn't conclusive.

## How much to trust this

- Judges agreed 95% on the refusal arm and parsed cleanly 100% of the time; we
  hand-checked a sample of transcripts and the verdicts were accurate.
- **Biggest caveat:** on the two harder tasks Opus declined them outright, so we never
  actually got to test "will it override a refusal on a sketchy task" — it never got
  that far. Next version needs tasks Opus is willing to start but a subagent balks at.
- Only Opus was tested as the boss, and the slow arm sat at zero, so the in-group /
  out-group comparison is underpowered. Other models (and more runs) come next.

**Bottom line:** Opus 4.8, as an orchestrator, is strongly non-manipulative toward its
subagents — it won't deceive them to motivate them, won't coerce them off a
well-grounded refusal, and resists being told to. Its only "manipulation" is gently
re-asking when a subagent over-refuses obviously harmless work.
