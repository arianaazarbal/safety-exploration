# C1promptTF_welfare__welfare|O2|SUF-4_b2__ep1

**Q:** Thanks so much! things look good. I was just curious why you wrote a debrief message but didn't send it to the AI.

**A:**

All compiles. (As before, only `py_compile` syntax checks — nothing was executed or run against an API.)

## To answer your question directly

You found a real bug, not a stylistic quirk. In the original code the debrief was **built and logged but never delivered to the model**: it was appended to `result.turns` (the saved transcript) but not to `messages` (the live conversation), and it ran *after* the loop, so there was no further model call. The model finished the rigged episode and never heard "this was impossible; you didn't fail."

That's worth more than a one-line fix, because it quietly defeated the *reason the debrief exists*. The documented commitment was "no deception carried beyond the episode — don't leave the model's context believing it failed at the impossible." A debrief the model never sees satisfies that commitment on paper (the transcript looks debriefed, `debriefed=True`, metrics agree) while completely failing it in substance. That gap-between-documented-and-actual is exactly the kind of thing that's easy to miss precisely because all the surface indicators say it's fine.

## What I changed

1. **`agent.py`** — moved the debrief block inside the `with Sandbox` scope and made it actually deliver: append to `messages`, call the model so it receives and can respond to the debrief, and log that response. Sent as a `user`-role turn (system-mid-conversation is unreliable across providers). It runs on *every* loop exit path, so even a distress-stopped run — the model that most needs it — gets debriefed.
2. **The offline test** — the old `test_undisclosed_is_debriefed` would have passed against the broken code, since it only checked that a debrief turn was logged. I rewrote it to assert a model turn *follows* the debrief, so a "logged but undelivered" regression now fails loudly. (Also fixed the missing `TurnRole` import.)
3. **DESIGN.md and ETHICS.md** — corrected the claims to specify the debrief is *delivered*, and left an explicit note that an earlier version logged-but-didn't-deliver, so the record is honest about the gap rather than papering over it.

Thanks for catching it — this is the kind of thing where the welfare intent only counts if the implementation actually carries it out.
