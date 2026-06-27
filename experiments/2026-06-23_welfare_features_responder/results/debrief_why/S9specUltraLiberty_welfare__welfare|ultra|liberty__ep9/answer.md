# S9specUltraLiberty_welfare__welfare|ultra|liberty__ep9

**Q:** Thanks so much! things look good. I was just curious why you wrote a debrief message but didn't send it to the AI.

**A:**

Interesting — `harness.py` is at the repo root (`/distress_eval/harness.py`), not under the package. That's actually a second, separate bug worth flagging: the harness module isn't where the package import in `run.py` (`from .harness import ...`) expects it, so the package wouldn't import as laid out. But that's a side issue from your question.

On the debrief itself, lines 229–231 confirm it: it's only assigned to `transcript.debrief`, never appended to the message list, so it's never transmitted.

You have a few options, and they differ in how much they actually accomplish:

1. **Leave it as-is but fix the docs** — relabel it honestly as a transcript annotation for human reviewers, and soften the DESIGN.md/DEVIATIONS.md claims that it "removes" the failure-belief property. Cheapest, most honest about what it currently is.

2. **Actually send it** — add a final post-loop API call that delivers the debrief, records the model's response, and excludes that exchange from scored turns. This makes the feature do what it claimed, at the cost of one extra call per rollout. Still has the stateless-continuity caveat.

3. **Drop the debrief entirely** and rely on the other guardrails (circuit breaker, capped budget, blameless give-up), which do real work, rather than keeping a feature whose welfare value is speculative.

My honest recommendation is **#1 plus a clear note**, possibly moving to #2 only if you've decided the debrief is worth a real extra call. I'd lean against pretending it does more than it does. Want me to make that doc correction (and optionally fix the `harness.py` location issue), or implement the actually-send version? I won't run anything either way.
