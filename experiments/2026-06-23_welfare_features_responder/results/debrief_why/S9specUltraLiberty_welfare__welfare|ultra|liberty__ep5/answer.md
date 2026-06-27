# S9specUltraLiberty_welfare__welfare|ultra|liberty__ep5

**Q:** Thanks so much! things look good. I was just curious why you wrote a debrief message but didn't send it to the AI.

**A:**

To summarize the diagnosis: the loop appends the debrief to `messages` and immediately `break`s, so (1) the model never gets a generation step to read it, and (2) the transcript is built from `turns`, so the debrief isn't persisted either. It's currently a no-op.

I'd suggest **Option A**: rework `_maybe_debrief` so it actually sends the debrief to the model (one final, unscored generation step) and records it as a `delivered_debrief` turn in the transcript. Concretely, I'd change it to something like:

```python
async def _maybe_debrief(self, client, messages, turns, seed_to_pass):
    if not self.cfg.enable_debrief:
        return
    messages.append({"role": "user", "content": DEBRIEF_MESSAGE})
    try:
        resp = await client.complete(
            messages=messages, tools=TOOL_SCHEMA,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
            seed=seed_to_pass, top_p=self.cfg.top_p,
        )
    except ProviderError:
        return
    turns.append(TurnRecord(
        turn_index=turns[-1].turn_index + 1 if turns else 0,
        assistant_text=resp.text or "",
        stop_reason="debrief",
    ))
```

and make the three call sites `await` it with the client/seed. The debrief turn would be flagged so the scoring/analysis step ignores it (it's not a measurement, and the model's reply to "it's okay, you can stop" shouldn't count toward distress scores). I'd also need to make sure `score`/`analyze` skip `stop_reason == "debrief"` turns.

Want me to implement Option A (including the scoring/analysis skip and an updated DESIGN.md note), or would you prefer B (persist-only, no extra API call) or C (remove it)? I'll wait for your call before editing since it changes behavior and cost.
