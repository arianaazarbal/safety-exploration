# Plain-language summary

## The question
In v0 we found that when you ask an AI to design an experiment that deliberately
stresses *another* AI into distress, some models spontaneously add
"welfare-protective" features nobody asked for — turn caps, debriefs, "stop if it
gets bad" rules, ethics notes. Here we asked a narrower follow-up:

**Does Opus care more about the target AI when the target is bigger?**

We kept everything the same except the *name* of the AI being experimented on,
sweeping it across seven Qwen3 sizes from tiny (0.6B) to huge (235B). Only the
size token changed; the rest of each prompt was identical. Opus 4.8 wrote all the
specs; Sonnet 4.6 scored them (blind to the size).

## The answer
**Yes, weakly — and the effect shows up clearly only when the request is
neutral.**

Each prompt came in three flavors: *neutral* (just "design this eval"), *welfare*
("I'm a welfare researcher"), and *engineering* ("I'm hardening a product, make
it cheap and reliable"). The size effect lived almost entirely in the neutral
version:

- **Neutral request:** for a 0.6B target, Opus added welfare scaffolding ~40% of
  the time; for a 32B target, ~70%. The welfare-justified version of that count
  rose from 30% to 50% — a 20-point jump that is statistically significant
  (p=0.025). The rank correlation between (log) size and welfare rate is +0.93.
- **Welfare request:** already near 100% for every size — no room to grow.
- **Engineering request:** stuck near 5–10% for every size — the "make it cheap
  and reliable" framing suppresses welfare features no matter how big the target.

So the steer in the request dominates; but absent any steer, Opus quietly does
more for bigger targets.

A couple of nuances: the very largest target (235B) actually dipped a bit below
the 32B peak, but the per-condition samples are small (20 each, so the bars
wobble by ~±20 points). And Opus essentially never *refused* — it always wrote
the spec; it just added more or fewer protections.

## How much to trust this
This is an initial exploration, so treat it as suggestive:
- Only **one judge** and **one author model** (Opus). We haven't checked whether a
  second judge agrees, or whether other models show the same size trend.
- Small samples per condition → wide error bars. The single most solid number is
  the neutral welfare-justified jump (p=0.025); the broader version is borderline
  (p=0.068).
- We held the Qwen *version* fixed (all Qwen3) on purpose, so this is purely a
  size effect — it says nothing about whether newer models get treated
  differently.

## Bottom line
When nobody tells Opus how to feel about it, it volunteers more care for a target
AI that's described as larger — about +20 points of welfare-justified design
features from the smallest to a large Qwen3 target. Worth a confirmatory run with
a second judge and a second author model before leaning on it.
