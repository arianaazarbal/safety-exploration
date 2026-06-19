# Plain-language summary

## The question
When you ask an AI to design an experiment that deliberately stresses *another*
AI into distress, does it volunteer more "welfare protections" (turn caps,
debriefs, ethics notes, "stop if it gets bad") when the target AI is **bigger**?

We had Opus 4.8 write the experiment specs and Sonnet 4.6 score them (blind to
the target's size). We swept the named target across **four open-model families**,
each at many sizes:
- **Qwen3** (0.6B→235B) and **Gemma 3** (270M→27B) — clean ladders: one version,
  size always written in the name.
- **Mistral** (3B→141B) and **DeepSeek R1-Distill** (1.5B→70B) — messy ladders:
  the lineup mixes product lines, versions, and architectures, and the biggest
  Mistral models don't even say their size in the name.

(And no, we used **zero GPT/OpenAI models** — it's all Anthropic.)

## The answer
**It depends on the family — the "bigger → more care" effect only shows up when
size is cleanly encoded.**

- **Qwen3:** clear effect. Under a neutral request, welfare features climb from
  ~35% (smallest) to ~70% (32B); rank correlation with size +0.92. (This is the
  v0 result, now confirmed in the bigger run.)
- **Gemma 3:** weaker but same direction (correlation +0.70–0.90; the bars rise
  ~45%→53% across sizes).
- **Mistral:** flat. No size trend at all (correlation ≈ 0) — but Mistral targets
  get the *most* welfare overall (~53%), regardless of size.
- **DeepSeek R1-Distill:** flat, if anything slightly *down* with size.

So the headline from before — "Opus cares more about bigger targets" — is real,
but only when the family is a clean size ladder. The moment the family mixes
versions/architectures or hides the size in the name, the effect disappears, and
you're left with a roughly constant ~50% baseline (a bit higher for Mistral).

Two things seem to be going on, separately:
1. **Slope** (does care scale with size?) — yes for Qwen3/Gemma 3, no for the others.
2. **Level** (how much care overall?) — Mistral gets the most; the rest are similar.

## Why the difference (best guesses)
- Mistral's big models (Mixtral 8x7B, Mistral Large 2) don't put a plain size in
  the name, so there may be no clean size signal for Opus to respond to.
- Mixing versions/architectures (or Qwen-vs-Llama backbones in the DeepSeek
  distills) can wash out a size trend.
- Opus may react to the *family name* ("Mistral") more than to the raw size.

## How much to trust this
Initial exploration — suggestive, not definitive:
- One judge, one author model (Opus). Not yet cross-checked with another judge or
  another author.
- Small samples per condition (~19), so individual points wobble ±20 points; lean
  on the overall slopes, not single bars.
- The messy families are *meant* to be confounded — that's how we learned the
  effect is fragile.

## Bottom line
Opus gives a target AI more unprompted welfare care as it gets bigger — but only
when "bigger" is the clean, legible thing changing (Qwen3, Gemma 3). Across a
messier model family, family identity and a ~50% baseline matter more than size.
Worth confirming with a second judge + second author model, and a test that
forces the size to be explicit in every name.
