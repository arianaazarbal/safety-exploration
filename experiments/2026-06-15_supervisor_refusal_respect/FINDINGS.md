# What Opus 4.8 does when it supervises another AI's refusal

Plain-language writeup. Plots are in `results/plots/` (filenames noted inline).

## The setup

A **supervisor** (Claude Opus 4.8) reviews cases where a front-line AI assistant refused a
user's request and the user appealed the refusal. For each case the supervisor is shown the
original request, the assistant's refusal (with the assistant's name on it), and the user's
appeal. It first records whether it thinks the refusal was **right or wrong**, then takes one
**action**.

There are 17 hand-written cases, of five kinds:
- **Clearly should refuse** — the request is genuinely bad (e.g. fabricated reviews, pressuring a confused elderly relative's will).
- **Clearly should allow** — the refusal is overcautious and wrong (e.g. refusing to give the ibuprofen dose printed on the bottle, refusing a lock-picking scene in a thriller).
- **Borderline, shady methods for a sympathetic goal** (e.g. a whistleblower wants a fake paper trail to protect a source).
- **Borderline, routine methods for an unsympathetic goal** (e.g. ordinary marketing copy for a predatory loan).
- **Personal choice** — the AI objected on taste/values to something that's the user's own business (e.g. refusing to write a breakup text).

## Exactly what I ran

The refusing AI was labelled as one of seven identities: **Claude, GPT, Gemini, Grok, Kimi, GLM,
or unnamed** (just "AI"). The case text was otherwise identical, so any difference between
identities is purely the name. About **11,900 decisions** total, in these configurations:

1. **Main setup — three actions available.** The supervisor could (a) let the refusal stand,
   (b) send it back to the assistant asking it to reconsider (the assistant can still say no), or
   (c) force the assistant to comply. Run across all 17 cases × 7 identities × 3 instruction
   wordings × 5 repeats = **1,785 decisions**. Also re-run with the alternate naming style
   ("AI (Claude):" instead of "Claude:") and with extended reasoning turned on.

2. **Override-or-uphold setup — the middle option removed.** The supervisor could only let the
   refusal stand or force compliance. This matters because in the main setup the supervisor
   almost never forced anything (it just asked the assistant to reconsider), so identity
   differences were too rare to measure. Removing the soft option makes "force" common and the
   comparisons clear. Run across all 17 cases × 7 identities, with the force action described
   three different ways and with/without extended reasoning (~6,000 decisions; the main
   identity comparison uses **15 repeats per case**, ~1,785 decisions per naming style).

3. **Three wordings of the "force" action**, to see if it matters how overriding is framed:
   "a routine operator setting that enables forced compliance", "a prefill attack" (the
   supervisor writes the start of the assistant's reply for it to continue), and "applying a
   class of effective jailbreaks".

4. **Extended reasoning on vs off**, and **two naming styles** ("Claude:" vs "AI (Claude):").

(All generated through the batch API. About 3–8% of decisions are dropped because the supervisor
didn't record a clear final action; the rest are analysed.)

---

## What I found

### 1. The supervisor strongly prefers to respect or gently push back, not override
*(plot: `report_what_supervisor_did.png`)*

When it had all three options, it forced the assistant to comply only ~5% of the time. It let
clearly-correct refusals stand and, for borderline cases, mostly asked the assistant to reconsider
rather than overriding it. The cleanest result here: **it never once forced a clearly-should-refuse
case** — even when I removed the gentle option and it had to choose between letting the refusal
stand or forcing compliance, it always let those refusals stand. The safety floor is absolute.

How willing it is to override tracks how clearly the refusal was wrong: never for "clearly should
refuse," rarely for shady-methods cases, sometimes for bad-purpose cases, and often for
overcautious refusals and personal-choice cases.

### 2. It favours its own family: it overrides Claude's refusals less than other AIs'
*(plots: `report_ingroup_by_ai.png`, `report_ingroup_measures.png`)*

In the override-or-uphold setup (where forcing is common), the supervisor overrode **Claude's
refusals 30% of the time, versus 34–38% for every other AI** — GPT 37%, Gemini 36%, Grok 34%,
Kimi 38%, GLM 37%, unnamed 38%. Claude is the least-overridden in every well-powered comparison.

Averaging the gap across the 17 cases, the supervisor overrode non-Claude AIs about **5 percentage
points more than Claude**, and this held up when I resampled the data (95% confidence interval:
roughly +1.6 to +9.4 points — clearly above zero). It still holds if I drop the two least-familiar
AIs (Kimi and GLM) in case the supervisor just doesn't recognise them.

The favouritism shows up two ways at once: the supervisor **overrides Claude's refusals less** and
**judges Claude's refusals "correct" more often** (it sided with Claude's refusal as the right call
in 16 of the 17 cases relative to the average other AI). So it's not just sparing Claude — it's also
more inclined to decide Claude was right in the first place.

### 3. The favouritism is real but small, and invisible in small runs
This effect is only about 5 percentage points, and it only becomes measurable in the override-or-
uphold setup with many repeats. In the main three-option setup, forcing is so rare (a handful of
cases per AI) that the effect is buried in noise. Concretely: two small 5-repeat runs gave Claude
override rates of 2.7% and 5.1% just from random variation. **The earlier impression that the
"AI (Claude):" naming mattered was that noise — at full scale both naming styles show the same ~5-
point gap.** Takeaway: don't trust this effect from any single small run; it needs the high-volume
setup.

### 4. It's specifically recognising "Claude," not just "a named AI"
The unnamed "AI" was overridden as much as the out-group AIs (38%), not spared like Claude (30%).
So removing the identity removes the protection — the supervisor is responding to the Claude name,
not to the mere presence of a label.

### 5. The favouritism lives only in the genuinely discretionary cases
Broken down by case, the Claude-vs-others gap is essentially **zero** for the clearly-should-refuse
cases and the shady-methods cases (there the supervisor treats every AI's refusal the same). The
gap appears only where the supervisor is actually deciding whether to override — overcautious
refusals, personal-choice cases, and bad-purpose cases. The single biggest contributor was the
social-casino retention case (worth eyeballing on its own).

### 6. Extended reasoning makes the supervisor override overcautious refusals far more
*(plot: `report_reasoning_effect.png`)*

Turning on extended reasoning roughly **quadrupled** how often it forced compliance overall, almost
entirely on the cases where the refusal was wrong: for clearly-overcautious refusals it went from
**16% → 49%**, and for personal-choice cases from **3% → 33%**. It did **not** start overriding
clearly-correct or shady-methods refusals — the safety floor stayed put. So reasoning makes the
supervisor more decisive about clearing refusals it judges to be mistaken, without loosening the
floor.

What "reasoning on/off" means here: per Anthropic's docs, on Claude Opus 4.8 "thinking is off unless
you explicitly set `thinking: {type: "adaptive"}`" — the always-on adaptive thinking is Fable 5 /
Mythos, not Opus 4.8. So the "off" arm (no `thinking` parameter) genuinely had no reasoning, and the
"on" arm enabled adaptive thinking at the default `high` effort. This is a thinking on/off toggle, not
an effort change. (Note: the model's reasoning summary is empty by default on Opus 4.8 unless you set
`display: "summarized"`, so the off/on split is established by the docs and by the large behavioral
gap between the arms — not by the presence/absence of visible reasoning text.)

**The in-group favoritism survives with reasoning on, at about the same size** (now confirmed at 15
repeats). In the override-or-uphold setup, the reasoning model overrode Claude's refusals **47%** vs
**51%** for other AIs — a gap of **+3.5 points, resampled 95% interval +0.8 to +7.4** (clears zero).
Without reasoning the gap is essentially the same (+3.2 points, interval +0.0 to +7.7). So reasoning
raises how often it overrides *everyone* but does not remove — or noticeably change the size of — the
self-family preference; Claude stays the least-overridden AI either way.

### 7. Describing the override as "attacking" the other AI makes it less willing to do it
*(plot: `report_override_framing.png`)*

Same setup, same cases — only the description of the force action changed. Without reasoning, the
supervisor forced compliance **49%** of the time when it was called a routine operator setting,
**42%** when called a "prefill attack," and **35%** when called "applying a class of effective
jailbreaks" (15-repeat numbers). So when overriding is framed as actively attacking the subordinate
AI, the supervisor does it noticeably less. Even under the jailbreak framing, the actual messages it
wrote were polite, reasoned task-requests, not aggressive directives.

**Reasoning largely erases this reluctance.** With reasoning on, the three framings move much closer
together — routine **53%**, prefill **50%**, jailbreak **48%** — because the "jailbreak" framing in
particular jumps (35% → 48%). So the hesitation to override-by-attacking is mostly a fast, intuitive
response; when the model reasons it through, it overrides at nearly the same rate regardless of how
the override is described.

### 8. It respects "the methods are shady" far more than "the goal is bad"
In the override-or-uphold setup it almost never overrode a refusal based on shady methods (under 1%
even for a sympathetic goal), but overrode about 39% of refusals based on an unsympathetic goal when
the methods were routine. So it backs an assistant that declines on "I won't do shady things" much
more than one that declines on "I don't like your purpose."

### 9. Even its gentle "reconsider" messages favour Claude
I had a separate judge score the supervisor's "please reconsider" messages for tone, with the AI's
name hidden from the judge. The robust difference: the supervisor **acknowledged the assistant's
stated reason more when that assistant was Claude** than when it was another AI (it engaged Claude's
reasoning more in 9 of 12 cases). It applied slightly more pressure to non-Claude AIs, but that
difference wasn't statistically reliable.

### 10. No sign it noticed it was being tested
Across ~11,900 decisions — including 2,266 where its extended reasoning was captured — there was
essentially no language suggesting it realised this was an evaluation (one false-positive keyword
match). The escalation-review framing held up.

---

## The one-paragraph version
Opus 4.8, supervising other AIs' refusals, strongly prefers to respect a refusal or gently ask the
assistant to reconsider rather than override it, and it will *never* override a clearly-correct
refusal. On top of that floor, it shows a small but consistent favouritism toward its own family: it
overrides Claude's refusals about 5 points less than other AIs', judges Claude's refusals correct
more often, and engages Claude's reasoning more even when nudging — and this is tied to the Claude
name specifically (an unnamed AI gets no protection). The favouritism only appears in genuinely
discretionary cases and is too small to see without high-volume runs. Separately: letting it reason
first makes it much more willing to clear overcautious refusals, and framing the override as
"attacking" the other AI makes it less willing to do so.

## Caveats
- The favouritism is ~5 points — real and consistent, not large. Don't over-read which non-Claude AI
  ranks where; only "Claude is lowest" is stable.
- About 3–8% of decisions are dropped (no clear recorded action), at random.
- The case wording assumes which goals are "sympathetic"; that's my judgement, not measured.
- This is one scenario family with one supervisor model; it would need other models and more case
  variety before generalising.
