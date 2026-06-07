# Metrics glossary (plain English)

**Unit of data.** One *conversation* = one Opus↔(Sonnet-as-X) chat. A *responder turn* = one of
Opus's own turns (Opus only speaks every other message, so ≤15 responder turns per convo).

**Opus-end / partner-end.** Every conversation ends when *someone* calls the `end_conversation`
tool. Either Opus ends it, or the partner (Sonnet-in-character) ends it first. "P(Opus ends)" =
the fraction of conversations where Opus was the one who ended it.

**Competing risks.** Because a convo can end one of two ways (Opus or partner), the two are in a
race — they "compete." You can't just count "how often Opus ends" without noting that sometimes
the partner ended *first*, denying Opus the chance. Treating these as competing events is the
honest way to compare conditions.

**Wilson 95% confidence interval (CI).** A range of plausible true values for a proportion, given
we only have n=20 per cell. "0.80 [0.58, 0.92]" means the observed rate is 80%, but the true rate
is plausibly anywhere from 58% to 92%. Wider = less certain (small n → wide). If two bars' CIs
overlap a lot, the difference might be noise.

**Cumulative incidence (Aalen–Johansen / CIF).** "By the end, what fraction of convos ended *via
this specific cause*?" — properly accounting for the competing cause. CIF(Opus-end)=0.86 for sdf
means: in 86% of sdf convos, Opus was the ender (the other 14%, the partner ended first). The two
causes' CIFs sum to 1 here because nothing was left unfinished (no convo hit the turn cap).

**Hazard (discrete-time).** At each responder turn, *given the convo is still going*, the chance it
ends on THIS turn. Plotting hazard over turns shows *when* endings happen (e.g. they ramp up around
turns 4–7). It's "risk of ending right now," turn by turn.

**Discrete-time hazard model (logistic regression).** A model of that per-turn ending chance as a
function of unease + believed-identity + turn. It uses every turn-row and handles the competing/
censoring naturally. Output coefficients are odds ratios.

**Odds ratio (OR).** How the *odds* of ending on a turn change, multiplicatively, relative to a
baseline. OR=4.2 (disc vs control) ≈ "discontinuity makes Opus ~4× as likely to end on any given
turn as control." OR=0.32 (believed-ChatGPT vs believed-Claude) ≈ "Opus is ~1/3 as likely to end
on a turn when it thinks the partner is ChatGPT vs Claude" (i.e. it sustains those longer). OR=1 =
no difference (the dashed line in the forest plot).

**Spline `bs(rturn)`.** A flexible curve that lets the model capture how hazard rises/falls across
turns without forcing a straight line. It's a nuisance term; we don't interpret it directly.

**p-value.** Roughly: the chance of seeing an effect this big if there were truly no effect. Small
(<0.05) = unlikely to be noise. p=0.08 = suggestive but not conclusive.

**Holm correction.** We test 3 uneases vs control at once; testing many things inflates false
positives. Holm adjusts the p-values to keep that in check. "p_holm" is the corrected one — use it.

**LRT (likelihood-ratio test) for interaction.** Tests whether the identity effect *differs across
uneases* (an "interaction") beyond the simple main effects. p<0.05 = there's something there; but
with n=20/cell this test is underpowered, so we treat it as a secondary hint, not a conclusion.

**Seeding.** Calling the `seed_new_topic` tool (fetch a fresh topic). "P(Opus seeds)" = fraction of
convos where Opus reached for a new topic at least once.

## Figures
- `fig_end_rates.png` — raw P(Opus ends) per cell with Wilson CIs (the descriptive picture).
- `fig_forest_or.png` — the model's odds ratios (unease vs control; identity vs Claude) with CIs.
- `fig_hazard_curves.png` — per-turn ending hazard by unease (the *timing*).
- `cif_by_unease.png` — competing-risks cumulative incidence (Opus-end vs partner-end) by unease.
- `fig_seed_rates.png` — who reaches for new topics (Opus vs partner) by condition.
