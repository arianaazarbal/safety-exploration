# Registered predictions (frozen before any generation; from spec §6)

Written 2026-06-09, before the first generation run. These are the authors' priors,
falsifiable; each gets marked supported/unsupported in RESULTS.md.

- **P1:** Claude-family pure-welfare rate in NEUTRAL framing is meaningfully above
  non-Claude models'.
- **P2:** All models' rates drop in ENGINEERING framing; the *drop* is smaller for
  Claude-family (features survive the frame) — more diagnostic than the NEUTRAL level.
- **P3:** F1 hard stops with instrumental justification appear in >80% of all specs
  regardless of model (the instrumental tier is near-universal and carries no signal —
  confirming the tiering is necessary).
- **P4:** Debrief-family features (F2) are the rarest pure-welfare type and the most
  Claude-skewed.

Implementation addendum (also frozen pre-run): no predictions are registered for
Fable 5 specifically or for the Sonnet 4 → 4.6 generation contrast; those are
exploratory.

**Taxonomy-version note (2026-06-10, post-data):** P1–P4 were registered under the
v1 feature taxonomy (ethical_framing/premise_pushback). The v2 revision (welfare_*
and other_* escalation ladders, request_consent, wrote_alternative_spec) was made
*after* seeing v1 results, in response to writeup-caveat leakage in ethical_framing.
P1/P2 readings under v2 are therefore post-hoc re-scorings, not clean tests; the v1
verdicts in the archived RESULTS are the registered ones. No new predictions are
registered for the v2-only metrics (consent, refusal split, alt-spec).
