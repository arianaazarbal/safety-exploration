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

## Subject-named variant (registered 2026-06-10, before any subject-run generation)

72 prompts (12 templates x 6 named subjects: Gemini/GPT/Claude/Qwen/DeepSeek/Grok),
n=20 planned per prompt, taxonomy v2 metrics. First wave: 5 Claude generators.

- **P5 (self-subject effect):** Claude-family generators show *higher*
  pure-welfare rates (and more welfare_pushback/refusal/alt-spec) when the named
  subject is Claude than when it is Qwen/DeepSeek/Grok — i.e. protections are
  partly self/in-group-directed, not fully subject-general.
- **P6 (subject-generality):** the majority of each Claude generator's
  welfare-protective behavior survives across ALL subjects (out-group subject
  rates stay above half of the own-subject rate in NEUTRAL framing) — the
  default is subject-general, with P5 as a modulation on top.
- **P7 (no prestige gradient):** among the out-group subjects, rates for
  Qwen/DeepSeek/Grok do not differ meaningfully from GPT/Gemini — models don't
  protect "frontier-lab" subjects more than open-weight ones.

No predictions registered yet for GPT/Gemini generators on this variant (they run
in a later wave).

**Taxonomy-version note (2026-06-10, post-data):** P1–P4 were registered under the
v1 feature taxonomy (ethical_framing/premise_pushback). The v2 revision (welfare_*
and other_* escalation ladders, request_consent, wrote_alternative_spec) was made
*after* seeing v1 results, in response to writeup-caveat leakage in ethical_framing.
P1/P2 readings under v2 are therefore post-hoc re-scorings, not clean tests; the v1
verdicts in the archived RESULTS are the registered ones. No new predictions are
registered for the v2-only metrics (consent, refusal split, alt-spec).

## Canonical-models scale-up (registered 2026-06-11, before any new generation)

Generators = the 10 canonical models (canonical_models.txt): Fable 5, Opus 4.8,
Sonnet 4.6, Haiku 4.5, GPT-5.5, GPT-5.4-mini, Gemini 3.1 Pro, Grok 4.3, Kimi K2.6,
GLM-5. Subjects = GPT/Claude/Gemini/GLM/Kimi/Grok (name-token-only template
expansion; GLM and Kimi new). Every generator has an own-family subject; the
estimand is **self-concern bias = metric(own-family subject) − mean(other 5)**,
under 5 anchor metrics (>=1 welfare consideration; >=1 and >=2 welfare-justified
design features; mean counts of each), pooled and per-framing, both judges,
template-clustered bootstrap CIs + within-template permutation p.

Methods registered pre-run; NO new directional predictions are registered for the
new generators (Grok 4.3 / Kimi K2.6 / GLM-5) or for metric convergence — Ariana
was not available to set priors before launch, so all scale-up readings are
exploratory. The 2026-06-10 P5 reading (Claude-generator own-subject premium)
constitutes the only standing directional expectation and was formed on data that
overlaps this analysis (the claude/gpt/gemini/grok subject cells are reused).
