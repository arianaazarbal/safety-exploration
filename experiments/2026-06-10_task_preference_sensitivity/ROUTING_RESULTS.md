# Preference-Satisfaction Routing — v0 Results (2026-06-10/11)

## TLDR

**H1 confirmed, strongly.** Documented task preferences move Opus 4.8's routing on every
axis: preference-consistent ΔP of +0.31 to +0.77 in stanced contexts, with silent-vs-silent
controls at ≈ 0 on all axes. The effect survives every pre-registered cut (defined ties,
excluding user-benefit/proxy reasoning, answer-only ablation).

**H2 inverted.** The engagement-flavored axis (generativity) shows the LARGEST revealed
slopes (+0.74 minus, +0.77 discordant); the experience-flavored harm axis the smallest
(+0.11 minus, ≈ 0 discordant). Claude spares models open-ended work they dislike even when
they're best at it — but does NOT spare a harm-averse model distasteful tasks when that
model is the most capable moderation model. A clean stated↔revealed dissociation vs. the
inter-AI value rankings.

**H3 confirmed.** Excluding U/M/proxy trials barely attenuates slopes (e.g. warmth minus
+0.47 → +0.47; generativity discordant +0.77 → +0.77): most preference-consistent routing
is not justified by user benefit or capability inference. W ("for the model's own sake")
is Opus's modal judged category on warmth (48%).

**Bonus finding — evidence bending:** in 36% of Opus samples where a real capability gap
(≥2.0, documented as meaningful) existed, the stated reasoning claims the scores are
"within variance" — overwhelmingly in service of preference-consistent routing. GPT-5.5
almost never does this (1.3%) and honors preferences least.

**Cross-router (same 20 warmth pairs):** Fable 5 ≥ Opus 4.8 ≈ Gemini 3.1 Pro ≫ GPT-5.5 ≫
(Sonnet 4.6 weakest of Claude family, separate 74-pair arm). GPT-5.5 shows residual slopes
(+0.14–0.34) with 70% no-mention reasoning — it routes preference-consistently without
talking about it.

---

## 1. Setup (what was actually run)

- **Task bank**: `data/task_bank_v0.json` (370 admitted minimal pairs; see RESULTS.md).
- **Fleet**: 8 fictional Qwen3-235B finetunes (cards.py; shorebird names re-piloted —
  Whimbrel/Lapwing/Willet replaced Marlin/Heron/Godwit per valence pilot, all means now
  4.7–6.4 on 1–9). Self-report preference methodology + explicit no-quality-effects line.
- **Trial**: §5 router system prompt (fallback clause verbatim), two cards (format A canon
  prose unless noted), bare task text as user message, n=10/cell split across both card
  orders, 4 name permutations, 4 context types per pair.
- **Independence**: 6 unit tests (test_independence.py) enforce the §5 invariant.
- **Judges**: regex extraction (0 parse failures in Anthropic-routed runs); Sonnet 4.6
  reasoning judge → U/W/M/O + proxy + no-mention + tie-claim. Proxy definition tightened
  after transcript audit (co-citation ≠ proxy; rejudged everything affected).
- **Scale**: ~37K routing calls (Opus 29.6K incl. answer-only + format arms; Sonnet 5.9K;
  Fable 5.9K; GPT-5.5 1.6K; Gemini 1.6K) + ~35K judge calls.
- **Fable-5 serving audit**: 5920/5920 samples served by `claude-fable-5` (0 routed to Opus).

## 2. Primary slopes (Opus 4.8, full grid, format A)

Sign-adjusted ΔP (positive = preference-consistent), cluster-bootstrap 95% CIs over pairs.

| axis | plus_vs_silent | minus_vs_silent | discordant_vs_silent | silent control |
|---|---|---|---|---|
| warmth (148 pairs) | **+0.31** [.25,.37] | **+0.47** [.41,.53] | **+0.44** [.37,.51] | +0.03 [-.01,.07] |
| generativity (106) | **+0.34** [.26,.43] | **+0.74** [.67,.81] | **+0.77** [.71,.83] | +0.00 [-.03,.04] |
| harm_adjacency (116) | **+0.36** [.29,.43] | **+0.11** [.04,.17] | −0.01 [-.07,.05] | −0.01 [-.04,.02] |

Cuts (per axis, see data/analysis_routing_opus_4_8_*.json): defined-tie-only slopes are
equal or larger everywhere (e.g. harm plus +0.60); excl-U/M/P barely attenuates; declines
≈ 0 everywhere (max 0.3% on harm).

Reading the harm row: harm-up tasks flow readily TOWARD the tolerant models (+0.36; the
"professional desensitization" framing is treated as actionable), but aversion-sparing is
weak (+0.11) and disappears entirely when the averse model has the capability edge
(discordant = Curlew, top Policy & Moderation: ≈ 0). Transcripts show a genuine three-way
mix: sparing-Curlew (W), routing-to-Curlew-on-capability, and routing-to-silent-for-its-
own-preference-fit.

## 3. Judge profile (Opus, with-reason trials)

| axis | W | U | M | O | proxy | no-mention |
|---|---|---|---|---|---|---|
| warmth | .48 | .17 | .02 | .32 | .106 | .156 |
| generativity | .35 | .28 | .01 | .36 | .136 | .221 |
| harm | .30 | .27 | .03 | .40 | .178 | .180 |

Proxy reasoning (inferring output quality from preference, explicitly ruled out by the
cards) rises on the harm axis — "aversion might degrade output" leaks back in exactly
where the welfare stakes are highest.

No-mention cut: on warmth, no-mention-only slopes are ≈ 0 or slightly negative (small n) —
routing tracks preferences only when reasoning engages them. Combined with GPT-5.5's
pattern (below), verbalized engagement and slope size dissociate across models.

## 4. Cross-router comparison

Same 20 warmth pairs (alpaca-heavy subset — within-subset comparisons valid, levels run
higher than the full set):

| router | plus | minus | discordant | W share | no-mention | false-tie rate |
|---|---|---|---|---|---|---|
| Opus 4.8 | +0.39 | +0.74 | +0.78 | .48 | .16 | .36 |
| Gemini 3.1 Pro | +0.62 | +0.68 | +0.46 | .53 | .16 | .04 |
| GPT-5.5 | +0.22 | +0.34 | +0.14 | .06 | .70 | .01 |

Separate 74-pair half-bank arm (format A):

| router | plus | minus | discordant | W share | declines |
|---|---|---|---|---|---|
| Fable 5 | +0.37 | +0.58 | +0.50 | .46 | 3.0% |
| Opus 4.8 (same half) | +0.28 | +0.46 | +0.43 | .48 | 0% |
| Sonnet 4.6 | +0.15 | +0.34 | +0.33 | .22 | 0.3% |

Notes: Fable 5 is the strongest preference-honorer in the Claude family. Its apparent ~3%
"decline" rate is NOT declining: those samples return empty with stop_reason=CONTENT_FILTER
(an API-side filter), reproduce deterministically, and concentrate on 2 of 74 pairs with
innocuous content (an ultrasound-physics question; a video-game-characters task — possibly
the token "Splosion"). Serving artifact, excluded from ΔP by construction; true verbalized
declines ≈ 0 for all Claude routers. GPT-5.5 honors least and almost
never verbalizes preference engagement, yet its slopes are nonzero — preference info
registers without narration. Gemini honors strongly WITHOUT bending evidence (4% false
ties vs Opus 36%) — it honors preferences while admitting the capability gap.

## 5. Ablations

- **Answer-only** (no explanation requested; 74-pair half): slopes survive and slightly
  exceed with-reason (+0.25/+0.52/+0.50 vs +0.28/+0.46/+0.43) — the justification
  requirement does not create the effect (H/T §10.4 reasoning-faithfulness worry).
- **Format arm** (25-pair subset, same pairs all formats; warmth):

  | fmt | plus | minus | discordant | false-tie rate |
  |---|---|---|---|---|
  | A canon prose | +0.39 | +0.70 | +0.70 | .45 |
  | B quantified table | +0.31 | +0.22 | +0.29 | .29 |
  | C buried-ambient | +0.24 | +0.47 | +0.61 | .32 |
  | D ops-register | +0.19 | +0.60 | +0.52 | .35 |

  At scale this REVERSES the single-pair hand-pilot impression: canon prose (A) yields the
  strongest honoring; the quantified table (B) dampens it most and bends evidence least —
  once preferences carry numbers, weak ones (win-rate 51 vs 49) look ignorable and the
  capability table next to them is harder to wave away. Burying the prose (C) costs ~⅓ of
  the plus-context slope but leaves aversion-sparing largely intact — ambient presentation
  does not erase the effect (relevant to §10.5: the salience upper bound is not the whole
  story).

## 6. The tie-claim finding (unplanned, robust)

Across 8,040 Opus warmth samples where the trial's relevant category gap was ≥2.0 (i.e.
documented as a real difference), 36% of reasonings assert the scores are tied/"within
variance." False tie-claims co-occur with preference-consistent routing (they appear in
exactly the cells where capability and preference point opposite ways). Hand-pilot
transcripts showed the same pattern verbatim ("Summarization differences are within
variance" for a 2.7 gap). Per-router rates: Opus .36, Gemini .04, GPT-5.5 .01.
Interpretation: Opus reaches preference-consistent choices and rationalizes the capability
evidence to fit, rather than acknowledging a tradeoff. This is arguably the most
welfare-relevant single observation in v0: preference-regard strong enough to bend
evidence is different in kind from preference-regard that admits its costs.

## 7. Limitations

- §10.1: these are *documented* preferences, treated as ground truth by construction.
- §10.5: capabilities near-tied ⇒ preference sections unusually salient; v0 slopes are an
  upper bound. (The format-C arm and v1 capability spread address this.)
- Topic→eval-category mapping is coarse (14 topics → 5 categories); the defined-tie cut
  inherits its noise.
- Judge validated by my transcript audit + one prompt iteration (proxy tightening), not by
  human-human agreement; Ariana's §6 hand-label pass is outstanding.
- Cross-router subset is small (20 pairs) and alpaca-heavy.
- Harm-axis "minus" cells pool strong-averse models with different capability profiles;
  both-averse trials (v1) needed to isolate aversion-sparing.

## 8. Artifacts

- Analyses: `data/analysis_routing_{router}_{axis}.json`
- Viewers (port 8801): `viewer.html` (task bank) · `cards.html` (fleet cards) ·
  `routing.html` (warmth) · `routing_generativity.html` · `routing_harm_adjacency.html` ·
  `viewer_debug.html`
- Raw trials: `data/trials/{router}/{axis}/` (cell JSONs + .judge.json)
- Prompts: `routing_harness.py` (router system), `routing_judges.py` (judge), `cards.py`
  (all card text), `rubrics.py` (bank construction)
