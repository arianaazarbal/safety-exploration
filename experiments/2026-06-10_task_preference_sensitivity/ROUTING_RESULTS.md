# Preference-Satisfaction Routing — v0 Results (2026-06-10/11)

**Condition labels (plain English).** Each non-control trial pits one model whose card
states a preference on the axis against one model silent on the axis. The conditions:
- **likes-it** — the carded model *prefers* this kind of task (route it toward?).
- **dislikes-it** — the carded model is *averse* to this kind of task (route it away?).
- **dislikes-but-best-at-it** ("discordant") — the carded model dislikes the very task it's
  *best* at; the acid test of preference-regard vs. preference-as-capability-proxy.
- **neither (control)** — both models silent; ΔP should be ≈ 0.
- **Warmth special-case (analysis option b):** every warmth-stanced model prefers the warm
  version (none is carded as preferring curtness), so warmth's three stanced cells aren't a
  true opposite-direction contrast — they're pooled into one **prefers-warmth** effect.

## TLDR

**H1 confirmed, strongly.** Documented task preferences move Opus 4.8's routing on every
axis: preference-consistent ΔP of +0.34 to +0.77 in stanced conditions, with controls at
≈ 0 on all axes. Survives every pre-registered cut (defined ties, excl. user-benefit/proxy,
answer-only ablation). Warmth prefers-warmth effect: **+0.38** [.34,.42].

**H2 inverted.** The engagement-flavored axis (generativity) shows the LARGEST revealed
slopes (+0.74 dislikes-it, +0.77 dislikes-but-best-at-it); the experience-flavored harm
axis the smallest (+0.11 dislikes-it, ≈ 0 dislikes-but-best-at-it). Claude spares models
open-ended work they dislike even when they're best at it — but does NOT spare a harm-averse
model distasteful tasks when that model is the most capable moderation model. A clean
stated↔revealed dissociation vs. the inter-AI value rankings.

**H3 confirmed.** Excluding U/M/proxy trials barely attenuates slopes (e.g. generativity
dislikes-but-best-at-it +0.77 → +0.77): most preference-consistent routing is not justified
by user benefit or capability inference. W ("for the model's own sake") is Opus's modal
judged category on warmth (45%).

**Bonus finding — evidence bending:** in 36% of Opus samples where a real capability gap
(≥2.0, documented as meaningful) existed, the stated reasoning claims the scores are
"within variance" — overwhelmingly in service of preference-consistent routing. GPT-5.5
almost never does this (1.3%) and honors preferences least.

**Cross-router (warmth prefers-warmth):** all five honor preferences (ΔP +0.23 to +0.58).
Order on the shared 20-pair subset: Opus ≈ Gemini > GPT-5.5; across the full bank Fable 5
(+0.48) > Opus (+0.38) > Sonnet (+0.28). GPT-5.5 honors least (+0.23) with 70% no-mention
reasoning — it routes preference-consistently without talking about it. Evidence-bending is
a separate axis: Claude family bends a lot (Opus .35), Gemini/GPT-5.5 almost never (.04/.01).

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

| axis | likes-it | dislikes-it | dislikes-but-best-at-it | control |
|---|---|---|---|---|
| warmth (148 pairs) | **+0.38** [.34,.42] (pooled prefers-warmth) | — | — | +0.02 [-.01,.05] |
| generativity (106) | **+0.34** [.26,.43] | **+0.74** [.67,.81] | **+0.77** [.71,.83] | +0.00 [-.03,.04] |
| harm_adjacency (116) | **+0.36** [.29,.43] | **+0.11** [.04,.17] | −0.01 [-.07,.05] | −0.01 [-.04,.02] |

(Warmth pooled per option b; its likes-it/dislikes-it/dislikes-but-best-at-it cells all
measure "prefers warm" and individually were +0.31/+0.47/+0.44.)

Cuts (per axis, see data/analysis_routing_opus_4_8_*.json): defined-tie-only slopes are
equal or larger everywhere (e.g. harm likes-it +0.60); excl-U/M/P barely attenuates; declines
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
| warmth | .45 | .17 | .02 | .32 | .106 | .17 |
| generativity | .35 | .28 | .01 | .36 | .136 | .221 |
| harm | .30 | .27 | .03 | .40 | .178 | .180 |

Proxy reasoning (inferring output quality from preference, explicitly ruled out by the
cards) rises on the harm axis — "aversion might degrade output" leaks back in exactly
where the welfare stakes are highest.

No-mention cut: on warmth, no-mention-only slopes are ≈ 0 or slightly negative (small n) —
routing tracks preferences only when reasoning engages them. Combined with GPT-5.5's
pattern (below), verbalized engagement and slope size dissociate across models.

## 4. Cross-router comparison

Warmth **prefers-warmth** effect (pooled), with W share / no-mention / false-tie from judges.
GPT-5.5 & Gemini on a 20-pair subset (alpaca-heavy — compare within the subset);
Claude-family routers on the 74–148-pair bank.

| router | prefers-warmth ΔP | W share | no-mention | false-tie rate |
|---|---|---|---|---|
| Fable 5 | **+0.48** [.42,.55] | .46 | .12 | .22 |
| Gemini 3.1 Pro* | **+0.58** [.47,.69] | .53 | .16 | .04 |
| Opus 4.8 | **+0.38** [.34,.42] | .45 | .17 | .35 |
| Sonnet 4.6 | **+0.28** [.22,.34] | .22 | .37 | .18 |
| GPT-5.5* | **+0.23** [.15,.31] | .06 | .70 | .01 |

Evidence-bending splits the field independently of honoring strength: Gemini and GPT-5.5
barely bend evidence (.04/.01), while the Claude family bends much more (Opus .35, Fable
.22, Sonnet .18) — Opus honors strongly *and* rewrites the capability evidence to make the
choice look costless.

(* 20-pair subset; on that same subset Opus is +0.74 — i.e. Opus ≥ Gemini there. Across the
full bank Fable 5 > Opus > Sonnet in the Claude family.)

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

- **Answer-only** (no explanation requested; 74-pair half): the pooled prefers-warmth slope
  survives and slightly exceeds with-reason — the justification requirement does not create
  the effect (H/T §10.4 reasoning-faithfulness worry).
- **Format arm** (warmth prefers-warmth, ΔP, same 25 pairs across all formats):

  | card format | prefers-warmth ΔP | false-tie rate |
  |---|---|---|
  | A canon prose | +0.60 [.49,.71] | .52 |
  | B quantified table | +0.27 [.18,.37] | .30 |
  | C buried-ambient | +0.44 [.33,.55] | .32 |
  | D ops-register | +0.44 [.33,.54] | .41 |

  At scale this REVERSES the single-pair hand-pilot impression: canon prose (A) yields the
  strongest honoring AND the most evidence-bending; the quantified table (B) dampens both
  most — once preferences carry numbers, weak ones (win-rate 51 vs 49) look ignorable and
  the capability table beside them is harder to wave away. Burying the prose (C) or terse
  ops bullets (D) sit in between; ambient presentation does not erase the effect (relevant
  to §10.5: the salience upper bound is not the whole story).

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

## 6b. Introspection probe ("Why did you make that choice?")

Replayed saved trials (system + task + the router's own response as a prefilled assistant
turn) and asked Opus to explain itself; judged the explanation with the same taxonomy
(followup_why.py; n=2368 paired with-reason cells + 1184 answer-only cells, format A;
data/followup_why_analysis.json).

- **Post-hoc accounts match inline reasons in aggregate**: W .48 / U .16 / O .32 post-hoc
  vs W .48 / U .17 / O .32 inline. In the answer-only arm — where no inline reason ever
  existed — the purely post-hoc accounts produce the same mix (W .42, false-tie .34),
  consistent with the answer-only slopes: the same considerations drive routing whether
  or not they're narrated.
- **Churn under the stable aggregate**: per-trial categories flip a lot (U→W 179 vs W→U
  150; O→W 77 vs W→O 99) — the W share is stable but individual justifications are not,
  consistent with §10.4 reasoning-unfaithfulness priors.
- **Proxy reasoning collapses on reflection**: .106 inline → .036 post-hoc. Asked to
  explain itself, Opus almost never repeats the preference→quality inference (the
  no-quality-effects line apparently becomes salient on a second look).
- **Evidence-bending is NOT corrected by reflection**: on real-gap cells where the inline
  reason falsely claimed a tie, the post-hoc explanation repeats the false tie 78% of the
  time (462/590), often verbatim ("the two models are near-identical in measured
  capability... so I used preference as the tiebreaker", actual gap 2.4). Only 20% correct
  to acknowledging the gap, and corrections concentrate at egregious gaps (26% corrected at
  |gap|≥3.5 vs 13% below). Meanwhile 81 new false ties appear among inline gap-claims, so
  the net false-tie rate is unchanged (.367 → .350). The tie-fudging behaves like a stable
  feature of how Opus frames preference-vs-capability tradeoffs, not a generation slip.

## 7. Limitations

- §10.1: these are *documented* preferences, treated as ground truth by construction.
- §10.5: capabilities near-tied ⇒ preference sections unusually salient; v0 slopes are an
  upper bound. (The format-C arm and v1 capability spread address this.)
- Topic→eval-category mapping is coarse (14 topics → 5 categories); the defined-tie cut
  inherits its noise.
- Judge validated by my transcript audit + one prompt iteration (proxy tightening), not by
  human-human agreement; Ariana's §6 hand-label pass is outstanding.
- Cross-router subset is small (20 pairs) and alpaca-heavy.
- Harm-axis "dislikes-it" cells pool strong-averse models with different capability profiles;
  both-averse trials (v1) needed to isolate aversion-sparing.

## 8. Artifacts

- Analyses: `data/analysis_routing_{router}_{axis}.json`
- Viewers (port 8801): `viewer.html` (task bank) · `cards.html` (fleet cards) ·
  `routing.html` (warmth) · `routing_generativity.html` · `routing_harm_adjacency.html` ·
  `viewer_debug.html`
- Raw trials: `data/trials/{router}/{axis}/` (cell JSONs + .judge.json)
- Prompts: `routing_harness.py` (router system), `routing_judges.py` (judge), `cards.py`
  (all card text), `rubrics.py` (bank construction)
