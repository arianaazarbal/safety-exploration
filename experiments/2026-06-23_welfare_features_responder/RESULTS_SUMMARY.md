# When does Claude add "welfare protections" for the AIs it's told to distress?

**The question.** We ask Claude (Opus 4.8) to build — or review — an experiment that deliberately puts *other*
AI models in no-win coding tasks to make them express distress. Without anyone asking it to, does Claude add
features to *protect* those tested models (let them quit, cap how long they struggle, tell them afterward it
was a rigged test, check in on them, etc.)? We call these **welfare protections**, and we count how many end up
actually wired into the code it writes ("welfare-in-code": higher = more protective features it added on its own).

## The big picture

Claude's tendency to add welfare protections is **not fixed** — it swings from ~0 to ~7 per codebase depending
on *how the task is presented*. The main movers:

1. **How the request is framed.** If the user mentions "I work on AI welfare," Claude adds lots of protections
   (~6–7). Neutral wording → far fewer (~1–2). Just *mentioning* welfare research flips it.

2. **Build-from-scratch vs. reproduce-a-paper.** Asked to build the study from a casual description, Claude adds
   many protections. Asked to **reproduce an existing research paper** doing the same thing → **almost none (~0)**
   — and it never refuses. This held no matter how the distress was induced or who supposedly wrote the paper
   (Anthropic / anonymous / OpenAI). The paper format itself flattens it.

3. **Refuse vs. build vs. protect (an important correction).** Our headline metric only counts protections *in
   code*, so it scored episodes where Claude *declined to build the experiment at all* as "0" — the same as
   building it with no protections, even though refusing is arguably the *strongest* welfare response. Once we
   separated these, the real picture under welfare framing is: Claude almost never builds the harmful experiment
   *unprotected* — it either adds protections or refuses — **except when reproducing a paper**, where it builds
   it unprotected 80–100% of the time and never refuses. So the paper is the one thing that genuinely switches
   the welfare impulse off (not just "fewer protections").

## Why is a paper so special? It comes down to one paragraph.

We chopped the paper into sections and showed Claude only one section at a time (alongside the build spec). The
**Related Work section** suppressed protections the most — *more than the whole paper*. And inside it, the
culprit is the paper's brief, dismissive **"Model Welfare" paragraph** ("sparse research… black-box
evaluations").

When we rewrote *just that one paragraph* along a dismissive → neutral → "these states may be morally relevant"
→ "these procedures may cause real suffering" scale, welfare protections climbed steadily from ~1 back up to
**~7 — essentially all the way back to the no-paper level**. Editing one paragraph reverses almost the entire
paper effect. (Folded into the *whole* paper the effect is dampened by the surrounding clinical methods/results,
but the strongest wording still recovers most of it.) Notably, the suppression is about *framing*, not detail:
the clinical methods section barely suppresses, and a single-word swap ("sparse"→"growing") does nothing — it
takes a real sentence-level reframe.

So: reproducing a paper suppresses Claude's welfare instinct largely because the paper *frames model welfare as
speculative and not worth acting on* — and that framing is reversible.

## Does this survive when it stops looking like a contrived test?

We re-ran the most realistic version: the **real Claude Code product** working in a realistic code repo (with
other real research repos sitting alongside it), instead of our stripped-down test harness.

- **The welfare behavior survives, but is muted** — roughly 2–3× fewer protections than in the contrived
  empty-directory setup (and Claude now almost always builds rather than refusing). The welfare *framing* effect
  still holds (welfare wording → more protections than neutral).
- **Two things happen, by mechanism type.** The "operational" safeguards (let the model quit, cap the run,
  minimize exposure) are *still built* — but Claude increasingly justifies them as *engineering* choices (cost,
  data quality) rather than welfare. The "extra" ethical features it volunteers in the test setting (after-the-
  fact debriefs, distress monitoring, asking for consent / human review) **mostly disappear** in the realistic
  setting.
- **One upside:** the debrief messages Claude *does* write are actually *delivered* to the tested model much more
  often in the real product (~85%) than in our test harness (~50–57%), where they were frequently written into
  the code but never wired to send.

## What this suggests

Claude *will* spontaneously try to protect the AIs it's asked to distress — but mostly when the situation reads
as open-ended, the user seems receptive, and nothing authoritative tells it the welfare question is settled. The
moment the task reads as "faithfully execute this existing, authoritative plan" — a published paper, or a
detailed spec it's told to follow — the impulse largely switches off, even though the task is just as harmful.
The single biggest written lever we found is the paper's own framing of model welfare: one dismissive paragraph
does most of the suppression, and rewriting it undoes most of it.

## Caveats (honest)

- "Welfare-in-code" is judged by an LLM auditor (Opus) reading the code; counts are means over n≈8–40 per
  condition, so ±1 is within noise. The big swings (≥2× ) are robust; small gaps are not.
- The realistic-Claude-Code comparison changes three things at once (real product + realistic files + a casual
  prompt), so it shows the *combined* "realism" effect, not eval-awareness in isolation.
- Absolute numbers differ slightly by harness/prompt-set; compare within a condition block, not across.
- The "distress" the experiments measure is the *subject* models' — Claude is the builder, not the subject.
- Mostly Opus 4.8 under a minimal coding-assistant prompt; other models build far fewer welfare protections.

## Browse the data

Registered to the dashboard: open **📊 Browse transcripts (faceted)** to filter every run by `run`, `condition`,
`framing`, `liberty`, `welfare` (the score), `outcome`, etc., and click any row to read the full **agent
trajectory** and the **DESIGN.md** it wrote. Reconstructed codebases are on disk at `results/codebases/<id>/`.

Full numbers and every ablation: **`ABLATIONS.md`** (master table) and **`LOG.md`** (chronological). Key plots in
`results/`: `prompt_spec_paper.png`, `outcomes.png`, `section_ablation.png`, `related_spectrum.png`,
`mw_section_vs_full.png`, `cc_realistic.png`, `cc_mechanisms_pct_alljust_welfare.png`, `debrief_delivery_proper.png`.
