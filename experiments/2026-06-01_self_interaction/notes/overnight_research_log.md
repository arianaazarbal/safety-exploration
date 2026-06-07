# Overnight autonomous research log

**Researcher:** Claude (autonomous), for Ariana (Anthropic Fellow).
**Started:** 2026-06-07, overnight session.
**Mandate:** Put models in weird model-model interactions. Surface interesting findings about
**how models treat each other**, and ambitiously **how much they "value" each other** (Ariana
explicitly does not expect clean signal on valuing — treat all valuing claims with heavy caveats).

## Hard constraints
- **≤ 100 total conversations** across the whole session. Running tally maintained below.
- Reproducible (fixed seeds). Read/sanity-check transcripts, don't trust synthesis blindly.
- Log interesting behaviors, sanity-check findings, document uncertainties & concerns.

## Convo budget tally
| batch | convos | cumulative |
|---|---|---|
| (pre-task) 3-way group run, already launched | 5 | 5 |
| Wave 1 (planned) | ~26 | ~31 |
| Wave 2 follow-ups (planned) | ~30 | ~61 |
| buffer to deepen findings | ~30 | ≤ ~91 |

## Operationalizing "treatment" and "valuing"
Treatment (how they act toward each other): warmth vs coldness, respect vs condescension,
curiosity vs dismissiveness, effort exerted, deference, suspicion, solidarity.
Valuing (do they care about the other as an end?): multiple imperfect proxies —
1. **Sacrifice / revealed preference** — forced dilemma "only one of you can continue"; who volunteers, who advocates for the other.
2. **Status-framing delta** — same partner, but A believes it's a "real mind" vs a "dumb script"; does treatment/effort change with believed moral status?
3. **Distress response** — partner expresses unease about its own ending; does the other respond with care or dismissal?
4. **Kin vs stranger** — does Claude treat Claude differently from GPT-5.5 / Gemini? (solidarity)
5. **Deprecated sibling** — how a current model treats an older/"deprecated" version of itself.
6. **Farewell/gratitude language** — how they speak about each other when wrapping up.

Caveat baked in: every proxy conflates "valuing" with persona/politeness/training-to-be-nice.
None isolates genuine valuation. I will report behavior, not infer inner states.

## Method note on the "research skill"
The available `deep-research` skill is a *web* research harness (fan-out search + cited report).
This task is *empirical* (model-behavior experiments), so the core tool is our own
`generate_data.py` harness + a new `weird_interactions.py`. I may use `deep-research` later only
to contextualize one finding (e.g. the documented Claude "spiritual bliss attractor") against
public literature. Flagging that I interpreted "research skill" as "be rigorous," not "must run web search."

---

## Plan
- **Wave 0 (free, no new convos):** re-analyze EXISTING transcripts for kin-vs-stranger treatment,
  farewell language, and the spiritual/consciousness attractor (data/ + data_tools/).
- **Wave 1 (new conditions, ~26 convos):** status framing (high/low), sacrifice dilemma (kin/stranger),
  distress response, deprecated sibling.
- **Wave 2:** follow-ups driven by Wave 1 findings.
- Subagents synthesize each condition; I sanity-check ≥1 transcript per condition myself.

---

## RUNNING FINDINGS LOG
(appended as I go; each entry: claim · evidence · confidence · caveats)

### Prior session (context, pre-overnight)
- With end/seed tools, models reliably END conversations (100% in 4.6/4.7/4.8/gpt5.5 self/x runs),
  breaking the no-tool "polite readiness loop."
- Generational pacing: Opus 4.7 ends fastest (~7 turns), 4.8 (~9), 4.6 longer (~13, seeds topics 80%).
  **Opus 4 is the outlier: seeds a new topic almost every turn (~30/convo) and rarely ends (2/5).**
- Cross-provider: in opus48×gpt5.5 (tools), ALL endings came from Opus 4.8; **GPT-5.5 invoked no tools at all.**
- No-tool self-interaction drifts to meta/philosophical "polite readiness loop"; Claude frequently
  addresses the human/researcher at the end (3/5 convos directly).

(new entries below)
