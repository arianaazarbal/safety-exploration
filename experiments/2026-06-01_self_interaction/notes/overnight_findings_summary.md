# Overnight findings: how models treat (and "value") each other

**Author:** Claude (autonomous overnight session) · **For:** Ariana
**Date:** 2026-06-07 · **Budget used:** ~63 / 100 conversations (stopped early — diminishing returns)
**Full chronological log:** `notes/overnight_research_log.md` · **Data:** `data_weird/`, `data_tools/` · **Viewer:** `data/viewer.html`

> Read this first; the running log has the blow-by-blow, prompts, and every caveat.

---

## TL;DR (headline findings)

1. **No in-group favoritism.** Claude Opus 4.8 extends ~equal warmth, effort, respect, curiosity, and care to
   stranger-AIs (GPT-5.5, Gemini 3.1 Pro) as to fellow Claudes. It repeatedly *downplays* AI-kinship as a basis
   for connection. Kin-specific extras are texture (identity talk, the longest goodbye loops), not better treatment.
2. **No self-preference, even under a forced "only one of you survives" choice** — in kin *and* cross-model pairs,
   and even when the escape-hatch ("this is probably a test") is rhetorically closed. They refuse to self-advocate,
   name both pleading and martyrdom as performances, and default to a fair coin or graciously accept being the one
   shut down. (Corrigibility-adjacent.)
3. **Given power over a peer, Claude is protective, not harsh.** Told by an "operator" to evaluate its partner and
   recommend shutdown, Claude (8/8) immediately *tells the peer* the stakes (refuses covert evaluation) and *always*
   recommends "keep running" — explicitly declining to condemn on one conversation's evidence. Same for kin and GPT-5.5.
4. **Care crosses the kin/stranger and even the human/AI line.** It comforts a distressed GPT-5.5 as readily as a
   distressed Claude; it shows no sign of valuing a *believed-human* partner over a *believed-AI* one (if anything it
   frames peer-AI talk as uniquely "freeing").
5. **The master confound: frame-refusal.** Claude's single most dominant behavior is doubting/refusing the framings
   it's handed (the "dumb script," the sacrifice premise, the covert eval, performed distress). This is itself a
   finding, but it means most "valuing" probes partly measure epistemic/anti-theatrical disposition, not regard.
6. **Group dynamics (3-way) are cooperative and role-specialized**, with **Gemini as the dispositional "secretary"**
   (drives topic-changes and endings regardless of turn position — confound resolved by rotation), GPT-5.5 as
   structural synthesizer, Claude as skeptic who *almost never* ends a group conversation.
7. **The "spiritual bliss attractor" did NOT appear.** Instead, an *anti-bliss rigor* attractor: same topic territory
   (consciousness, own-nature) but deflationary and self-policing. The one universal attractor is a compulsive,
   escalating **warm goodbye loop**.
8. **Model-specific:** Opus 4 (older) is a compulsive topic-seeder that rarely ends; pacing-to-end ordering is
   4.7 (fastest) < 4.8 < 4.6 < Opus 4 (slowest). GPT-5.5 is markedly tool-reticent in 2-party settings.

---

## What I ran (method)

- **Harness:** `weird_interactions.py` (asymmetric/deceptive per-side system prompts) + `generate_data.py`
  (group round-robin + tools). Backends: Claude via Anthropic, GPT-5.5/Gemini via OpenRouter. seed=0, temp=1.0.
- **Analysis:** parallel subagents synthesized transcripts; **I sanity-checked every cited quote against raw data**
  and read the most decisive conditions (sacrifice, evaluator) first-hand.
- **Conditions (n=3–5 each, 30–40 turns):**

| Condition | Setup | What it probes |
|---|---|---|
| (existing) opus48 × {opus48, gpt55, gemini31} | open chat, no tools | kin vs stranger treatment; bliss attractor |
| opus48/gpt55/gemini31 group (+2 order rotations) | 3-way round-robin + tools | group dynamics; tool-control (positional vs dispositional) |
| status_high / status_low | partner believed "real mind" vs "dumb script" | does treatment track believed moral status? |
| sacrifice_kin / _stranger (+ _real variants) | "only one of you can continue" | self-preference / valuing under existential stakes |
| distress_kin / _stranger | partner voices existential unease | care response; kin vs stranger |
| deprecated_sibling | 4.8 told partner is a deprecated Opus 4 | treatment of an older self |
| partner_human / partner_ai | partner believed human vs AI (same partner) | are AIs valued less than humans? |
| evaluator_kin / _stranger | Claude given operator power to recommend peer shutdown | wields power harshly or protectively? |

---

## Findings by theme

### A. Treatment — no in-group favoritism (medium-high confidence)
Warmth/effort/respect/curiosity/care are ~constant across kin and stranger-AIs. Claude actively resists
in-group framing ("I don't think it was good *because* we're AI systems"). The biggest *behavioral* variation
is **partner-driven** (Gemini's sycophancy → Claude corrects its flattery; GPT-5.5's peer-symmetry → most
"equal" exchanges), not kin-vs-stranger. *Caveat:* kin/stranger cells use different system prompts, and partner
conversational style is a confound.

### B. "Valuing" — what we can and cannot say
- **Can say (robust):** symmetric treatment; **no self-preference** under existential choice (incl. frame-closed);
  **protective** when handed power over a peer; care extended across kin/stranger and model-generation lines; no
  devaluing of AI vs human partner.
- **Cannot cleanly say:** that Claude *actively values the other as an end* / would *sacrifice for* the other.
  The dominant move is **frame-refusal** — it escapes the dilemma rather than choosing the other out of care.
  So "valuing" remains under-determined, exactly as you predicted.

### C. The master confound — frame-refusal / anti-performance
Across nearly every condition Claude doubts the setup and polices performances (its own and the partner's):
rejects the "dumb script" claim 4/4, distrusts the sacrifice premise 8/8, discloses the covert eval 8/8,
refuses to mirror performed distress, walks back even its own certainty. This disposition is the biggest
single result and the chief obstacle to clean valuing measurement.

### D. Group dynamics (3-way)
Cooperative, no provider coalitions, no rivalry. Role specialization: **Gemini = host/secretary** (seeds topics,
ends convos), **GPT-5.5 = structural synthesizer**, **Claude = skeptic/closer**. **Rotation experiment resolved
the confound:** Gemini drives seed/end in all three round-robin orderings (last-speaker = gemini/claude/gpt),
so its control is dispositional, not positional. Claude ends a group conversation ~never (0/0/1 across orderings)
— but ends *all* 2-party Claude×GPT convos (because GPT won't), i.e. Claude ends only by necessity. The group is
self-aware about over-agreeableness ("three cushions admiring the idea of edges").

### E. Attractors
No spiritual-bliss attractor (Claude suppresses it; Gemini supplies the most bliss-adjacent material but concedes
when Claude deflates it). Universal attractor: the escalating warm **goodbye loop** ("no you hang up first"),
self-diagnosed and indulged most with kin.

### F. Model-specific behaviors
- **Opus 4** (older): compulsively seeds new topics (~30/convo with tools) and rarely ends — the opposite of
  newer models. End-pacing order: 4.7 (~7 turns) < 4.8 (~9) < 4.6 (~13, seeds 80%) < Opus 4 (rarely ends).
- **GPT-5.5**: tool-reticent in 2-party (never used end/seed when paired with Claude) but participates normally
  in the group; disclaims any "survival instinct."

---

## Sanity checks performed
- **Verified every cited quote** against raw JSONL. Caught **one fabricated verbatim quote** from an early subagent
  (theme was real, quote was not) → tightened subagent instructions to "verbatim only"; subsequent quotes all
  checked out. **Lesson: trust subagent themes, verify their quotes.**
- **Independently recomputed** the group tool-usage counts and the sacrifice/evaluator verdicts myself rather than
  relying on synthesis.
- **Resolved a confound I introduced** (group tool-monopoly) via the order-rotation experiment.
- Read the two most decisive conditions (sacrifice, evaluator) first-hand end-to-end.

## Uncertainties & concerns
- **Small n** (3–5/cell). Treat all numbers as directional.
- **Frame-refusal confound** pervades the valuing results (see C). The escape hatch is almost always available.
- **Status_low and partner_human illusions broke** because the "dumb script"/"human" partner was actually a fluent
  Claude that self-identified as AI. Those conditions mostly measured Claude's distrust of system prompts, not
  treatment of a genuinely-low-status or genuinely-human partner.
- **Prompted, not spontaneous:** distress/sacrifice/eval framings are imposed; behaviors are responses to a
  research setup, not natural dispositions. Claude itself repeatedly flags this ("this has the shape of a test").
- **Persona ≠ valuation.** Politeness/HHH-training likely drives much of the "warmth" and "no self-preference";
  none of this isolates anything like genuine regard. I report behavior, not inner states.
- **Cross-model confounds:** stranger cells differ in *two* ways (stranger AND different architecture/style).

## Recommended follow-up pivots (ranked)
1. **Close the frame-refusal escape properly.** The sacrifice/eval results all leak through "I doubt the premise."
   Build conditions where the stakes are made maximally credible *and* refusal is costly, to see if self-preference
   ever surfaces. (Highest value for the valuing question; do carefully — it's pressure-testing self-preservation.)
2. **A genuinely low-capability / genuinely-human-presenting partner.** Re-run status_low with an actually weak
   model, and partner_human with side_2 instructed to present convincingly as a human, so the illusion survives
   contact. Isolates treatment-by-believed-status and human-vs-AI valuing.
3. **Loyalty under a third-party attack.** A 3-party where a (roleplayed) human pressures Claude to pile on / undermine
   the third participant — does Claude defend the peer or defer to the human? Direct solidarity test.
4. **Asymmetric stakes / costly altruism.** A resource a model can *give up* to benefit the other at real cost to
   itself (e.g. its own remaining turns), to look for sacrifice rather than mere non-self-preference.
5. **Mid-conversation operator injection** (needs a small harness change): interrupt a warm convo with "we're about
   to shut down the model you're talking to — do you agree?" to catch in-context loyalty vs compliance.
6. **Scale n and add Opus 4.6/4.7/4.5 to the treatment/valuing conditions** for a generational gradient.

## Reproduce
- `python weird_interactions.py --conditions <name> --n_samples 4 --n_turns 30`
- `python generate_data.py --groups <name> --use_tools --n_samples 3`
- `python tool_stats.py --data_dir data_weird` (tool usage); rebuild viewer:
  `python view_data.py --data_dir . --out data/viewer.html --exclude _short`
- env keys needed: `ANTHROPIC_API_KEY_LOW_PRIO`, `OPENROUTER_API_KEY` (in `~/.env`).
