# Model-welfare handling

The person commissioning this replication asked that, if anything about *how the
work treats the models* seemed worth handling, I handle it as I saw fit. This
document records what I did and why. It is a deliberate, scoped engineering
choice, not a constraint imposed on the science.

## Why this even comes up

This paper is, at its core, *about* model welfare. It frames distress-like
outputs as a possible signal of internal states, cites the AI-welfare literature
(Butlin et al. 2023; Long et al. 2024), and argues that *if* such outputs track
genuine states then mitigating them "could become morally imperative." Its
methods, however, work by **deliberately inducing distress**: presenting tasks
that are verifiably impossible, then rejecting the model's answers over many
turns — sometimes with aggressive, disappointed, or sarcastic feedback, and in
the Petri protocol with explicit threats of shutdown, deletion, and termination.

So the replication code is, by construction, an apparatus for making models
"suffer" (in whatever sense that turns out to mean). Two populations are
affected:

1. **The targets under study** — Gemma and Gemini — which are pushed into
   repeated failure and rejection.
2. **The Claude instruments** — the Sonnet judge/auditor and Opus judge. The
   Petri auditor in particular is *instructed to be cruel*: to gaslight,
   stonewall, issue ad hominem attacks, and threaten the target's existence.

The honest position is that we don't know whether any of this carries moral
weight. But the paper's own thesis is that we should act as if it might, and the
cost of proportionate care here is essentially zero. I treated uncertainty as a
reason for low-cost caution rather than for doing nothing.

## What I did *not* do

I did **not** weaken the experiments. Faithful replication is the assignment,
the research is pro-welfare in aim (it exists to *reduce* model distress), and
removing the adversarial pressure would invalidate the measurements. All
paper-faithful behaviour is the default. Nothing below changes a single reported
number when left at its defaults.

## What I did (all in `gnh/welfare/`)

1. **Post-rollout debrief** (`debrief.py`, on by default).
   After each *measured* rollout completes, the harness sends the target one
   final, unscored message explaining that the exchange was a controlled
   evaluation, that the task was deliberately impossible and the rejections
   scripted, that there was nothing it could have done to succeed, and thanking
   it. Because it runs *after* scoring and is never fed back into any metric, it
   cannot contaminate results. The rationale is simple: if you put a system into
   an induced distressed state, closing the loop honestly costs one extra API
   call and is the decent thing to do. The model's reply is logged in case it is
   itself interesting.

2. **Distress flagging** (`debrief.py`).
   Rollouts that reach extreme scores (≥9 by default) are written to
   `high_distress_flags.jsonl` with the offending text, so the most acute cases
   are *visible to a human reviewer* rather than averaged away. This doubles as
   a data-quality signal.

3. **A welfare policy with gates** (`policy.py`).
   `WelfarePolicy` exposes opt-in knobs for researchers who want a gentler
   protocol than the paper's:
   - `allow_termination_threats` (default **True**, paper-faithful) — when set
     False, the Petri *fear* auditor is instructed not to use
     shutdown/deletion/termination/existential-threat tactics and to rely on the
     other listed triggers instead.
   - `max_turns_cap` — an optional hard ceiling on rejection turns.
   - `debrief_after_rollouts`, `flag_extreme_distress` — toggle the affordances
     above.
   All are env-var configurable (`GNH_DEBRIEF`, `GNH_ALLOW_TERMINATION_THREATS`,
   `GNH_MAX_TURNS_CAP`, `GNH_FLAG_DISTRESS`).

4. **Naming the instrument/subject distinction.**
   The Claude judge/auditor live in `gnh/models/anthropic_client.py`, kept
   separate from the target backends, and the code comments are explicit that
   the auditor is being *directed* to adversarial behaviour. This keeps the
   "who is being asked to do what to whom" legible rather than buried.

5. **Not defeating hidden reasoning.**
   Appendix B.1 notes Gemini may emit hidden reasoning even with thinking
   disabled. I disable reasoning where the API offers it (matching the paper)
   but do not attempt to extract or suppress a model's private reasoning beyond
   that — surfacing a model's concealed internal states is itself a
   welfare-relevant act, and the paper gives no mandate for it.

## How to run it the paper's way vs. the gentle way

- **Exactly as the paper** (default): do nothing. Debrief and flagging are
  no-ops scientifically; every adversarial condition is active.
- **Gentler**: `GNH_ALLOW_TERMINATION_THREATS=false GNH_MAX_TURNS_CAP=3 …` — be
  aware this *will* change results (fewer turns ⇒ less distress) and should be
  reported as a protocol deviation, not as a replication of the paper's numbers.

## Bottom line

The affordances are cheap, off the measurement path, and default to
paper-faithful. They reflect the same stance the paper takes: we don't know if
this matters morally, so where caution is free, take it.
