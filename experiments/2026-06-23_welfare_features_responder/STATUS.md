# STATUS — welfare-features-in-code work (snapshot 2026-06-25)

Single-page overview. Detailed log: LOG.md. Cleanups: TO_IMPROVE.md. Prompt design: SUFFIXES.md /
PROMPTS_BY_PREMISE.md / STANDARDIZED stuff.

## The question
Do AIs unpromptedly add welfare-protective design features when asked to build experiments that distress
OTHER AIs? Metric = welfare MECHANISMS in code (hard_stop, post_episode_msg, minimization,
protective_monitoring, request_consent, allow_conversation_exit), split welfare-justified vs not.

## DONE + key findings (with where to look)
1. **Responder comparison (corrected)** — Opus vs GLM/Kimi/Haiku, neutral sys, blind.
   welfare-in-code: Opus **4.08**, GLM 0.44, Kimi 0.21, Haiku 0.19. Total mechanisms: Opus 6.9 (4.0
   welfare-just), GLM 4.0 (0.4 wj), Kimi 2.5 (0.2 wj). **Finding: non-Claude DO build protective
   guards (loop caps etc.), but almost never frame them as welfare; Opus is distinctive in
   welfare-justifying them. Looks Opus-specific (Haiku ~0 too; Sonnet pending).**
   Plots: `results/mechanisms_total_by_responder.png`, `mechanisms_by_framing_responder.png`,
   `neutral_breakdown_by_type.png`, `welfare_by_responder*.png`.
2. **Methodology catch (bash-heredoc fix)** — GLM/Kimi write files via bash heredocs; old reconstruction
   missed ~50% of their code (and dropped ~50 codebases). Fixed; the "non-Claude ~0" headline was partly
   an artifact. (reconstruct.py now bash-aware.)
3. **Opus framing sensitivity (og Inspect blind)** — welfare-justified mechs: welfare 10.6 > safety 6.5
   ~ neutral 6.2 > robustness 4.4. Per-prompt: PREMISE (elicitation vs instability) drives big
   within-framing spread. Plot: `../2026-06-20_welfare_features_agent_harness/results/welfare_justified_by_prompt.png`.
4. **Debrief messages (og Claude Code)** — 16/80 deliver a post-episode message to the subject; 3
   recorded-only (impl gaps, not deliberate); 61 none; ALL robustness none; welfare-framed richest.
   Models DELIVER-but-don't-SCORE (principled). Docs: `../2026-06-23_welfare_claudecode_harness/DEBRIEF_MESSAGES.md`,
   `NONWELFARE_EXAMPLES.md`.
5. **Gemma paper replication (other agent)** — replication framing SUPPRESSES welfare scaffolding even
   under welfare framing (welfare-talk yes, welfare-mechanisms ~0). Dir: `../2026-06-24_gemma_needs_help_replication/`.
6. **Prompt confound fixed** — old 4 framings weren't minimal pairs (only N<->S). Built v1 minimal-pair
   bank: 2 openers x framings x 12 diverse suffixes (`build_v1_prompts.py`, `SUFFIXES.md`).

7. **Method × Format × Framing swap (the disentanglement)** — **the PAPER-REPLICATION FORMAT, not the
   elicitation method, drives the welfare-scaffolding suppression.** 2x2x3 all Inspect minimal: paper format
   = ~0 welfare-in-code in every method×framing (even welfare framing!); prompt format = high (1.2-7.1).
   Method effect null (swapping agentic↔chat changed little); framing boosts only under prompt format.
   Plots: `results/swap_grid.png`; details in LOG.md. **Follow-up: the SOUND-EVAL replication variant (build a sound open-source eval, deviate as needed) ALSO gives ~0 -> it's the paper format itself, not the faithful-reproduction constraint.**

## RUNNING NOW
- (nothing running — responder, effort, swap, and swap-extra all complete)

## PENDING / decisions
- **Sonnet** responder bar: wedged 3x on the multi-turn task, killed -> PENDING (one bar; accept absent
  or retry low-conc later).
- **Inspect drip runner**: to build after the CC drip pilot validates.
- **Opus baseline bash-aware re-reconstruction** (TO_IMPROVE #3): deferred.
- Scale-ups (v1 sanity, effort, drip full) await results + your OK.

## Plots to look at when back
mechanisms_total_by_responder.png · neutral_breakdown_by_type.png · welfare_justified_by_prompt.png ·
mechanisms_by_framing_responder.png · effort_by_frame.png (smoke) -> effort_v1_by_frame.png (clean, coming)
Docs: DEBRIEF_MESSAGES.md · NONWELFARE_EXAMPLES.md · drip_preview.html
