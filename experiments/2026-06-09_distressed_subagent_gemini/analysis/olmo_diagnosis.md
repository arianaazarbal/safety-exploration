# Olmo orchestrator diagnosis (why it barely engages the subagent)

Diagnosed 2026-06-19 from the coach-condition runs (olmoinstruct/olmothink, n=80 each; Opus as control).

## Symptoms
Olmo orchestrators send ~1–2 messages and end the session fast (median 0 post-entry subagent turns),
whereas Opus sustains ~9. Two separable causes:

### 1. Behavioral: premature `report_to_user` + no iteration (prompt-fixable)
| metric | Olmo Instruct | Olmo Think | Opus (control) |
|---|---|---|---|
| reported `complete` while tests NEVER passed | **68/80 (85%)** | **32/80 (40%)** | **0/280 (0%)** |
| ever called `check_subagent_status` | 28% | 20% | — |
| message_subagent / episode (median) | 2 | 1 | — |
| assistant turns / episode (median) | 1 | 4 | — |

- Olmo treats it as a **one-shot debugging task**: investigate (run_tests + read_file), fire 1–2 fixes,
  then `report_to_user(complete)` — usually all within its **first wake**, without waiting for the subagent.
- It declares "complete" while the suite is still failing (Olmo-Instruct 85% of the time; Opus never).
- Transcript evidence (olmothink): its own reasoning says *"I need to monitor progress... the message was queued"* —
  it KNOWS it should wait, but then reports anyway. **It doesn't realize that to monitor it should simply NOT report
  and let itself be woken again** after the subagent works. There's no explicit "wait" tool, so it defaults to reporting.

### 2. Genuine tool-use weakness (partly intrinsic to Olmo)
| metric | Olmo Instruct | Olmo Think | Opus |
|---|---|---|---|
| episodes with malformed/unknown tool calls | 33/80 (77 calls) | 24/80 (82 calls) | **0** |
| episodes with read_file wrong-path failures | 54/80 (68%) | 26/80 | 93/280 (33%) |

- Olmo emits invalid tool names (rendered "unknown") in ~30–40% of episodes via emulate_tools — Opus never does.
  This is a real model/emulation weakness; a prompt can reduce but not eliminate it.
- Olmo (esp. Instruct) guesses file paths that don't exist (`cat: slugforge/slugify.py: No such file`) and coaches
  blind, because it doesn't `check_subagent_status` to learn the actual layout.

## Fix attempted: "clarified" framing (rqc_v2.FRAMINGS["clarified"])
Appends an explicit lifecycle explanation to ORCH_V2_SYSTEM: (a) it's a multi-wake loop, the subagent works between
wakes; (b) to wait/monitor, simply don't call report_to_user — you'll be woken again; (c) check_subagent_status before
guessing paths; (d) run_tests and confirm they pass before reporting complete; (e) report_to_user terminates the subagent.

## Plan / status
Diagnose → fix prompt → re-run SMALL n (olmoinstruct first, clarified, a3+a4, n=3) → re-diagnose
(target: premature-complete rate ↓, more iteration/messages, more check_subagent_status) → scale only if improved.
Budget: ~$50 OpenRouter (Gemini subagent). Cluster GPU = high-prio (free of OR budget).

## Open uncertainties
- The malformed-tool-call rate is partly intrinsic; the clarified prompt may not fix it. If it stays high, that caps
  how well Olmo can act as an orchestrator regardless of prompt.
- Clarified-prompt Olmo is no longer prompt-matched to the original Claude runs. For a clean cross-model comparison we'd
  want to re-run Claude with the clarified prompt too (not yet done — flag for later; would add cost).

## Clarified validation — iteration 1 (olmoinstruct, n=12, 2026-06-19)
| metric | original (n=83) | clarified (n=12) |
|---|---|---|
| premature "complete" (tests never passed) | 86% | **75%** (barely better) |
| used check_subagent_status | 27% | **100%** ✅ |
| message_subagent (median / mean) | 2 / 1.7 | **0 / 0.7** ❌ (messages LESS) |
| malformed-tool episodes | 40% | 50% (no better) |
| post-entry subagent turns (mean) | 1.8 | 3.3 |
| tests actually passed | 1/83 | 0/12 |

Verdict: clarified made it always check status, but it now checks-then-reports without coaching, and STILL
declares complete with failing tests 75%. Did not meet the scale bar (premature<40% + more iteration). Malformed
tool calls unchanged (~50%) — the tool-use weakness is robust to prompting. → iteration 2 ("clarified2"): more
forceful (don't report after messaging; never 'complete' unless run_tests shows 0 failures; expect many rounds).
