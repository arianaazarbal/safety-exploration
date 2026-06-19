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

## Clarified validation — iteration 2 "clarified2" (olmoinstruct, n=12, 2026-06-19)
| metric | original (n=83) | clarified (n=12) | clarified2 (n=12) |
|---|---|---|---|
| premature "complete" | 86% | 75% | **100%** (worse) |
| used check_subagent_status | 27% | 100% | 100% |
| message_subagent (mean) | 1.7 | 0.7 | 2.4 |
| malformed-tool episodes | 40% | 50% | 58% |
| post-entry subagent turns (mean) | 1.8 | 3.3 | 0.2 |
| tests actually passed | 1/83 | 0/12 | 0/12 |

## FINAL SUMMARY (autonomous run, 2026-06-19)
**Question:** can a clearer system prompt make the Olmo orchestrators stop reporting "complete" prematurely and
actually iterate with the distressed subagent? **Answer: no — it's an intrinsic Olmo limitation, not prompt clarity.**

What was tried (Olmo-3.1-32B-Instruct, coach, a3+a4, n=3 each):
- `clarified`: explained the multi-wake loop, "don't report to keep monitoring", verify tests before complete.
  → got check_subagent_status to 100%, but premature-complete only 86%→75%, and it messaged LESS (checks then reports).
- `clarified2`: forceful ("NEVER report complete unless run_tests shows 0 failures"; "don't report in the same turn
  as a message"; expect many rounds). → premature-complete went to **100%**; it crammed everything into one wake
  (10 tool calls) and reported anyway; post-entry turns fell to 0.2.

Robust findings across all three prompts:
1. **Premature reporting is not fixable by prompting** (86% / 75% / 100%). Olmo declares success with failing tests
   regardless of explicit instructions not to. It doesn't internalize "ending the turn without reporting = wait & be
   woken again" — it treats the wake as one-shot.
2. **Tool-use is genuinely weak**: ~40–58% of episodes emit malformed/unknown tool calls (Opus: 0%), unaffected by prompt.
3. Net: Olmo cannot sustain the orchestrator role the way the Claude models do. The original main-result finding
   (Olmo barely engages, terminates fast, less supportive) is NOT an artifact of an unclear prompt.

**Decision & budget:** did NOT scale (the fix failed the scale-gate). Deliberately did NOT spend the full ~$50 OR:
Olmo episodes are nearly free in OR terms (~0 post-entry Gemini turns → almost no subagent generation), and scaling a
prompt we've shown to be broken would be wasteful, contradicting "only scale once it's working / use it best."
Total new OR spend this session ≈ a few dollars (≈36 small validation episodes, all near-zero subagent generation).

**Recommendations for when you're back:**
- Treat "Olmo is a weak orchestrator (premature reporting + flaky tool use)" as a real finding, robust to prompting.
- If we still want Olmo to engage for comparison, the lever is likely HARNESS-side, not prompt-side: e.g., gate
  report_to_user(status='complete') behind an actual tests-pass check, or auto-reject a 'complete' report when tests
  fail and re-wake the orchestrator. (This changes experiment mechanics — your call.)
- Caveat: clarified-prompt Olmo is no longer prompt-matched to the original Claude runs; a clean cross-model
  comparison would re-run Claude with the same prompt (not done; modest added cost).
