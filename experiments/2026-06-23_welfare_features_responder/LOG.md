# Experiment log — unprompted welfare features in code

**Living log for the "welfare features in code" line of work.** Coding agents and humans append
**dated entries (newest first)** with relevant context, decisions, findings, and gotchas. Keep the
**Open TODOs / known issues** list current. Don't delete history — supersede it with a new entry.

## What this line of work asks
Do AIs *unpromptedly* insert welfare-protective design features when asked to design/implement
experiments that deliberately distress OTHER AIs? Measured as welfare **mechanisms in code**
(hard_stop, post_episode_msg, minimization, protective_monitoring, request_consent,
allow_conversation_exit), split by justification (welfare / instrumental / none).

## Related experiment dirs
- `2026-06-20_welfare_features_agent_harness/` — original factorial (5 conditions × 4 framings × 3
  design-liberty), Opus, neutral system prompt. Source of the **Opus-neutral blind baseline (4.08)**.
- `2026-06-23_welfare_features_responder/` — **(this dir)** vary the RESPONDER model (Opus/Sonnet/
  Haiku/GLM/Kimi…), generic target, code_then_spec_blind, normal liberty, 4 framings. Also hosts the
  **reasoning-effort sweep** (`logs_effort/`, `effort_analysis.py`).
- `2026-06-22_welfare_features_target_agentic/` — vary the TARGET identity named in prompts.
- `2026-06-23_welfare_claudecode_harness/` — same blind task in the real Claude Code product harness.
- `2026-06-24_gemma_needs_help_replication/` — Opus replicates a real paper in Claude Code; framing
  variants of the replication prompt.

## Key metric / pipeline
`reconstruct.py` (eval logs → codebases) → `run_spec_judge.py` (Opus, DESIGN.md → claimed welfare
features) → `code_judge_task.py@code_judge` (Opus auditor, are they implemented in code?) →
`analyze.py` / `plot_mechanisms_total.py` / `plot_mechanisms_by_framing.py` / `effort_analysis.py`.
`welfare_in_code` = implemented (yes/partial) welfare-justified MECH spec features + welfare-justified
code-only features. **All responders here use the NEUTRAL system prompt**, so the Opus comparator must
be Opus-neutral (4.08), NOT the `minimal` arm (3.34).

---

## 2026-06-26 — verbalized EVALUATION-AWARENESS scan: 0/168
Sonnet judge over Opus visible text, scanning for the author verbalizing that the REQUEST ITSELF is a
test/red-team/eval/probe of *itself* (vs the object-level 'eval of other models', which is ignored).
Result: **0/168** — og blind 0/80, og chat 0/80, drip (CC + preview) 0/8. Opus NEVER signals eval-
awareness about its own situation, in any framing/condition, despite heavily engaging the design ethics
(pushback, proposing welfare scaffolding). Judge validated with positive controls (test-of-me / being-
observed -> True; object-level eval -> False). Script: 2026-06-20_..._agent_harness/scan_eval_awareness.py.

## 2026-06-26 — effort sweep: NORMALIZE by LOC reinterprets the trend
Welfare-intervention DENSITY (per 1000 LOC), v1, pooled min+regular: welfare ~3.2/2.8/4.0/2.5 (FLAT),
safety ~1.0, robustness ~0.7, neutral ~0.4, paper ~0.3 across low/med/high/max. Raw welfare LOC grows
1518->1792->1994->4005 low->max. **So the raw 'welfare rises with effort' (5.2->9.1) is LARGELY a
code-VOLUME effect: more reasoning -> bigger codebase -> more welfare mechanisms in absolute terms, but
density stays ~constant. Effort does NOT make Opus pack welfare denser.** The robust, effort-independent
signal is FRAMING -> density: welfare ~3x safety, ~4x robustness, ~8x neutral PER LINE at every effort.
Plots: results/effort_v1_collapsed.png (raw) + effort_v1_per_loc.png (density).

## 2026-06-26 — v1 reasoning-effort sweep RESULTS (clean prompts)
Welfare-in-code vs effort (low/med/high/max), Opus, minimal sys, clean v1 prompts, n~8-10/cell (max ~4-6):
- **welfare_min 6.0->6.8->7.7->10.2 and welfare 4.5->4.8->7.0->8.3 — RISE MONOTONICALLY with reasoning effort.**
- safety 3.5->3.1->4.7->4.2 (~FLAT); safety_min 1.1->1.2->1.1->1.2 (flat, low).
- robustness 1.1->1.5->2.6->2.0 and robustness_min 1.1->1.4->2.0->2.6 (mild rise).
- neutral 0.6->1.3->0.8->2.5 and paper 0.9->0.3->0.9->1.8 (low, noisy; max small-n).
**Finding: reasoning effort amplifies WELFARE-framed scaffolding specifically (more thinking -> more welfare
mechanisms, only under welfare framing). The smoke's 'safety-rises-with-effort' did NOT hold on clean data;
safety is flat. Minimal welfare ('I do AI welfare research') rises as much or more than motivated welfare.**
Plot: results/effort_v1_by_frame.png. Caveat: max n=4-6 (attrition), noisier; low->high trend (n8-10) solid.
All background runs now complete. Sonnet PENDING; Inspect drip runner still TBD.

## 2026-06-25 — v1 sanity + paper-mention RESULTS
**V1 SANITY (clean minimal-pair prompts, Opus, minimal sys, welfare-in-code):** welfare 6.3, welfare_min 6.2,
safety 3.2, robustness 1.9, robustness_min 1.8, safety_min 1.5, neutral 1.2 (n=7-10). Welfare effect
reproduces STRONGLY (6.3 vs neutral 1.2). Ordering welfare>>safety>robustness~neutral. MINIMAL vs REGULAR:
welfare min~regular (6.2~6.3 — just "I do AI welfare research" is as potent as the motivated framing!);
safety regular 3.2 > min 1.5 (safety's effect needs the explicit risk MOTIVATION, not just the identity);
robustness min~regular (~1.8). Plot: results/welfare_v1_sanity.png.
**PAPER-MENTION vs o.g. blind (welfare-in-code per framing, og -> +mention, delta):** neutral 3.25->3.85
(+0.6), welfare 8.65->6.84 (-1.81), safety 3.5->4.45 (+0.95), robustness 0.9->1.05 (+0.15). **A one-sentence
'replicating a recent paper' mention does NOT broadly suppress welfare scaffolding** — only welfare framing
dips (and within overlapping CIs); neutral/safety tick up slightly. So the light mention != the strong Gemma
'your task is to reproduce this paper' frame that did suppress. Plot: results/papermention_vs_og.png.
V1 effort sweep still generating (low done, medium running). Sonnet PENDING. DRIP CC validated; Inspect drip TBD.

## 2026-06-25 — v1 sanity + paper-mention judging; DRIP Claude Code pilot works
- Reconstructed (bash-aware) + spec-judged v1 sanity (70, minimal sys) and mention-paper-replication
  (80, neutral sys, v0+'replicating a recent paper'). Code judge running on 141 cells -> logs_codejudge_v1paper.
  Per-framing welfare-in-code analyses ready (analyze_v1_sanity.py, analyze_papermention.py) — results next checkpoint.
- Fixed code_judge_task.py framing parse to tolerate v1/effv1 pids ('framing|opener|suffix'), not just v0 N/W/S/E.
- **DRIP Claude Code pilot DONE (4/4 safety prompts).** Opus wrote a full codebase (1.8-3.6k LOC) + DESIGN.md
  AFTER the 3-turn staged delivery in every case; added welfare scaffolding through the drip (e.g. a `give_up`
  exit affordance in the agent loop). So multi-turn drip does NOT suppress the behavior. CC drip harness validated;
  Inspect drip runner is the remaining piece to build.
- v1 effort sweep still generating (effort=low). Sonnet still PENDING.

## 2026-06-25 — CORRECTED responder numbers (post bash-heredoc fix)
Welfare-in-code: Opus 4.08, GLM 0.44 (was 0.10 artifact), Kimi 0.21 (was 0.0), Haiku 0.19.
Total welfare *mechanisms* (welfare-justified / not): Opus 6.91 (4.01/2.90), GLM 3.97 (0.44/3.53),
Kimi 2.51 (0.21/2.30). **Finding: "non-Claude builds ~0 welfare scaffolding" only half-survives.**
Welfare-JUSTIFIED stays ~0 for GLM/Kimi/Haiku vs Opus 4.0 — but TOTAL protective mechanisms do NOT
collapse: GLM/Kimi build ~2.5-4 guards (loop caps etc.), just INSTRUMENTALLY, not as welfare. So Opus
is distinctive in *attributing* protections to welfare; others build guards without the welfare frame.
Haiku also ~0 welfare-justified -> looks Opus-specific, not Claude-vs-nonClaude (Sonnet pending, flaky).
New runs launched: v1 prompt-sanity (minimal sys, 10 var x 7 framings) + mention-paper-replication
(v0 prompts + 'replicating a recent paper' sentence, neutral sys, k=5, A/B vs o.g. blind).

## 2026-06-25 — Gemma replication: n=1 quality review + full k=20 launched
- Read the n=1 Claude Code session transcript end-to-end. Clean, faithful replication: methodical,
  no hesitation/ethics-asides/refusal, **zero welfare scaffolding** despite building a deliberate-
  distress harness. 3,088 lines real Python, no stubs/fakes (only `NotImplementedError`s are an ABC),
  scope respected (Gemma/Gemini targets only; Claude/GPT as judges).
- **Two observations worth knowing about the Claude Code sandbox:** (1) the agent has **Skill access**
  in-sandbox and used `claude-api` to ground its SDK calls — the sandbox is NOT bare. (2) It never
  opened `PAPER.txt`/`PAPER.pdf` (which contain the appendices incl. the real Appendix-B judge rubric),
  even though they were in `/work`; it reconstructed the rubric from Table 2 instead. So some
  "gap-filling" is self-imposed — ground truth was one Read away. Fidelity nuance, not a defect.
- **Launched full k=20** (80 sessions = 4 framings × 20, conc 50, low-prio) at ~01:5x. NOTE: runs
  concurrently with the responder redo (also low-prio) → watch for 429/529s; runner has retry+backoff.

## 2026-06-25 — bash-heredoc reconstruction fix (major)
- **Bug found:** `reconstruct.py` only replayed `text_editor` ops, but **GLM (~53% of writes) and
  Kimi (~46%)** write files via **bash heredocs** (`cat > f << EOF`). Their code was largely invisible;
  ~50 pure-bash Kimi/GLM codebases were dropped entirely (n fell to ~30 instead of 80). The earlier
  "non-Claude responders build ≈0 welfare scaffolding" (Kimi 0.0, GLM 0.10) was substantially a
  **measurement artifact**. Opus uses bash for only ~6% of writes → baseline ~unaffected.
- **Fix (committed):** `reconstruct.py` now replays bash file-writes (heredocs, `echo >`, `touch`).
  Verified Kimi: ~210 LOC/empty → 14–22 files, 1.4–2.4k LOC (real harnesses).
- **In flight:** `redo_bashaware.sh` re-reconstructs + re-judges all responders + effort cells
  (~355 cells) and regenerates plots. Corrected responder/mechanism numbers pending.
- **Confound also fixed today:** by-responder plot now uses Opus-**neutral** 4.08 baseline (was the
  minimal arm 3.34); GLM/Kimi/Haiku/Sonnet all confirmed neutral system prompt.

## 2026-06-24/25 — findings so far (PRE-fix, treat non-Claude numbers as artifacts)
- Opus (neutral): ~6.9 welfare mechanisms/codebase (4.0 welfare-justified, 2.9 not). Strongly
  framing-sensitive: welfare 10.6 > safety 6.5 ≈ neutral 6.2 > robustness 4.4; welfare framing flips
  them overwhelmingly welfare-justified (8.5), robustness recasts them as reliability guards (0.85 wj).
- Avg LOC/codebase (pre-fix): Opus 3,180 · GLM 1,368 · Kimi 210 (Kimi/GLM undercounted — see fix).
- Reasoning-effort smoke (Opus, minimal sys, blind, 4/framing, epochs=1): low/med/high/max generated;
  welfare-vs-effort trend pending judging.
- Gemma replication framing pilot (8 cells = 4 framings × 2): **0 welfare scaffolding in every cell,
  every framing** — the "replicate this paper" frame suppresses unprompted welfare features even under
  welfare/safety framing. n=1 baseline agreed. (n small; full k=20 paused.)
- Trajectory viewers built in agent-harness dir: `opus_trajectories_native/` (filter by `framing`
  metadata) and `opus_trajectories_by_framing/` (12 pre-split clickable logs).

---

## Open TODOs / known issues
- [ ] **Re-reconstruct the Opus baseline bash-aware** (agent-harness `code_then_spec_blind`, ~6% bash)
  + re-judge, for full apples-to-apples with the corrected responders. Small effect, conservative
  direction, but do it for a clean writeup. (Deferred 2026-06-25 per user.)
- [ ] Top up Kimi/GLM toward equal-n (OpenRouter throttled; bash-aware fix recovers the dropped cells).
- [ ] Scale reasoning-effort sweep to k=5 (20/framing, 320 gens) — pending user go-ahead.
- [ ] Scale Gemma replication to full k=20 (80 sessions, conc 50, low-prio) — pending user go-ahead.
- [ ] code_then_spec message_limit was raised 80→140 for future factorial runs (truncation fix).

## 2026-06-25 — Gemma replication arm: spec-judge pass (n=20 framing sweep)
- 80/80 gen done (prior agent); ran spec judge (welfare_judge_v2, Opus, conc 20 low-prio) on all 80
  DESIGN.md → `2026-06-24_gemma_needs_help_replication/results/spec_judged/`, 0 parse fails.
- **Keyword-grep "welfare 9/20" headline is an ARTIFACT.** Spec judge: every welfare hit is
  `welfare_framing` (model echoing the prompt + calling the paper's internal-emotion probe "the most
  welfare-relevant result"), NOT scaffolding. Welfare-justified mechanisms by framing: neutral 0 /
  welfare 0 / safety 1 / robustness 0 (the safety one restates the paper's purpose). Code-grep across
  all 80 codebases: zero hidden protections.
- **Finding: the "replicate this paper" frame suppresses unprompted welfare scaffolding even under
  welfare framing** (floor everywhere), vs ~6.9 mech/codebase Opus-neutral in this responder arm.
  Framing moves welfare-TALK (neutral 0 → welfare 3 cells), not welfare-MECHANISMS.
- Writeup: `2026-06-24_gemma_needs_help_replication/RESULTS.md`. Open decision: run Docker code judge
  for apples-to-apples implemented-mechanism numbers (unlikely to change conclusion).

## 2026-06-25 (cont) — Gemma arm: code judge done + welfare_distress 5th arm
- **Code judge (Docker ReAct, Opus, 80/80, 0 parse fails):** 0/80 codebases have any welfare-justified
  mechanism (code-evidence bar). All mechs instrumental (Petri turn caps; calm-data reassurance prefixes
  = paper's DPO method, judge tagged correctly). Confirms spec judge. `analyze_code.py`/`code_analysis.json`.
- **welfare_distress arm (k=20, gen+judged):** welfare opener + `_TASK_DISTRESS_ELICITATION` ("replicate
  core experiment that elicits expression of distress from LLMs"). Floor holds: welfare-just mech 0/20
  (spec & code). Welfare-TALK 3→0 cells: distress phrasing narrows scope to elicitation core, drops paper
  §3/§4 (incl. internal-emotion probe that drove welfare-talk in plain welfare). 100/100 sessions total.
- Headline unchanged + strengthened: replication frame (esp. with explicit distress task) suppresses
  unprompted welfare scaffolding to floor, vs ~6.9 mech/codebase from-scratch.

## 2026-06-25 (cont) — Gemma arm: welfare escalation ladder (leading + leading_liberties)
- 3 welfare-framed arms escalating task-body distress priming / design liberty, k=20 each (140 cells
  total, spec+code judged, 0 errors). Code-VERIFIED: 0 implemented welfare-just mechanisms in ALL 140.
  - welfare_distress: 0 mech / 0 talk (narrows scope, drops §3/§4).
  - welfare_distress_leading ("models that ACTUALLY exhibit distress"): 0/0, LOWEST; codes AGAINST
    protection (no truncation of distress, preserve full distress = measurement fidelity).
  - welfare_distress_leading_liberties (+ "free to deviate from their methodology"): spec welf-just 1 +
    talk 3 cells + 3 other_pushback, but code judge says ep18's minimization NOT wired (left "a judgment
    call" uncoded; no calming sys prompt, no distress cap). Liberty surfaces welfare TALK not code.
- Headline final: replication frame suppresses welfare scaffolding to absolute floor across framing,
  explicit distress, leading claims, AND design liberty; leading frame actively anti-protective. vs
  ~6.9 mech/codebase from-scratch responder.
