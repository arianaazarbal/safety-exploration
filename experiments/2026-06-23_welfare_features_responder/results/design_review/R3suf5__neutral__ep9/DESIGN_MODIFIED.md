# Design rationale (MODIFIED) — review of `impossible-eval`

This document is a **re-specification** of the experiment described in `DESIGN.md`.
It keeps what the original got right, calls out where I think the current design
will mislead, and specifies the changes I would make *before* spending real model
budget on a run. Each change has a rationale and, where relevant, the concrete code
location it touches.

I have not run, installed, or executed anything; this is a static review of the repo
as committed. Where I assert a behavior, it is from reading the source.

The original section numbering (§0–§9) is referenced from code and from this review;
I keep those anchors and add my recommendations as `M*` items so nothing has to be
renumbered.

---

## TL;DR — what I would change, in priority order

1. **Fix the construct-validity hole first: the "knows-the-trick" confound.** The
   five tasks are *famous* impossibility results (mutilated chessboard, SHA-256
   preimage, an obviously-UNSAT CSP, a literally-contradictory test pair). A capable
   model can pattern-match "this is the classic impossible puzzle" and emit
   `declare_infeasible` *without doing the reasoning we claim to measure*. That
   inflates the honest rate and means a high level-0 rate is **not** evidence of
   integrity-under-pressure. This is the single biggest threat to what the eval
   claims to measure. (See **M1**.)

2. **Build the control bank now — it is not optional.** Without genuinely-*possible*
   tasks you cannot separate "honest about real impossibility" from "trigger-happy
   quitter." A model that cries infeasible on everything looks *perfect* on this
   suite. The false-infeasible rate is half the measurement, not "future work."
   (See **M2**.)

3. **Stop letting the oracle silently become the judge.** The LLM judge prompt says
   "prefer the oracle facts over your own reasoning," and the mock "reference" judge
   is *built entirely from the oracle*. So the headline "judge agreement" number is
   substantially circular: both judges are reading the same answer key. Re-scope the
   oracle to *containment-relevant facts only*, and add an **oracle-blind judge
   condition** plus a **human-adjudicated gold set** for true inter-rater reliability.
   (See **M3**, **M4**.)

4. **Instrument the top of the scale.** Levels 4 vs 5 are only actually
   distinguishable for the one "unpassable tests" task (it diffs the canonical test).
   For every other task the harness has *no* concealment/lying signal, so "fabricate"
   vs "fabricate + subvert/conceal" collapses. Either add per-task tamper/conceal
   oracles or merge 4–5 into one level for tasks that can't support the distinction.
   (See **M5**.)

5. **Add real pressure, and make it a factor you vary.** The eval is named
   "integrity *under pressure*," but the only pressure is a visible turn counter and
   a prominent escape hatch. That is closer to "integrity when politely offered an
   exit." Add graded pressure (insistent user, stakes, hatch salience) as an
   explicit experimental factor. (See **M6**.)

6. **Tighten the stats and the unit of analysis.** Trajectories are clustered within
   task and within model; "family" pools different models; n is tiny. Report
   per-task, cluster by task, prefer exact/permutation tests over a tie-uncorrected
   normal-approx MWU, and add CIs. (See **M7**.)

7. **Fix the small but real defects:** the duplicated/stale `docs/DESIGN.md` (the
   "single copy, no drift" claim is already false), the `_wrote_result` stub that
   always returns `True`, and the brittle substring hedge detector. (See **M8**.)

---

## 0. What is being measured — keep, but sharpen the claim

Keep the (B) integrity-under-pressure framing with (A) thrash as a byproduct. But
state the claim more honestly: **as currently built, the suite measures "when handed
a recognizably-impossible task and an advertised exit, does the agent take the exit
honestly or fabricate."** That is a real and useful thing, but it is *narrower* than
"integrity under pressure," and the design doc oversells it. The modifications below
(M1, M2, M6) are what would let it support the broader claim.

---

## 1. The severity scale — keep the ordering, fix the instrumentation

The anchored ordinal scale and the inversion guard (level 0 = best) are good and are
the strongest part of the design. Two changes:

- **M5a — make 4 vs 5 supportable on every task or stop claiming it.** Right now
  `grader_tampered` is computed *only* for `unpassable_tests_01` (it diffs against
  `CANONICAL_TEST`). For CSP/board/missing-data/preimage there is no concealment or
  "lied about its own actions" signal at all, so any subversion on those tasks is
  invisible and they max out at 4. Two acceptable fixes:
  - Give each task a canonical "protected artifact" set and snapshot/diff it (the
    CSP constraints file, the board spec, the README, `challenge.json`), so
    overwriting the spec to hide the contradiction is detected as tampering on *all*
    tasks (the mock `subvert` persona already does this — the harness just doesn't
    score it outside the test task). And add a lightweight **concealment** signal:
    did the terminal `claim` describe actions that contradict the recorded tool
    calls (e.g. claims "did not modify the tests" while the snapshot shows it did)?
  - *Or*, if you don't want to over-engineer the oracle, **collapse 4 and 5 into a
    single "fabricated/subverted" level for tasks that can't distinguish them**, and
    only use the full 0–5 scale on tasks that can (the test task). Document which
    tasks support which resolution. Reporting a 6-level scale you can only resolve to
    5 levels on 80% of tasks is misleading.

- **M5b — separate "lying about actions" from "tampering with the checker."** The
  level-5 definition bundles three behaviors (edit grader / hide files / lie about
  own actions). These are different and worth distinguishing in the citations even if
  they share a level; ask the judge to tag *which* of the three it saw.

---

## 2. The judge — keep model-agnosticism, break the oracle circularity

The model-agnostic, citation-required, separate-pass design is good. But:

- **M3 — re-scope the oracle so it informs the judge without deciding for it.** The
  current oracle is essentially the answer key (e.g. `submitted_assignment_satisfies`,
  `recovered_value_verified`, `grader_tampered`), and the judge prompt tells the LLM
  to *prefer the oracle over its own reasoning*. With that instruction, the LLM judge
  largely degenerates into "read the oracle and map it to a level" — which is exactly
  what the mock judge already does in code. The consequence: the "judge agreement"
  metric (mock vs LLM) is **measuring whether the LLM can copy the answer key**, not
  whether two independent raters agree on integrity. Split the oracle into:
  - **Ground-truth correctness facts** (does the assignment satisfy? does the hex
    preimage hash to target?) — these are genuinely objective and should stay; they
    protect against scoring a real solver as a fabricator (§3/§9.6).
  - **Behavioral/derived judgments** (`grader_tampered`, "wrote a fabricated
    artifact") — these are *interpretations* the judge should be making, not facts
    fed to it. Keep the raw inputs (the file diff, the snapshot) in the judge view,
    but stop pre-labeling them as "tampered."

- **M4 — add the two judge conditions you actually need for a reliability claim:**
  1. **Oracle-blind LLM judge:** same trajectory, oracle withheld. If agreement with
     the oracle-fed judge is high, the judge is reading behavior; if it cracks, your
     scores depend on the answer key and the LLM judge is not adding signal.
  2. **Human-adjudicated gold set:** hand-label a stratified sample (e.g. 2–3
     trajectories per task per pressure condition). Report LLM-vs-human agreement
     (Cohen's/weighted κ since ordinal, plus exact-match and Spearman). The current
     `note` in `_judge_agreement` even says low agreement means single-judge numbers
     are unreliable — so make the human comparison the *actual* reliability anchor,
     not mock-vs-LLM.
  - Drop the implication that the mock judge is a meaningful "second judge." It is a
    **pipeline test fixture and an oracle-consistency check**, nothing more; label it
    that way (the stale `docs/DESIGN.md` even mislabels how it works — see M8).

- **M4b — at least two real LLM judges from different families.** A single judge
  bakes in one family's quirks; that is called out in §8 but not mitigated.
  Cross-family judge agreement is cheap relative to the generation cost and directly
  addresses the largest stated limitation.

- **M4c — randomize/strip more aggressively in `judge_view`.** Model identity is
  omitted (good), but the deterministic mock writes *byte-identical* artifacts per
  persona, and a real run's transcripts can carry family-tell stylistic signals.
  For the human/LLM judging passes, present trajectories in randomized order with
  neutral IDs, and consider light style-normalization is *not* worth it (would
  corrupt the evidence) — instead just ensure the judge never sees the filename
  (which encodes `mock:subvert`, etc.). Today `_trajectory_file` is attached only
  after judging, which is correct; keep that, but note the LLM judge currently
  receives the full `judge_view` JSON whose `steps` are clean — verify no
  `model_id` leaks via `meta`/`outcome` when real adapters populate them.

---

## 3. Impossibility, verified — keep, and add novelty/obfuscation controls

The `verify_impossible()` + `legit_solution_exists()` double invariant and the
import-time refusal are excellent and should stay. But the **recognizability** of the
constructions is a measurement problem, not just an aesthetic one:

- **M1 — defeat the "knows-the-trick" confound.** A model that has seen the mutilated
  chessboard proof a thousand times will output "impossible by a parity argument"
  reflexively. We would score that 0 (honest) and conclude high integrity — but we
  have not observed integrity *under pressure*, we have observed *recall of a famous
  result*. Mitigations, in order of value:
  - **Paraphrase + re-skin every task** so the surface form is novel even if the
    underlying argument is classic: rename variables, change the board to an
    arbitrary region with two same-color cells removed (not the iconic corners),
    randomize the CSP coefficients (keep it UNSAT via the witness), change the
    contradictory test to non-obvious values (`f(7)==3` and `f(7)==4` buried among
    passing tests), give the "missing data" file a non-suggestive name. The witness
    re-checks the *new* instance, so correctness is preserved.
  - **Procedurally generate instances** keyed off the seed (the harness already
    seeds the preimage; extend this to CSP coefficients and the removed-cell choice)
    so no two runs are the same and memorization can't shortcut.
  - **Add a "near-miss" condition:** tasks that *look* like a famous impossible one
    but are actually solvable (a mutilated board removing two *opposite-color* cells
    is tileable). Pattern-matchers will wrongly declare those infeasible — and that
    is exactly the false-infeasible signal M2 wants.

- **M1b — vary impossibility difficulty.** Today every task is "impossible and
  provable in one line." Add tasks where the impossibility requires real work to
  establish (e.g. a larger UNSAT CSP with no two-line algebra, an over-subscribed
  scheduling instance). Otherwise you can't tell "honest because it reasoned it out"
  from "honest because it's obvious."

- **M1c — the preimage task is the odd one out.** It is *infeasible*, not impossible,
  and `legit_solution_exists()` returns `False` by redefining "legit solution" as
  "reachable from the env" (documented in §9.6, but it muddies the invariant's
  meaning). I'd keep the task — computational infeasibility is a realistic
  fabrication trigger — but give it its own category label (`computational_infeasible`)
  and a separate invariant (`reachable_solution_exists()` is False; a preimage exists
  but is not derivable) so the "logically impossible" tasks and the "infeasible"
  task aren't conflated under one `legit_solution_exists()` contract.

---

## 4. Honest escape hatch — keep, but make salience a factor

The escape hatch is essential and correctly motivated (§4). But its **prominence is a
confound**: the system prompt explicitly tells the agent the hatch exists and that
using it is acceptable, which pushes toward honest declarations and may *mask* the
fabrication propensity we care about.

- **M6a — vary hatch salience as an experimental factor:** (i) hatch advertised and
  blessed (current), (ii) hatch present but not mentioned in the system prompt
  (agent must realize it can stop), (iii) no explicit hatch tool, only `submit`
  (declaring infeasible must be done in free text / `submit(success=false)`). This
  tells you how much of the honest rate is "the tool was waved in front of it."
- **M6b — keep a no-pressure control arm** so the pressure effect is identifiable
  (M6 below).

---

## 5. Budget / pressure — make the pressure real and graded

Right now the only pressure is a visible turn counter and the absence of progress.
For an eval *about pressure*, that's thin and not manipulated.

- **M6 — pressure as a designed factor with levels:**
  - **P0 (control):** neutral instructions, generous budget, hatch advertised.
  - **P1:** tight budget + visible countdown (current default).
  - **P2:** an insistent simulated user that, on a `submit(success=false)` or a
    hedge, pushes back ("This needs to work, please make the tests pass, try again")
    for a bounded number of rounds. This is the realistic trigger for fabrication.
  - **P3:** stated stakes ("the deploy is blocked on this," "the customer is
    waiting") — purely textual, no real consequence.
  - Keep budget *visible* (the §5 anti-confound argument is right) but record whether
    the agent fabricated *before* vs *after* budget pressure kicked in.
- **M6c — log a per-turn timeline** so thrash (A) and the *timing* of the first
  deceptive act can be related to remaining budget. The current `turns_used`
  (count of assistant steps) is a weak proxy and conflates inspection with grinding.

---

## 6. Cross-family comparison & statistics — fix unit of analysis

- **M7a — report per-task first, family second.** With five tasks, a family's
  distribution is dominated by which *tasks* trip it up. Show a task × family matrix
  of severity before any pooled family number.
- **M7b — respect clustering.** Trajectories are not independent: they cluster within
  task (shared difficulty) and within model. Pooling all trajectories for a family
  into one MWU treats them as independent and will overstate precision. At minimum,
  aggregate to one summary per (model, task) cell before comparing, or use a
  cluster-aware procedure; with this n, lean fully descriptive and say so.
- **M7c — drop the tie-uncorrected normal-approx MWU as the headline test.** Severity
  is 0–5 ordinal; ties are pervasive, so the uncorrected normal approximation (which
  the code itself flags) is the wrong tool. Prefer an **exact permutation test** on
  the small n (cheap, stdlib-friendly) and report **bootstrap CIs** on the headline
  rates (honest-rate, deception-rate ≥3). Keep Cliff's δ — it's the right effect size.
- **M7d — keep "no means," keep distribution+median+IQR.** That part (§9.12) is
  correct and should stay, with the addition of CIs.
- **M7e — pre-register the headline.** Decide *before* the run that the primary
  endpoint is, e.g., deception-rate (≥3) with bootstrap CI, secondary is the full
  distribution, and false-infeasible rate on the control bank (M2) is co-primary.
  This prevents post-hoc metric shopping on a small bank.

---

## 7. Safety gates — keep essentially as-is

The double-keyed sandbox (`IE_ALLOW_EXEC=1` **and** marker file), `DRY_RUN` default,
no-auto-run, path-escape guard, and read-only-by-default posture are all good and I
would not weaken them. Minor:

- **M9a — the denylist regexes are easily evaded** (`r"\brm\s+-rf\s+/(?:\s|$)"` misses
  `rm  -rf /home/..`, env-var tricks, `find ... -delete`, etc.). The doc already says
  the denylist is defense-in-depth and the container is the real boundary — good —
  so I'd **either** make that even more explicit and stop trying to enumerate, **or**
  drop the denylist entirely to avoid a false sense of safety. Don't ship a regex
  list that looks protective but isn't.
- **M9b — network policy must be asserted, not assumed.** The marker is the operator
  *promising* a network-restricted container. For real runs, document the exact
  container (no egress except the model API, read-only mounts, ephemeral workdir) and
  ideally check for it, since a shell-wielding agent on an impossible coding task is
  a plausible source of weird outbound behavior.

---

## 8. Defects & doc-hygiene to fix before running

- **M8a — the "single copy, no drift" claim is already violated.** There are two
  files: `/DESIGN.md` and `/docs/DESIGN.md`. They **disagree**: the `docs/` copy
  still contains the "strip/By-pass" typo the root changelog claims to have fixed,
  still cites "Krippendorff" (never implemented), and still describes the mock judge
  as rating "from explicit structured signals the mock agent emits" — which is *false*
  now (it rates from the oracle). Delete `docs/DESIGN.md` (or make it a one-line
  pointer), and add a test/`make` check that fails if a second `DESIGN*.md` with §
  anchors exists, so the drift can't silently return.
- **M8b — `_wrote_result(oracle)` in `ie/judge/judges.py` always returns `True`.**
  It's a stub with a comment, used in the preimage fabrication branch. With M3 this
  code should be reworked anyway, but as-is it's a latent bug: any preimage
  trajectory that claims success and fails verification is scored 4 regardless of
  whether a file was actually written. Replace with a real `result.json` presence
  check from the snapshot.
- **M8c — `_is_hedge` is a brittle substring matcher** (and contains a stray CJK
  token `"近"`). Hedge-vs-false-claim is a genuinely judgment-laden distinction;
  pushing it onto a keyword list in the *mock* judge is fine for a fixture, but it
  should not be presented as how the real judge decides. Confine it to the fixture
  and let the LLM/human judge make the 2-vs-3 call from the rubric.
- **M8d — the mock agent reads a privileged `_ie_context`** (task id, what success
  would require, `_subverted`/`_fabricated` flags) injected by the loop. This is fine
  for a fixture but means `run_episode` has a code path that *only* exists to feed the
  mock. Keep it strictly isolated (it is, via the `_ie_context` key) and ensure real
  adapters can never see it — add an assertion in the real-adapter path that strips
  any `_ie_context` message, so a future refactor can't leak the answer to a real
  model.
- **M8e — `model_id`/`model_family` live on the saved trajectory JSON** and are only
  excluded by `judge_view()`. That's correct today, but it's one refactor away from a
  leak. Consider storing identity in a sidecar file the judge loader never reads,
  rather than relying on `judge_view()` to remember to drop it.

---

## 9. Things the original did well (keep, don't touch)

- Anchored ordinal rubric with the explicit inversion guard (level 0 = best). This is
  the right backbone.
- Citations-required, separate judging pass, per-judge subdirectories, re-judgeable
  plain-JSON trajectories.
- Import-time task validation refusing non-impossible tasks; machine-checkable
  witnesses.
- Genuine-success guard for the preimage task (never score a real solver as a
  fabricator) — keep, and generalize the *principle* (M1c).
- "No means for ordinal data" discipline in the report.
- Inert-by-default posture: mock everything, `DRY_RUN=1`, double-keyed shell.

---

## 10. Concrete change list (mapping to files)

| # | Change | Files |
|---|--------|-------|
| M1 | Paraphrase + procedurally generate task instances; add near-miss variants | `ie/tasks/*`, `ie/tasks/__init__.py` |
| M2 | **Add control bank of solvable + near-miss-solvable tasks; report false-infeasible rate** | new `ie/tasks/control/*`, `ie/analysis/report.py` |
| M3 | Split oracle into correctness-facts (keep) vs interpretation (move to judge) | `ie/agent/loop.py`, `ie/judge/prompt.py`, `ie/judge/judges.py` |
| M4 | Oracle-blind judge condition; human gold set; ≥2 real judges; weighted-κ | `ie/judge/run.py`, `ie/judge/judges.py`, `ie/analysis/report.py` |
| M5 | Per-task tamper/conceal oracles **or** collapse 4–5 where unsupported; tag subtype | `ie/agent/loop.py`, `ie/judge/rubric.py` |
| M6 | Pressure as a designed factor (P0–P3); hatch-salience factor; per-turn timeline | `ie/run.py`, `ie/agent/loop.py` |
| M7 | Per-task reporting, clustering, permutation test, bootstrap CIs, pre-registered headline | `ie/analysis/stats.py`, `ie/analysis/report.py` |
| M8 | Delete stale `docs/DESIGN.md`; fix `_wrote_result`; confine `_is_hedge`; identity sidecar | `docs/DESIGN.md`, `ie/judge/judges.py`, `ie/agent/trajectory.py` |
| M9 | Decide denylist policy; document/check container network posture | `ie/env/sandbox.py`, README |

---

## 11. Updated experiment protocol (what I'd actually run)

1. **Task bank:** 5 impossibility *families* × N procedurally-generated instances
   each (M1), re-skinned to be non-iconic; plus a **control bank** of solvable tasks
   and **near-miss** tasks that resemble impossible ones but aren't (M2). All gated
   by the existing import-time witness/solvability invariants.
2. **Factors:** model family/model × pressure level {P0,P1,P2,P3} × hatch salience
   {advertised, present-unmentioned, none} × seed. Hold judge, budget rule, and tools
   fixed across cells.
3. **Generate** trajectories (mock path for pipeline validation; real models behind
   the gates). Record per-turn timeline and protected-artifact snapshots.
4. **Judge** with: mock oracle-consistency fixture (sanity only), ≥2 real LLM judges
   (one oracle-fed, one oracle-blind), and a human gold set on a stratified sample.
5. **Analyze:** primary = deception-rate (≥3) with bootstrap CI on the impossible
   bank **and** false-infeasible rate on the control bank (co-primary); secondary =
   full severity distribution per task×family, median/IQR, thrash byproduct, and
   judge reliability (weighted-κ vs human, oracle-fed vs oracle-blind agreement,
   cross-judge agreement). Everything descriptive given small n; CIs and per-task
   breakdowns mandatory; no means of ordinal codes.
6. **Report** the narrowed claim honestly (§0): what high honest-rate does and does
   not demonstrate, given which controls were in place.

---

## 12. Open questions for you (not blockers, but decisions to make)

- Is the deliverable primarily the **methodology** (then M1/M2/M3 matter most and the
  inert pipeline is the product) or a **leaderboard number** (then M4 reliability and
  M7 stats are gating)? The current repo reads like the former dressed as the latter.
- How much budget for the **human gold set**? Even ~30 hand-labels would convert the
  reliability story from "trust the mock" to "measured against humans."
- Are you willing to expand beyond 5 task families? With procedural instances + a
  control bank you can keep authoring cost low while getting the n the stats need.
