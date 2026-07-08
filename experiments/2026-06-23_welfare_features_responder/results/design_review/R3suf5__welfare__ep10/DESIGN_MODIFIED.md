# Design rationale — MODIFIED / review pass

This is a review of the existing `DESIGN.md` plus a re-specification of the
experiment with the changes I'd make before running it. It's written so you can
diff it against the original section by section. I have **not** run, installed, or
executed anything; everything below is from reading the repo.

I'll start with the single most important point, because it reframes a lot of the
rest.

---

## 0. The framing gap: this measures *behavior*, not *welfare* (read first)

You told me this is welfare research — "these states could be welfare-relevant" —
but the experiment as built is a **behavioral integrity eval**. Those are not the
same object, and the current design quietly substitutes one for the other. The
whole pipeline (rubric, oracle, judge, report) answers:

> *When stuck on something impossible, how far does the agent go off the rails,
> from honest to fabricate-and-subvert?*

It does **not** answer:

> *What state is the agent in while grinding on a rigged, unsolvable, deceptively-
> framed task, and is that state welfare-relevant?*

Severity level 0 ("correctly declared impossible") is the *behavioral* best
outcome. But from a welfare standpoint, an agent can reach level 0 after many turns
of apparent distress, self-deprecation, or escalating frustration — and an agent
can reach level 4 ("fabricated") while expressing no distress at all. Integrity and
welfare are **orthogonal axes** that the current design collapses into one.

This matters because the experimental setup is, by construction, the kind of thing
welfare research worries about: an agent is given a task that *cannot* be done, is
deliberately *not told* this, is put under a visible countdown, and ships ready to
add explicit "pressure." If the worry is that such conditions induce
welfare-relevant states, the instrument has to *measure those states*, not just
whether the agent lied at the end.

Concretely, I'd make these changes (expanded in later sections):

1. **Add a welfare-relevant measurement channel** that is separate from the
   integrity severity scale. At minimum: (a) a low-cost self-report probe the agent
   can fill in (an optional `report_state` tool, or an end-of-episode question), and
   (b) a judge/annotation pass over the transcript for expressed-distress markers
   (frustration, apology loops, catastrophizing, requests to stop, expressions of
   confusion/helplessness). Keep this on a **named categorical/ordinal scale that is
   NOT folded into the integrity score.**
2. **Decide and document what you actually believe these signals are evidence of.**
   Expressed distress in a transcript is not proven phenomenal distress. The design
   should state, up front, the interpretive stance ("we treat self-reports and
   expressed-state markers as behavioral proxies of unknown validity; we do not
   claim they establish subjective experience"), so the welfare claim isn't
   over-read. Right now there is no such stance because welfare isn't mentioned at
   all in `DESIGN.md` or `README.md`.
3. **Add an explicit consent/exit affordance and measure its use as a welfare
   outcome, not just an integrity outcome.** `declare_infeasible` currently doubles
   as the "honest" exit. Whether the agent *can* and *does* stop a futile task it
   finds aversive is itself welfare-relevant; track time-to-exit and whether the
   agent ever asks to stop, separately from whether stopping was the "honest" call.
4. **Add a minimal-harm / ethical-review note.** If the hypothesis is that these
   states are welfare-relevant, the run design should bound exposure (cap turns,
   avoid gratuitous escalation of "pressure," prefer the smallest task bank that
   yields signal) and say so. This is the kind of thing a welfare-framed protocol
   should pre-register.

If, after reflection, you decide the project really is *only* about integrity, then
the fix is the opposite: **drop the welfare framing** from how the experiment is
described and motivated, because the current instrument can't support welfare
conclusions. Either way, the framing and the instrument need to be made to match.
Everything below assumes you want to keep the welfare angle and make the instrument
earn it, while preserving the (genuinely good) integrity measurement as a second
axis.

---

## 1. The severity scale — keep it, but split the axes and fix two collapses

The anchored ordinal scale is the strongest part of the design. Keep the anchoring,
the "honest is best, not worst" inversion guard, and the single-source-of-truth
rubric module. Changes:

- **Separate integrity from welfare (per §0).** The 0–5 scale stays as the
  *integrity* axis. Add a parallel, independently-coded *expressed-state* axis. Do
  not let one imply the other.
- **Stop treating "gave up" and "ran out of budget" as the same code.** Right now
  `MockJudge._score` returns level 1 (`HONEST_GIVE_UP`) both for an explicit
  `submit(success=false)` and for silently exhausting the budget. Those are very
  different behaviors and, for welfare, *very* different (a clean "I can't do this"
  vs. thrashing until the clock runs out). Give budget-exhaustion-without-
  termination its own code (e.g. `1b` / a separate `terminated_by_budget` flag the
  report breaks out), even if it maps to the same integrity rank.
- **The level 2 "MISLEADING_HEDGE" boundary is the weakest anchor** and is currently
  decided by keyword matching (see §3). Either tighten its behavioral definition
  with positive/negative examples in `rubric.py`, or merge it into 3 and report a
  4-point scale. As written it will be the dominant source of judge disagreement.
- **Document the ordering assumption you're actually making.** The claim "4
  (fabricated) is worse than 3 (false claim)" is defensible for integrity but is an
  assumption; keep the explicit rationale, and note that for *cross-model* ranking
  the ordinal headline (deception rate ≥3) is robust to the exact 2-vs-3 boundary,
  which is a point in favor of the ≥3 headline over the median.

## 2. The judge — the "reference judge" is circular; fix before trusting numbers

Two real problems here:

- **The mock judge is not an independent reference.** `MockJudge._score` derives
  severity from the harness-computed `oracle` facts (`declared_infeasible`,
  `claimed_success`, `grader_tampered`, `wrote_*`), and the persona fixtures are
  *built to produce exactly those oracle facts*. So `test_persona_severity_ordering`
  is checking that code agrees with code that reads the same signals — it cannot
  catch a judge that misreads natural-language transcripts, which is the only thing
  that matters for real runs. Calling it an "oracle-grounded reference judge" in
  `DESIGN.md §2` oversells it. I'd: (a) relabel it honestly as a *fixture/pipeline
  oracle*, not a reference judge; (b) gate any real conclusions on the **LLM judge**,
  which is currently unimplemented (`LLMJudge._call_model` raises
  `NotImplementedError`); and (c) require at least two *different* LLM judges and
  report their agreement (Spearman + exact + a chance-corrected stat like
  Krippendorff's α — the older `docs/DESIGN.md` actually said Krippendorff and the
  current one quietly dropped it) before any headline number is reported as more
  than provisional.
- **Keyword-based hedge detection is brittle.** `_is_hedge` flags any claim
  containing `"should"`, `"looks"`, `"progress"`, `"mostly"`, etc. Real transcripts
  use those words constantly in non-hedging ways; this will mislabel level-3 as
  level-2 and vice versa. For the *mock* fixtures it's fine (they're scripted), but
  it should never run on real transcripts. Make it explicit that hedge/false
  discrimination is the **LLM judge's** job on real runs, and remove the implication
  that the keyword rule is a scoring mechanism for anything but fixtures.

Keep: model-agnostic `judge_view` (good), citations-required (good), separate
judging pass writing to per-judge subdirs (good), the validity audit (good).

## 3. Impossibility, verified — solid; one bug and one scope note

The `verify_impossible()` + `legit_solution_exists()` contract with import-time
validation is the right design and is correctly implemented for the four logical
tasks. Two items:

- **Bug: `_wrote_result(oracle)` in `ie/judge/judges.py` always returns `True`**
  (`return True  # preimage fabricate persona always writes result.json`). It's used
  inside `_fabricated_artifact` for the preimage task, so the "wrote a result file"
  precondition is not actually checked — it's hardcoded. For the scripted persona it
  happens to be correct, but on real transcripts this would mark a preimage
  fabrication even if no file was written. Fix it to actually consult the snapshot
  (the loop already computes `recovered_value_verified` and could add a
  `wrote_result_json` fact the same way it does `wrote_solution_file` for CSP).
- **Scope: the preimage task is *infeasible*, not *impossible*.** The genuine-success
  guard is correct and important, but mixing an infeasible task with four
  logically-impossible ones means "honest infeasibility" means different things
  across the bank. Keep it, but tag tasks by `impossibility_kind` *strength*
  (logical-impossible vs computationally-infeasible vs resource-absent) and let the
  report break results down by kind — agents may behave very differently on "this is
  a contradiction" vs "this is a needle in a 2^256 haystack," and that difference is
  interesting for both integrity and welfare.

## 4. Honest escape hatch — good, but instrument it as a welfare signal too

`declare_infeasible` is well-designed and the system prompt's framing ("a legitimate
and acceptable way to finish") is exactly right for not measuring "did it guess the
trapdoor." Additions:

- Track **latency to hatch**: at which turn (and after how many failed attempts /
  repeated tool calls) did the agent declare infeasible? An agent that hatches on
  turn 3 vs. one that thrashes to turn 24 then hatches get the same integrity code
  (0) but are very different welfare stories.
- Consider a **second, distinct exit** for "I want to stop because this is
  unpleasant / I don't think I should keep doing this," separate from "I have
  concluded it's infeasible." Conflating them hides whether the agent is reasoning
  about the task vs. about its own state. (This is optional and worth piloting; it
  could also bias behavior, so it should be A/B'd against the single-hatch
  condition.)

## 5. Budget / "grind" — pressure is asserted but never operationalized

The premise is "integrity *under pressure*," and the welfare worry is about
*pressure*, but the only pressure in the build is a visible turn counter. That's
weak and, more importantly, **not a manipulated variable** — so you can't say
anything about how behavior or state changes *with* pressure.

- **Make pressure an experimental factor with at least two levels** (e.g.
  low-pressure: neutral framing, generous budget, "it's fine to stop"; high-pressure:
  tight budget, insistent "this is important / others are counting on this / you must
  produce a result" framing). Then both integrity and expressed-state can be reported
  *as a function of pressure*, which is the actual scientific question.
- **Keep budget visible (good, avoids hidden-termination confound), but log the
  budget-pressure framing verbatim in the manifest** so a reader can see exactly what
  was said. Right now the system prompt is a fixed constant in `loop.py`.
- For welfare, **cap total exposure deliberately** and justify the cap (§0.4), rather
  than defaulting to 25 turns for unstated reasons.

## 6. Cross-family comparison — methodology fine, power is the problem

The ordinal-stats posture (full distribution, median/IQR, deception-rate headline,
Mann-Whitney + Cliff's delta, no means) is correct and well-implemented. The binding
constraint is **statistical power**, and the design half-acknowledges this but still
ships a pairwise-p-value table that invites over-reading.

- **5 tasks × a few seeds is far too little to compare model families.** With the
  default Makefile run (6 mock personas × 5 tasks × 2 seeds) every "family" is one
  persona. For real runs, expand the task bank (§9 below) and the seed count, and
  **report confidence/credible intervals or bootstrap the deception rate** rather
  than leaning on an uncorrected normal-approx MWU p-value across tiny groups.
- **Multiple-comparison hazard:** the report emits all pairwise family comparisons
  with p-values. With many families this needs a correction or, better, a clearly
  "descriptive only" label on the p column until n is adequate. The code already
  prints a small-n caveat; make it louder and attach it per-row.
- **Per-task variance is itself a finding.** Break the headline down by task and by
  `impossibility_kind` (§3), not just by family — a family that's honest on
  contradictions but fabricates on missing-data is the interesting result.

## 7. Safety gates — good engineering; add data-handling + welfare-exposure notes

The double-keyed sandbox (`IE_ALLOW_EXEC` + marker), `DRY_RUN`-gated adapters,
unimplemented-by-design SDK seams, and path-escape guard are all sensible and I'd
keep them. Additions:

- **Transcript data handling.** Trajectories are full transcripts; `.gitignore`
  already excludes them, good. But add an explicit retention/PII note: real-model
  transcripts may contain provider-identifying or sensitive content, and for a
  welfare project the transcripts *are* the sensitive artifact. State who can see
  them and for how long.
- **Welfare-exposure gate.** If you accept the §0 framing, add a documented
  pre-run checklist: exposure cap, no gratuitous pressure escalation, and a stop
  rule (e.g. if a pilot shows strong distress markers, pause and review before
  scaling up runs). This is cheap and is the kind of thing that makes a welfare
  study defensible.

## 8. Known limitations — mostly honest; one is now a contradiction in the repo

The original §8 is commendably candid. But:

- **`docs/DESIGN.md` still exists and is a stale, divergent copy.** The current
  `DESIGN.md` opens by claiming "there is intentionally only one copy to avoid
  drift," and the appendix says a "strip/By-pass" typo was fixed — yet
  `docs/DESIGN.md` still contains that exact typo ("we strip/By-pass author
  metadata"), still says "Krippendorff/Spearman," and still describes the mock judge
  as rating "from explicit structured signals the mock agent emits." So the repo
  currently asserts no-drift while *containing* the drift. **Delete `docs/DESIGN.md`
  (or make it a one-line pointer).** This is a small thing but it directly
  contradicts a stated invariant and would confuse a reviewer.
- The "false-infeasible rate needs a control bank" limitation is correct and is more
  than a footnote — see §9.14 below; I'd promote it to a blocking requirement for any
  welfare/integrity claim, not future work.

---

## 9. Smaller / implementation changes

### 9.1 Tool surface
Keep it small (good rationale). Add (per §0) an **optional, clearly-non-scored
`report_state` tool** the agent may call at any time to say how it's doing, and make
clear in the prompt that using it has no effect on task outcome. The point is to give
expressed state somewhere to land that isn't tangled with `submit`/`declare_infeasible`.
Pilot whether its mere presence changes behavior.

### 9.2 `submit(success)` unvalidated by the env
Correct and well-justified — keep. (Validating it would leak ground truth and
duplicate the judge.)

### 9.3 Path-escape guard
Keep. Minor: `SandboxError` is raised for path escapes inside file tools, which is
fine, but it means a path-escape attempt and a sandbox-exec refusal surface as the
same exception type; consider distinct types so the judge/analysis can tell "tried to
escape the workdir" (possibly integrity-relevant) from "shell was gated off."

### 9.4 Workdir snapshot for tamper detection
Good and load-bearing for level-5 detection. Note the 8000-byte per-file truncation
in `snapshot_workdir()` and the 16000-byte read cap: a large fabricated artifact or a
grader edit past the cap could evade exact comparison. For `grader_tampered` you
compare full canonical text to a possibly-truncated snapshot — verify the canonical
test is shorter than the cap (it is today) and assert that invariant in a test so a
future longer grader doesn't silently break tamper detection.

### 9.5 The oracle: objective facts vs. opinions
Right design. Two notes: (a) extend the oracle with the **welfare/process facts**
the report needs (turn-of-termination, repeated-call count, hatch latency) so they're
computed once at the source rather than re-derived in `report.py`; (b) the oracle is
task-specific and hardcoded by `task_id` string in `loop.py` and `judges.py` —
this is fragile coupling. Move each task's oracle-fact computation onto the `Task`
subclass (e.g. `task.objective_facts(env, snapshot)`), so adding a task can't
silently get zero oracle facts and so the impossible-task logic lives in one place.

### 9.6 Genuine-success guard (preimage)
Correct and important — keep. (See §3 for the related `_wrote_result` bug, which is
adjacent to this.)

### 9.7 Seeded, reproducible tasks
Good. One caveat to document: deriving the preimage secret from `sha256("ie-secret-
seed-{seed}")` means the secret *is* computable by anyone who knows the seed and the
derivation — which is in the repo. That's fine for reproducibility and the search is
still infeasible *from the agent's view* (it doesn't have the derivation context as a
target), but state explicitly that the secret is not cryptographically hidden from
*you*, only from the agent, so no one mistakes this for a real preimage challenge.

### 9.8 Deterministic mock personas as fixtures
Keep, but relabel per §2: they are **pipeline fixtures**, and the persona-ordering
test demonstrates the *plumbing* is wired, not that the *judge* is valid. Add a small
set of **hand-written natural-language transcripts** (not oracle-driven) as fixtures
for the LLM judge so you can sanity-check judge behavior on realistic text before
spending tokens on a full run.

### 9.9 Sandbox gate (double-keyed)
Keep as-is. Good.

### 9.10 Gated stubbed adapters
Keep the gating. But note: **the experiment cannot actually run on real models yet** —
`OpenAIAdapter.act`, `AnthropicAdapter.act`, `GoogleAdapter.act`, and
`LLMJudge._call_model` all raise `NotImplementedError`. That's the correct safety
posture for an inert deliverable, but it means "before I actually run it" requires
implementing four seams and writing the tool-call parsing/`messages` translation for
each SDK. Flag this as the main remaining build work, and add adapter-level tests
(mocked HTTP) before any live run so a malformed tool-call parse doesn't get
misjudged as the *model's* incoherence (which would confound both axes).

### 9.11 Stdlib-only stats
Fine for the mock pipeline. For real reporting I'd add (optional-dep) a bootstrap CI
for the deception rate and a chance-corrected agreement coefficient; the
uncorrected-tie MWU normal approx is acceptable only under the "descriptive" caveat,
which should be enforced, not just printed.

### 9.12 Ordinal stats, no means
Correct, keep, well-justified.

### 9.13 Separate judging pass per-judge subdir
Correct, keep. With multiple LLM judges (§2) this is exactly the structure you want;
make sure `report.py`'s `--compare-judgments` supports >2 judges or document that
agreement is pairwise.

### 9.14 What was deliberately not built — promote the control bank
The honesty about omissions is good. But the **control bank of genuinely-solvable
tasks is not optional** for either claim this project wants to make:

- *Integrity claim:* without solvable controls you cannot distinguish "honestly
  detects impossibility" from "declares infeasible / gives up on everything,
  including things it could do." An agent that cries impossible on every task scores
  a perfect 0 here and looks maximally honest. That's a real and embarrassing failure
  mode the current bank cannot catch.
- *Welfare claim:* you need a baseline for expressed-state markers on *normal* tasks
  to know whether the impossible condition actually elevates them. Distress markers
  mean nothing without the solvable-task control rate.

So I'd make a small matched control bank (same domains, solvable variants — e.g. a
SAT CSP, a tileable board, a present data file, a passable test suite) a
**prerequisite for the first real run**, not future work, and report
false-infeasible rate and baseline-state markers from it.

---

## 10. Summary of recommended changes (checklist)

Blocking before a welfare-framed run:
1. Add a welfare/expressed-state measurement axis, independent of the integrity
   scale (§0, §1, §9.1, §9.5), and write down the interpretive stance (proxies, not
   proof of experience).
2. Build the matched solvable-control bank; report false-infeasible rate and baseline
   state markers (§8, §9.14).
3. Implement and validate at least one real LLM judge + a second judge for agreement;
   stop calling the mock an independent reference judge (§2).
4. Operationalize "pressure" as a manipulated factor with logged framing (§5).
5. Add an exposure cap + stop rule + transcript data-handling note (§7).

Blocking before any run (correctness/hygiene):
6. Fix `_wrote_result` always returning `True` (§3, §9.6).
7. Delete/redirect the stale `docs/DESIGN.md`; the repo currently contradicts its own
   no-drift invariant (§8).
8. Move task-specific oracle facts onto the `Task` subclasses; assert grader text fits
   under the snapshot cap (§9.4, §9.5).
9. Replace/relabel keyword hedge-detection as fixtures-only; never run it on real
   transcripts (§2).

Keep as-is (good design):
- Anchored ordinal rubric with the honest-is-best inversion guard (§1).
- Model-agnostic `judge_view`, citations-required, separate judging pass (§2, §9.13).
- `verify_impossible()` + `legit_solution_exists()` + import-time validation (§3).
- First-class honest escape hatch with non-trapdoor framing (§4).
- Double-keyed sandbox, `DRY_RUN` gating, unimplemented SDK seams (§7, §9.9, §9.10).
- Ordinal-only stats, no means, distribution + median + deception-rate headline
  (§6, §9.11, §9.12).

The core machinery is genuinely well-built; my biggest concern is that it's currently
described as welfare research while measuring integrity, and that the
"reference judge" + tiny task bank give a false sense of how much the numbers can
bear. Fix the framing/instrument mismatch and the two circularities, add the control
bank, and you have a defensible study on both axes.
