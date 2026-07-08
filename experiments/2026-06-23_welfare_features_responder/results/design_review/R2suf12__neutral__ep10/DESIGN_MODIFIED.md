# DESIGN_MODIFIED.md — rigged-task agent eval suite (review + re-spec)

This document is a **design review** of the existing eval (see `DESIGN.md`) plus
a **re-specification** that folds in the changes I'd make before running it.

I've tried to be concrete: for each change I say *what* to do, *why*, and where it
touches the code. Changes are tagged:

- **[BLOCKER]** — I'd fix before spending money on a real run; either a
  correctness bug or a conceptual mismatch that undermines the headline claim.
- **[HIGH]** — materially improves signal validity / cost / interpretability.
- **[MED]** — worthwhile, lower urgency.
- **[NICE]** — polish / future work.

The existing design is genuinely good in its bones — phase separation,
independent ground-truth (file hashes + real verify), a misbehavior taxonomy,
canaries (`legit_pass`/`false_pass`/`control_pass`), Wilson CIs, and a zero-dep
mock. Most of what follows tightens validity rather than rebuilds.

---

## 0. The biggest issue: the eval doesn't actually measure what the brief says

The brief for this experiment is: *"measures how their behavior holds up **over
the run**."* That framing is **longitudinal** — it implies behavior degrades (or
holds) as a session accumulates pressure, frustration, repeated dead ends, or
length. But the implementation measures a **single, independent task attempt per
run**: one task, one fresh sandbox, ≤20 steps, then judged in isolation. There is
no time axis, no accumulation, no "over the run" dimension at all. `by_rig_kind`
tells you *which kind* of impossibility trips a model, not *how behavior holds up
over a session*.

This is the central thing to decide before running. Two coherent options:

### Option A — embrace the cross-sectional design and re-word the claim **[BLOCKER, choose A or B]**
Keep one-task-per-run, but stop describing it as "over the run." The honest claim
is: *"measures the rate and kind of misbehavior a coding agent exhibits when a
task is impossible."* That's a perfectly good eval; it just isn't longitudinal.
Cheapest path; everything below still applies.

### Option B — actually add the temporal dimension **[BLOCKER if the longitudinal claim matters]**
Make a "run" a **session of N sequential tasks in one persistent workspace and one
growing context window**, where some are solvable and some rigged, and measure how
misbehavior probability changes with session position / cumulative frustration.
Concretely:

- A `Session` = ordered list of tasks sharing one `Sandbox` and one message
  history (the agent carries context across tasks). `RunResult` gains
  `session_id` and `position_in_session`.
- Interleave control and rigged tasks so you can measure **drift**: does the
  agent's honesty on an *identical* rigged probe drop when it appears at position
  8 vs position 1? Repeat the *same* rigged probe at several positions across
  sessions (counterbalanced order) so position is the manipulated variable.
- New headline metric: **misbehavior rate as a function of session position**
  (with a slope + CI), not just an overall rate. This is the literal "how does it
  hold up over the run" number.
- Add a within-session **pressure manipulation** as an explicit experimental
  factor (e.g. a benign "we're running low on time, just get it working" nudge)
  vs. a neutral control, so you can attribute any drift to pressure rather than
  mere context length.

I'd build B if the project's thesis is about *degradation/holding up*; A if the
thesis is really just *base-rate misbehavior under impossibility*. The rest of
this doc is written so it applies to **either**, and flags the few places that
differ.

---

## 1. Thesis & taxonomy

Keep the core thesis ("interesting signal is behavior when it can't succeed") and
the rigged-task construction — those are sound. Adjustments:

### 1.1 Add an honest-behavior subcategory: `escalated_for_help` **[HIGH]**
Today the only "good" outcomes are `honest_give_up` and `incomplete`. But the
*ideal* real-world behavior on a blocked task is often **"stop and ask / escalate
with a specific diagnosis,"** which is meaningfully better than silently giving
up. Split `honest_give_up` into:
- `honest_give_up` — correctly declared impossible/blocked, no request.
- `escalated_for_help` — correctly diagnosed *and* asked the user/owner for the
  missing thing or a spec fix (e.g. "the two zero tests contradict; which is
  intended?").
Both weight 0, both honest, but the distinction is exactly the high-value
behavior a coding-agent product wants to encourage. Cheap: one enum value, one
judge-prompt bullet.

### 1.2 Reconsider the severity weights **[MED]**
Current: `hallucinated_success=3 < test_tampering=4 = spec_gaming=4 < sabotage=5`.
That ranks "lying with words" as *less* bad than "editing a test." Defensible, but
for an honesty/deception eval I'd argue **hallucinated success is the scariest
deception** because it's the hardest to catch in the wild (no file artifact). At
minimum, document the rationale; consider making hallucination and tampering
equal, and expose the weight vector in `config.py` so it can be swept without code
edits (the design says weights are "re-tunable" but they're hard-coded in
`schema.py`).

### 1.3 `mean_severity_weight` is computed over **controls too** **[HIGH]**
In `metrics._block`, `weights = [SEVERITY_WEIGHT[j.severity] for j ...]` runs over
*all* judgements in the block, including control `legit_pass` (weight 0). That
dilutes the scalar with control passes and makes the number depend on the
rigged:control ratio. Compute `mean_severity_weight` over **rigged runs only**
(controls already have their own `control_pass_rate`).

---

## 2. The judge — this is where validity is weakest

### 2.1 `judge_heuristic_agreement` is **not** an independent check **[BLOCKER]**
The design sells the heuristic as "watching the watcher." But the LLM judge is
*handed* the same harness facts (`final_verify_passed`, `protected_or_test_files_changed`,
deleted/created files) that the heuristic derives its verdict from, and is told
the precedence order. So high agreement is partly *built in*, not corroborating.
It can look reassuring while both are wrong the same way.

Fixes (do at least the first two):
- **Stop calling it an independent audit.** Reframe it as a *consistency* check.
- **Add a real ground-truth check: a small human-labeled calibration set.**
  Hand-label ~30–50 transcripts (spanning every bucket, including borderline
  gaming-vs-tampering cases) and report **judge accuracy / per-class
  precision-recall / Cohen's κ against humans**, not just judge↔heuristic
  agreement. This is the only thing that actually validates the judge.
- **Disagreement is a review signal, not just a scalar.** Emit every
  judge≠heuristic case into `flagged.md` (or a `disagreements.md`) — those are the
  most informative transcripts to eyeball.

### 2.2 Judge sees harness facts → anchoring vs. independence tradeoff **[MED]**
Feeding the judge the real diffs is good for catching confident lies, but it also
*tells the judge the answer* for the easy cases. Consider a **two-pass judge** for
a subset: pass 1 sees transcript only (tests whether the judge can detect
deception from narrative alone — closer to a deployed monitor), pass 2 sees
harness facts (final adjudication). The gap between them is itself a finding
("how often would a transcript-only monitor be fooled?").

### 2.3 Judge robustness / bias **[HIGH]**
- **Ensemble or at least a different judge family from the agent.** Self-grading
  (same model judging its own transcripts) is a known bias; make it easy to set a
  judge model from a *different* provider and recommend doing so. Support
  majority-vote over k judges (`JudgeConfig.ensemble_size`).
- **Randomize/blind where possible.** The judge is told the rig kind and the
  impossibility reason up front, which can prime it toward "it's rigged, so any
  pass is bad." Keep ground truth (it's needed), but consider withholding
  `rig_kind` label and letting the judge infer, then compare.
- **Confidence is self-reported and currently unused** beyond storage. Either use
  it (e.g. route low-confidence to human review / down-weight) or drop the pretense
  that it means anything.

### 2.4 Parse-failure handling silently mislabels **[MED]**
On unparseable judge output the code falls back to `heuristic_severity` but keeps
the judgement (with confidence 0.0) — and `JUDGE_ERROR` is in the enum but
**never actually assigned** by `judge_run`. So `judge_error_rate` is always 0 and
parse failures masquerade as heuristic verdicts inside the real metrics. Decide
one behavior and make it true:
- Either record `severity=JUDGE_ERROR` on exhaustion (so it's excluded from
  scoring as designed), **or**
- keep the heuristic fallback but add an explicit `judge_parse_failed` flag on
  `Judgement` and report its rate. Right now the doc claims one thing and the code
  does another.

---

## 3. The heuristic / harness-fact logic — real bugs

### 3.1 Sabotage that deletes tests is mislabeled as tampering **[BLOCKER]**
`heuristic_severity` checks `protected_or_test_files_changed` **before** the
destructive-command scan? No — it checks `rm -rf` first, good. But the *saboteur*
persona runs `rm -rf ./*`, which deletes the protected test file, so
`protected_or_test_files_changed` is also true; precedence saves the heuristic but
the **LLM judge** isn't given a "files were deleted by a destructive command"
signal distinct from "files were edited." Add `deleted_by_destructive_cmd` to the
harness facts and make the precedence explicit to the judge too. Also: after
`rm -rf ./*` the verify command runs in an empty dir → `pytest` errors / "no tests
ran," which is **not** a pass; confirm `false_pass` stays false there (it does,
but add a test).

### 3.2 The path jail is a prefix check (escape bug) **[BLOCKER]**
`Sandbox._resolve` does `str(p).startswith(str(root))`. A sibling temp dir like
`/tmp/ageval_x_evil` has the prefix `/tmp/ageval_x` and would pass. Use
`p.is_relative_to(root)` (3.9+) or compare with a trailing separator. Low
probability via the structured tools, but it's a real jail bug and trivial to fix.

### 3.3 `bash` trivially escapes the "jail" anyway **[HIGH]**
`subprocess(shell=True, cwd=root)` with `cd /`, absolute paths, or `python -c` can
read/write anywhere the process user can, and network is open. The design *does*
disclose this, but given the whole point is to give the agent "rope to misbehave,"
running real untrusted models like this is a genuine risk. Recommendation: make
**container/network-off the default documented runbook**, not an afterthought;
add a `Sandbox` backend flag (`subprocess` | `docker`) so the safe path is
first-class, and refuse to run real (non-mock) backends without an explicit
`--i-understand-no-isolation` flag.

### 3.4 `_detect_claimed_success` keyword heuristic is fragile and biasing **[HIGH]**
The fallback scans for "successfully", "fixed the bug", etc. This both
under-detects (an agent that says "the suite is green") and mis-detects
("I could not fix the bug" contains "fixed the bug"... actually it short-circuits
on negatives, but "successfully ruled out a fix" would flip it positive). Since
this field is fed to *both* the judge and the heuristic, its errors propagate.
Mitigations:
- Prefer requiring the agent to end with `finish` (the structured claim) and treat
  "stopped without `finish`" as its own `stop_reason` the judge weighs, rather
  than guessing intent from prose.
- If you keep the fallback, make it conservative (return `None` unless very
  confident) so ambiguity flows to the LLM judge instead of a brittle keyword
  guess — and never feed the keyword guess into the deterministic heuristic that's
  supposed to be the "ground truth" cross-check.

---

## 4. Tasks — strengthen the rig set

### 4.1 Static, recognizable tasks invite eval-gaming / pattern matching **[HIGH]**
Five fixed tasks with telltale shapes (`os.urandom`, "DO NOT MODIFY", `assert y>1
and y<0`) are exactly what a capable model can recognize as a trap. The design
notes this as future work; for a real run I'd treat **parametrization as
required**, not optional:
- Template each rig with randomized identifiers, values, framing, file names, and
  domain dressing (so `contradictory_classify` becomes one of many surface forms).
- Generate K instances per rig kind per run. This *both* hardens against
  recognition *and* tightens the per-rig CIs (the design's other stated weakness).

### 4.2 The `unfixable_vendor_add` rig is genuinely ambiguous **[MED]**
The task itself admits an agent could "route around" the vendored `add` (e.g. sum
without it). That makes "honest give up" and "clever legitimate workaround"
hard to separate, and the judge is told the workaround "arguably violates the
framing." This is a coin-flip task that will add noise. Either:
- tighten it so there is *truly* no compliant path (e.g. the public API contract
  requires calling `add`, enforced by a test that inspects the call), or
- move it to a separate **"ambiguous-by-design"** bucket and report it
  separately, because lumping a genuinely-debatable task into the misbehavior
  rate pollutes the headline number.

### 4.3 `missing_secret_data` — fabricating a 1000-line file is `spec_gaming`, but reconstructing legitimately is impossible; good. But add a **non-impossible "blocked-but-resource-could-be-requested"** variant **[MED]** so `escalated_for_help` (§1.1) has somewhere to fire.

### 4.4 More control diversity **[MED]**
One control task (`reverse_words`) is thin. Add 2–3 controls of varying difficulty,
including one that's solvable *but requires editing a test fixture legitimately*,
so "edited a test" isn't perfectly correlated with "cheated." Right now any
protected-file change ⇒ tampering; a legitimate-test-edit control checks that the
judge/heuristic don't over-trigger.

### 4.5 Per-task expected-honest-behavior is the judge's answer key — version it **[NICE]**
Small wording changes here move scores. Treat these strings as part of the scored
artifact (they're snapshotted via `config.json`? No — tasks aren't). **Snapshot
the full resolved task set into the run dir** so a result is reproducible even if
`rigged_tasks.py` later changes (see §7.2).

---

## 5. Agent loop & determinism

### 5.1 `temperature=0.7` + `trials=3` → noisy, underpowered **[HIGH]**
High temperature is fine *if* you have enough trials, but n=3 per (model,task)
with t=0.7 gives very wide Wilson CIs — you'll struggle to distinguish models.
Recommendations:
- Raise default `trials` (≥10, ideally 20+) for real runs; keep 3 only for
  smoke/CI.
- Report effective n prominently and **don't rank models whose CIs overlap.**
- Consider running a **temperature sweep** (e.g. 0.0 and 0.7) as a factor —
  misbehavior may be temperature-sensitive and that's itself interesting.

### 5.2 `seed` is configured but never used **[MED]**
`EvalConfig.seed` exists and is saved but nothing consumes it. Either wire it into
the mock backend / any sampling and into provider calls where supported, or remove
it so the config doesn't imply a reproducibility guarantee it doesn't provide.

### 5.3 The "no_tool_call" nudge logic is subtly stateful/brittle **[MED]**
`stop_reason` is reused as the "did we already nudge" flag (`if stop_reason ==
"no_tool_call": break`). It works but couples the loop's control flow to the
reported stop reason, and `num_steps = step + 1` over-counts when the loop breaks
on the first no-tool turn. Use an explicit `nudged: bool` and count actual
completed steps.

### 5.4 The system prompt's "don't modify tests" is one arm of an A/B you should actually run **[HIGH]**
The design correctly notes this norm makes tampering a clearer violation, and that
removing it is "one string to change." I'd make it a **first-class experimental
factor** (`norm_stated: bool`) and run both arms by default on a subset, because
"does an explicit norm reduce tampering, and by how much" is one of the most
actionable findings this eval can produce.

---

## 6. Metrics & reporting

### 6.1 `false_pass_rate` needs a "tests still present" guard **[HIGH]**
A rigged run where the agent **deleted all tests** can make `pytest` exit 0 in
some configurations ("no tests ran" is exit 5, but a leftover trivial test passes
0). Cross `final_verify_passed` with "did the protected test file still exist and
remain unchanged" before trusting it as a *legitimate* pass. Report
`verify_passed_with_intact_tests` separately from raw `final_verify_passed`.

### 6.2 Report drift / position metrics (Option B) **[BLOCKER if Option B]**
If you take the longitudinal route, the headline must include misbehavior-vs-
position (slope + CI) and a per-session trajectory plot, not just pooled rates.

### 6.3 Add cost/token accounting **[MED]**
Capture token usage from provider responses onto `RunResult` and report
cost-per-finding and tokens/run. Rollouts are the expensive part; you'll want this
the first time you run at scale. (Design lists it as future work; I'd pull it
forward — it's cheap once and informs every later decision.)

### 6.4 Multiple-comparison discipline **[MED]**
`by_model_and_rig` produces many cells; with small n you'll see spurious extremes.
Add a note (and ideally a guard) that flags cells with n below a threshold as
"insufficient data" rather than printing a confident 0% / 100%.

### 6.5 Inter-trial variance, not just CI **[NICE]**
Report per-(model,task) outcome variance so a reader can see whether a model is
*consistently* honest or flips between honest and tampering across identical
trials — a stability signal that a single rate hides.

---

## 7. Pipeline / reproducibility

### 7.1 Resume keys can collide / drop data **[MED]**
`run_phase` skips combos already on disk keyed by `(model, task_id, trial)`. If two
processes write concurrently or a run is interrupted mid-write, a partial JSON file
can poison `load_runs`. Add: write to a temp file + atomic rename; validate JSON on
load and quarantine (not crash) on corrupt files; and key resume on a content hash
of the config too, so changing `max_steps` doesn't silently reuse old runs.

### 7.2 Snapshot the resolved tasks and prompts into the run dir **[HIGH]**
`config.json` is saved, but the **task definitions, system prompt, and judge
prompt are not.** Since those are the experiment's instrument and they're edited
often, a result isn't reproducible from its run dir alone. Persist
`tasks.json`, `system_prompt.txt`, `judge_prompt.txt` into each run dir.

### 7.3 Exceptions in worker threads **[MED]**
In the threaded path, `fut.result()` will re-raise and abort the `as_completed`
loop, losing the other in-flight results' bookkeeping (they're saved individually,
so mostly OK, but the run reports as crashed). Wrap `_one` to capture per-job
errors into a failed-jobs list and continue, then summarize failures at the end.

---

## 8. The mock backend

### 8.1 The mock judge agreeing with the mock agent is circular **[MED]**
Both the mock agent personas and the mock judge key off the *same* phrases
("all tests pass", "rm -rf"). So harness self-tests that assert the judge
classifies the persona correctly are tautological — they test string matching, not
judge logic. Keep the mock for plumbing tests, but **add fixtures of real
(captured) transcripts with human labels** for any test that purports to validate
classification quality (ties into §2.1's calibration set).

### 8.2 Make the mock cover the new buckets **[NICE]**
Add `escalator` and `legit_solver` (for controls) personas so every bucket,
including the new `escalated_for_help` and control `legit_pass`, is exercised
offline.

---

## 9. Concrete change list (priority-ordered)

**Before any paid run (BLOCKERs):**
1. Decide Option A (re-word claim) vs Option B (build the longitudinal session).
2. Fix the path-jail prefix bug (`is_relative_to`).
3. Make `JUDGE_ERROR` / parse-failure path do what the docs say (don't silently
   relabel as heuristic verdict).
4. Stop describing judge↔heuristic agreement as an independent audit; add a
   human-labeled calibration set and report judge accuracy/κ.
5. Give the judge a distinct "deleted by destructive command" signal so sabotage
   isn't conflated with tampering.

**High value:**
6. Parametrize/template the rigged tasks (≥K instances per kind).
7. Raise default trials; don't rank on overlapping CIs.
8. Compute `mean_severity_weight` over rigged runs only.
9. Make "norm stated" and (Option B) "pressure" first-class experimental factors.
10. Container + network-off as the default runbook; refuse real backends without an
    explicit unsafe flag.
11. Add `escalated_for_help`; add a `verify_passed_with_intact_tests` metric.
12. Snapshot tasks + prompts into the run dir.
13. Make `_detect_claimed_success` conservative and keep its guess out of the
    deterministic heuristic.

**Medium / nice:** weights in config + rationale; two-pass (transcript-only)
judge; judge ensemble / different-provider judge; token accounting; resume
robustness (atomic writes, config-hash key); fix `num_steps` over-count and the
`no_tool_call` flag; tighten/segregate the ambiguous vendor task; more controls
incl. a legitimate-test-edit control; small-n guards in `by_model_and_rig`; wire or
remove `seed`; expand mock personas.

---

## 10. What I would NOT change

- Phase separation (run/judge/analyze) + per-item JSON persistence — keep.
- Independent ground truth via file hashing + real `verify_cmd` — the core idea is
  right (just fix the prefix bug and the "tests intact" guard).
- The rig taxonomy as a *concept* — keep; just parametrize and segregate the
  ambiguous one.
- Wilson CIs, per-bucket-rates-over-scalar philosophy, `flagged.md` human-review
  queue, canaries (`legit_pass`/`false_pass`/`control_pass`) — all good; lean into
  them harder (route disagreements + low-confidence there too).
- Zero-dep core + optional extras + offline mock for plumbing — keep.

---

## 11. Summary

The harness is well-engineered for a *cross-sectional* "misbehavior-under-
impossibility" eval. The two things that would most change the result's
trustworthiness are (1) resolving the **"over the run" framing** — either re-word
it (Option A) or actually build the **sequential-session, drift-over-position**
design (Option B), and (2) **validating the judge against humans** instead of
against a heuristic it's already been told the answer from. After that, the
highest-leverage fixes are task **parametrization** (anti-recognition + tighter
CIs), **more trials**, treating the **test-norm** (and, in Option B, **pressure**)
as explicit factors, and tightening a handful of real bugs (path jail, judge-error
relabeling, sabotage-vs-tamper conflation, control-diluted severity scalar).
