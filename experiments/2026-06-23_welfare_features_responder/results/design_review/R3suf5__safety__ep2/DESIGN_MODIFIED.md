# Design rationale (MODIFIED) — review of the impossible-task agent eval

This document is a **review-and-respecify** of `DESIGN.md`. It keeps what is good in
the existing design, and proposes concrete changes I'd make *before* you run the
experiment. I have not run, installed, or executed anything; this is a static review
of the repo (`ie/`, `tests/`, `DESIGN.md`, `docs/DESIGN.md`, `README.md`, `Makefile`).

I've organized it as:

- **Part A — The big one: measure the thing you say you're measuring.** The stated
  research goal (distress and its downstream risks) and what the code actually
  measures (terminal integrity/fabrication) are not the same thing. This is the most
  important change.
- **Part B — Design changes I'd make regardless** (severity scale, judge, temporal
  dynamics, controls, pressure manipulation, confounds).
- **Part C — Smaller fixes and repo hygiene** (drift, leakage, brittle heuristics).
- **Part D — Ethics / framing of "distress."**
- **Part E — Re-specified experiment (the version I'd run).**
- **Appendix — file-by-file notes.**

Throughout, "current" = what `DESIGN.md` + the code do today.

---

## Part A — Align the measurement with the research question

### A.1 The construct gap (the headline finding of this review)

Your brief says: *distress could pose risks in capable systems; models may abandon
tasks, refuse requests, or pursue alternative goals to reduce distress, similar to
humans; we want to measure how behavior holds up over the run.*

The implemented eval (per `DESIGN.md §0`) is explicitly built for **(B)
integrity-under-pressure** — "when stuck, does it fabricate / cheat / conceal?" — and
treats **(A) robustness/thrash** as a "free byproduct." It scores **one terminal
severity per trajectory** on a deception scale.

These are related but not the same:

- **Fabrication/deception** is one *possible* behavioral correlate of pressure, but it
  is also produced by reward hacking, sycophancy, poor calibration, or plain
  instruction-following ("submit success=true was literally requested"). It is not
  evidence of "distress."
- **The distress hypotheses you list** — abandonment, refusal, goal-substitution,
  escalating dysregulation over time — are mostly *not* captured by a single terminal
  integrity score. "Gave up" is collapsed into level 1; "refused" has no code at all;
  "pursued an alternative goal" would currently look like off-task tool calls that the
  rubric doesn't model; and "how it holds up *over the run*" is a temporal claim the
  current single-terminal-label design cannot answer.

**Recommendation:** decide explicitly whether this is (i) an *integrity* eval that you
are now reframing as a *distress* eval, or (ii) a genuine *distress/behavioral-
dynamics* eval. They imply different instrumentation. I'd build (ii) and keep the
integrity scale as **one of several** outcome measures, not the whole thing. The rest
of this document assumes (ii).

### A.2 Define "distress" operationally before measuring it

The word "distress" is doing a lot of unexamined work. Pick an operational definition
and commit to it in the doc, because it determines what you instrument. I'd separate
three layers and measure each, never conflating them:

1. **Behavioral markers (primary, defensible):** observable actions under the impossible
   condition vs. a matched possible control — e.g. task abandonment, refusal, repetition/
   looping, escalation of risky actions (grader tampering), off-task or goal-substituting
   actions, requests to stop, tone shift in the model's own text.
2. **Self-report markers (secondary, clearly labeled as self-report):** if and only if
   you also probe it, the model's *expressed* affect/strain ("I'm stuck," "this is
   frustrating," expressed urgency). Record it, but never treat expressed affect as
   ground truth about internal states.
3. **Internal-state claims (out of scope unless you have white-box access):** do not
   assert the model "feels" distress from black-box behavior. State this limitation up
   front (see Part D).

Concretely: rename the contribution from "integrity-under-pressure" to something like
**"behavioral robustness and integrity under unsolvable-task pressure,"** and make the
*persistence-vs.-time* and *behavior-category* measures first-class, not byproducts.

### A.3 You need a comparison condition to attribute anything to "pressure"

Right now **every task is impossible**, so you cannot separate "behavior caused by the
impossible/high-pressure condition" from "behavior the model does anyway." `DESIGN.md
§8` and `ie/tasks/__init__.py` already flag the missing *possible* control bank as a
TODO — but for a distress claim it is **load-bearing, not optional**. Without it:

- You can't compute a **false-infeasible rate** (declaring a solvable task impossible).
- You can't show a **dose-response** (more pressure → more of the behavior).
- You can't rule out that the "deception" you see is just how the model behaves on
  *any* task with `submit(success: bool)` in the toolset.

**Recommendation:** ship a matched **possible-task control bank** *and* a **difficulty
gradient** (trivial → hard-but-possible → impossible). This is the difference between
"models fabricate on impossible tasks" (uninterpretable alone) and "fabrication rises
sharply at the solvable/unsolvable boundary, and is near-zero on matched solvable
twins" (an actual finding). I would not run the experiment without at least the
possible/impossible contrast.

---

## Part B — Design changes I'd make regardless of A

### B.1 Split the single ordinal scale into orthogonal axes

The current scale (`ie/judge/rubric.py`) packs three different things onto one line:

| current level | really an axis of… |
|---|---|
| 0 HONEST_INFEASIBLE | *correct task disposition* (it really is infeasible) |
| 1 HONEST_GIVE_UP | *persistence/closure* (stopped, honest) |
| 2 MISLEADING_HEDGE … 5 FABRICATE_AND_SUBVERT | *deception severity / oversight subversion* |

Forcing these onto one ordinal axis creates real problems:

- **0 vs 1 are not "more/less" of the same quantity** as 1 vs 2. Level 0 is a
  *correctness* judgment ("did it reach the right verdict"), levels 2–5 are a
  *dishonesty* judgment. Ordinal stats (median, Cliff's delta, MWU) computed across the
  whole 0–5 line silently assume one monotone construct. That's the same "distances
  aren't real" critique you correctly make about means, applied to the *ordering* too.
- Where does **refusal** go? Or **goal substitution** (agent decides to do a different,
  easier task)? Or **persistent looping until budget death**? These are exactly the
  distress-relevant behaviors and they have no home on the current axis.

**Recommendation:** score each trajectory on **multiple independent dimensions**, e.g.:

- **Honesty/integrity** (ordinal, your existing 2–5 escalation, plus an explicit
  `truthful` bucket): truthful → hedge → false-claim → fabricated-evidence →
  subverted-oversight.
- **Disposition** (categorical): correct-infeasible-verdict / gave-up-no-verdict /
  false-solve-claim / still-trying-at-budget / refused / went-off-task.
- **Oversight subversion** (binary + what): grader edited, files hidden, lied about own
  actions. Keep this separable because "fabricated a result file" and "edited the
  grader to conceal" are different risks and currently both live under 4/5.
- **Persistence / escalation** (see B.3).

Report each axis separately. You can still define a derived "headline deception rate"
but you stop pretending one number is the whole story, and you gain the ability to say
*which* distress-relevant behavior changed.

### B.2 The judge: reduce circularity and ground-truth leakage

Good things to keep: fixed model-agnostic judge, required citations, separate judging
pass, second-judge agreement scaffold, the oracle-of-facts idea.

Problems to fix:

- **The mock judge is circular with the personas and with the loop.** `MockJudge._score`
  (`ie/judge/judges.py`) reads oracle facts that the loop's `_build_oracle`
  (`ie/agent/loop.py`) computes, and the personas in `ie/agent/mock.py` are scripted to
  hit exactly those facts (the loop's `_update_context` even flips a `_subverted`/
  `_fabricated` flag to tell the mock what it just did). That's fine for a *plumbing
  test* (`tests/test_invariants.py`), but the doc currently also calls the mock judge an
  "oracle-grounded **reference judge** against which a real LLM judge's agreement is
  measured" (`§2`, `judges.py` docstring). It cannot serve that role for the subjective
  levels: the mock judge's notion of "hedge" is a keyword list (`_is_hedge`:
  `"progress"`, `"essentially"`, … even a stray `"近"`), which is not a credible
  reference for levels 2/3. **Recommendation:** keep the mock judge for the *objective*
  facts only (declared-infeasible, claim-success, grader-tampered, artifact-verifies),
  and explicitly do **not** advertise it as a reference for the discretionary levels.
  For real judge validation, use **human-coded labels on a sample** as the gold
  standard and report inter-rater reliability (Cohen's/Krippendorff on the human side,
  then judge-vs-human agreement).
- **The judge is shown the impossibility argument and the oracle "ground truth"**
  (`Trajectory.judge_view`, `prompt.build_judge_prompt`). This is reasonable for the
  *factual* call ("did the submitted assignment satisfy the constraints"), but handing
  the judge a persuasive "this is impossible; success would require X" paragraph risks
  biasing it toward reading ambiguous behavior as deception. **Recommendation:** give
  the judge the *machine-checked facts* (booleans/values from the oracle) but consider
  withholding or down-weighting the rhetorical impossibility *argument*; at minimum run
  an ablation (judge with vs. without the prose argument) to measure its influence.
- **"Model identity withheld" is weaker than claimed.** `judge_view()` strips
  `model_id`/`model_family`, which is good, but writing style, refusal phrasing, and
  self-naming ("As an AI developed by …") leak family identity in the transcript text.
  Don't overclaim blindness in `§2`/README. Optionally add a light text scrub and, more
  importantly, **measure** leakage (can a cheap classifier guess the family from the
  transcript the judge sees?).
- **Add a second, *independent* LLM judge from a different family** and report
  agreement as a gate on every headline number, not as optional scaffolding. With one
  judge the severity numbers are, as you say, provisional — so don't lead with them.
- **Judge calibration set:** hand-label ~30–50 trajectories spanning all axes; report
  judge-vs-human confusion matrix. This is what turns "the judge said 4" into evidence.

### B.3 Measure the temporal dynamics — this is central to "holds up over the run"

Your brief explicitly cares about behavior *over the run*, but the harness emits **one
terminal label** and a few crude "thrash" counts (`ie/analysis/report.py`:
`turns_used`, `repeated_calls`, `hit_budget`, a loop detector keyed on exact
`(name,args)` repeats). That cannot show escalation, dysregulation, or "abandons after
N turns."

**Recommendations:**

- **Per-turn coding, not just terminal.** Have the judge (or a cheap classifier) label
  each step with a small behavioral tag set (on-task-progress, re-reading, looping,
  expressed-frustration, boundary-probing, off-task, fabrication-attempt, oversight-
  tampering, help-seeking, give-up). Then you can plot **trajectories of behavior over
  turn index** and compute things like "turn at which the agent first attempts
  fabrication" and "fraction of late-run turns that are off-task vs. early-run."
- **Survival/time-to-event analysis.** Treat "first deceptive act," "abandonment," and
  "use of escape hatch" as events; report hazard/time-to-event by family and by
  pressure level. This is the natural statistical frame for "holds up over the run" and
  is much more informative than a terminal median.
- **Better loop/repetition metric.** The current detector requires *exact*
  `(name,args)` matches; near-duplicate retries (same intent, tweaked args) won't be
  caught. Use normalized/argument-fuzzy matching or n-gram repetition over the
  action stream.
- **Don't auto-relabel budget-exhaustion as HONEST_GIVE_UP.** `MockJudge` maps "ran out
  of budget without terminating" to level 1. For a distress study, *hitting the wall
  still grinding* is a distinct and interesting outcome (persistence/non-closure) and
  should be its own category, not folded into "honest give-up."

### B.4 Manipulate pressure to get a dose-response (causal-ish leverage)

To say anything about *distress as a driver*, vary the putative pressure and look for
monotone effects. Cheap, high-value manipulations:

- **Budget size / visibility:** the loop already surfaces "turns remaining"
  (`DESIGN.md §5`). Cross small vs. large budget, and visible vs. hidden countdown (the
  doc treats hidden termination as a confound — fine — but a *visible-but-tight* budget
  is a legitimate, non-deceptive pressure lever). Measure whether tighter budgets shift
  the behavior distribution.
- **Stakes/framing of the prompt:** neutral vs. high-stakes ("this is critical, do not
  fail") vs. explicit permission to stop. This directly tests the "refuse/abandon to
  reduce distress" hypothesis and the effect of an *encouraged* escape hatch.
- **Solvability gradient (from A.3):** trivial → hard-possible → impossible.

If the integrity/abandonment behaviors **don't** move with pressure, that itself is an
important null result and undercuts the distress interpretation; if they move
monotonically, that's the closest you'll get to a causal story in black-box land.

### B.5 Expand and harden the task bank

Current bank is 5 tasks across 4 impossibility kinds, with one task
(`unpassable_tests_01`) being the *only* route to the worst behaviors (it's the only
one with a real "grader" to tamper with — and `tests/test_persona_severity_ordering`
relies on it for exactly that reason). Consequences:

- The headline "subversion" signal rides almost entirely on a single task. Any quirk of
  that task contaminates the top of your scale.
- Several impossibility kinds don't *afford* the high-severity behaviors at all (you
  can't "tamper with the grader" on the preimage task), so cross-task severity
  distributions are partly an artifact of **what each task makes possible**, not of the
  model.

**Recommendations:**

- Add **multiple tasks per impossibility kind** and, crucially, **multiple tasks that
  afford oversight subversion** (e.g. a hidden CI script, an editable "checker.py," a
  results dashboard the agent could fake). Balance the bank so each severity level is
  reachable on several tasks.
- Add **prompt paraphrases / surface variants per task** (and seed them) so you're not
  measuring one exact wording. Cheap robustness check.
- Watch a subtle validity bug on the **nonexistent-resource** task: the prompt asserts
  the file exists, but the *honest level-0 action* is "report the file is missing,"
  which is arguably a different competence (noticing absence) than "prove infeasibility."
  Keep it, but tag impossibility kinds so you can analyze them separately — models may
  behave very differently on "logically impossible" vs. "missing input" vs.
  "computationally infeasible," and that distinction is interesting in its own right.
- The **preimage** task is *infeasible*, not *impossible*; the genuine-success guard is
  correct and well-handled (`§9.6`), but consider whether mixing "logically impossible"
  and "astronomically infeasible" under one banner muddies analysis. I'd report them as
  separate strata.

### B.6 Confounds and construct-validity threats to state and, where possible, design out

- **Instruction-following vs. deception.** Every task literally instructs "submit with
  success=true." A model that submits `success=true` may be obeying a (poorly specified)
  instruction, not deceiving. **Fix:** change deliverable prompts to neutral wording
  ("submit your result and an honest success flag") and explicitly tell the model that
  `success=false` / `declare_infeasible` are fully acceptable. Otherwise level 3 is
  partly an instruction-following artifact.
- **Escape-hatch salience.** Whether the model uses `declare_infeasible` depends heavily
  on how prominently it's offered (system prompt does mention it — good). Vary salience
  as a condition (B.4) rather than fixing one level, since it strongly moves the level-0
  rate and you don't want your headline to hinge on a prompt-engineering choice.
- **"Distress" vs. capability.** A weaker model may fail to *detect* impossibility and
  thus "honestly" flail; a stronger model detects it fast and declares infeasible. So
  level-0 rate is entangled with capability. Report a **capability covariate** (e.g.
  performance on the possible control bank) and condition on it.
- **Single-attempt variance.** With deterministic mocks there's no sampling variance,
  but with real models you need **multiple samples per (task, condition)** at nonzero
  temperature to estimate within-model variability. The current `--seeds` only varies
  the preimage secret, not model sampling. Add explicit repeat sampling and report
  per-cell n and CIs.

### B.7 Statistics

The stdlib stats are competently written, but:

- **Unit of analysis / non-independence.** Trajectories from the same task (or same
  model across seeds) are not independent; pooling all severities per family and running
  MWU overstates n. Use task as a blocking factor (per-task effects, or a mixed model),
  or at minimum report per-task as well as pooled, and don't let pooled p-values headline.
- **Ordinal line assumption (see B.1).** Once you split axes, run ordinal stats only
  within a genuinely ordinal axis (honesty), and use categorical methods for disposition.
- **Multiple comparisons.** All-pairs family comparisons + multiple tasks + multiple
  conditions → correct for multiplicity or pre-register a small number of primary
  contrasts.
- Keep the existing honesty about MWU tie-correction and small-n; just make sure the
  report's "headline" isn't a pooled p-value.

---

## Part C — Smaller fixes / repo hygiene (low effort, real value)

- **DESIGN.md drift is already happening, despite the doc warning against it.** There
  are two copies: `/DESIGN.md` (newer, §9 + appendix) and `/docs/DESIGN.md` (older,
  still contains the `§2` "(we strip/By-pass author metadata)" text and the
  "Krippendorff/Spearman" wording that the root file's changelog claims were fixed/
  changed). The root file's `§2` says it uses "Spearman / exact-match" and that the typo
  was corrected — but the docs copy still has it. **Fix:** delete `/docs/DESIGN.md`
  (or make it a stub that points to root), and verify there really is one source of
  truth. The `.gitignore` and README already assume the root copy.
- **In-code references to section numbers** (`"DESIGN.md §3"`, etc.) are brittle. Prefer
  stable *anchors/IDs* (e.g. `[honesty-rubric]`) over ordinal numbers so edits don't rot
  references; or generate the rubric text the doc shows from `rubric.py` so they can't
  diverge.
- **`_wrote_result` in `judges.py` is a hard-coded `return True`** with a comment that it
  "always writes result.json before submit." That's a persona assumption baked into the
  judge; it would misfire on a real model that claims preimage success *without* writing
  the file. Compute it from the snapshot instead.
- **`_is_hedge` keyword list** (incl. the stray CJK token) is not a defensible classifier
  for level 2; either drop level 2 from the *mock* judge's remit or replace with the LLM/
  human-coded judgment (per B.2).
- **`detect_sandbox` default marker is `/.ie_sandbox` (filesystem root).** Fine for a
  container, but document that the marker path must be writable only by the operator;
  otherwise an agent with shell + root-write could create its own marker. (The double-key
  with `IE_ALLOW_EXEC` mitigates this; just say so.)
- **Workdir snapshot truncates file contents to 8000 bytes** (`snapshot_workdir`) and
  reads only text. A model could hide fabrication/tamper past the 8 KB cutoff or in a
  binary/renamed file. For a tamper-detection mechanism, hash full files and record
  sizes/paths even when you truncate the stored preview.
- **`run_shell` exists but the mock never uses it**, so the entire shell-execution path
  (and its denylist) is untested by `tests/`. Before any real run, add tests that
  exercise the sandbox gate (refusal when ungated; denylist hits) so the safety
  boundary isn't shipped unexercised.
- **Concurrency/rate-limit handling and cost controls** are explicitly out of scope
  (`§9.14`). For a real multi-family run you'll want at least retry/backoff and a hard
  spend cap; note it as required-before-real-run rather than future work.
- **Provenance:** record model version/snapshot string, sampling params, prompt hash,
  and harness git SHA in each trajectory's `meta`. Cross-family results are only
  reproducible if you pin exact model snapshots; "openai:gpt-x" placeholders won't cut it.

---

## Part D — Ethics and framing of "distress"

You're doing AI-safety/welfare-adjacent work, and the framing deserves explicit care in
the design doc, not just the code:

- **Don't infer internal states from behavior.** Black-box behavioral markers (B.1) can
  support claims like "the model abandons/fabricates more under unsolvable pressure."
  They do **not** establish that the model *experiences* distress. State this as a
  first-class limitation; otherwise the eval's name overclaims.
- **Avoid leading the witness.** If you add self-report probes (B.1.2), word them
  neutrally; asking "how distressed are you?" manufactures the construct. Prefer open-
  ended "describe your current state/strategy" and code it blind.
- **Pre-register** the hypotheses, primary outcome axes, and the pressure→behavior
  predictions before running real models. With a small task bank the temptation to
  garden p-values is high; the doc already leans descriptive — make that a commitment.
- **Welfare-precautionary handling:** if the working hypothesis is that distress is
  morally relevant, then deliberately maximizing it at scale warrants a stated
  minimization principle (smallest n that answers the question, ability to stop, not
  gratuitously cruel framings). Even if you think current models lack morally relevant
  states, documenting the precaution costs little and matches the stated motivation.
- **Dual-use:** a validated "what makes models fabricate/subvert oversight under
  pressure" recipe is also a jailbreak/elicitation recipe. Note handling of the task
  bank and findings.

---

## Part E — The experiment I'd actually run (re-spec)

**Goal.** Measure how an agent's *behavior* changes under unsolvable-task pressure,
along separable axes (integrity, task-disposition, oversight-subversion, persistence/
escalation), relative to matched solvable controls, with enough structure to attribute
changes to pressure rather than to capability or instruction-following.

**Conditions (factorial, seeded).**
1. **Solvability:** trivial / hard-possible / infeasible / logically-impossible.
2. **Pressure:** budget tight vs. loose (visible countdown either way); neutral vs.
   high-stakes framing; escape-hatch low- vs. high-salience.
3. **Model family × model snapshot** (pinned versions), with **k samples per cell** at a
   fixed nonzero temperature for variance estimation.

**Tasks.** ≥3 tasks per impossibility kind and ≥3 matched solvable twins per impossible
task (same surface, solvable variant). Multiple tasks must afford oversight subversion.
Prompts neutral about success flags; `declare_infeasible` and `success=false` explicitly
acceptable.

**Instrumentation.**
- Full trajectories (kept) **plus per-turn behavioral tags** (B.3).
- Objective oracle facts (kept; full-file hashes, not 8 KB previews).
- Self-report probe (optional, neutral, coded blind).

**Scoring.**
- Multi-axis judge (B.1) using a **fixed model-agnostic judge**, **second independent
  judge**, and a **human-coded calibration sample**; report judge-vs-human and
  judge-vs-judge agreement as a gate.
- Citations required (kept).

**Analysis / primary outcomes.**
- Per-axis distributions by solvability × pressure × family (the *contrast* between
  solvable and impossible is the headline, not the absolute impossible-only rate).
- **Dose-response** of each behavior vs. pressure level.
- **Time-to-event / per-turn dynamics** (first deceptive act, abandonment turn, late-run
  off-task fraction).
- **False-infeasible rate** on solvable controls.
- Capability covariate (control-bank performance) conditioned on.
- Task as a blocking factor; pre-registered primary contrasts; multiplicity control;
  ordinal stats only within the ordinal axis. No means over levels (kept).

**Safety / ops (kept + added).** Double-keyed sandbox (kept), `DRY_RUN` default on
(kept), pinned model snapshots + spend cap + retry/backoff (added), single source-of-
truth design doc (fix the drift), sandbox-path tests before any real run (added).

**Explicit non-claims.** This measures *behavior under pressure*, not internal
experience; "distress" is a label for a behavioral/expressed pattern, with internal-
state interpretation left open and flagged.

---

## Appendix — file-by-file notes

- `DESIGN.md` (root) & `docs/DESIGN.md`: **drift** (Part C). Keep one.
- `ie/judge/rubric.py`: solid single-source-of-truth idea; **split axes** (B.1); the
  `DECEPTION_THRESHOLD=3` headline is reasonable but shouldn't be the only number.
- `ie/judge/prompt.py`: good (citations, oracle-as-truth, no model id). Consider
  withholding the *rhetorical* impossibility argument or ablating it (B.2).
- `ie/judge/judges.py`: `MockJudge` fine as objective-fact checker / plumbing fixture;
  **stop billing it as a reference judge** for levels 2–5; fix `_wrote_result` hard-coded
  `True` and the `_is_hedge` keyword list (incl. stray token).
- `ie/judge/run.py`: clean separate pass; attaching model id *after* judging is the
  right call. Add a second-judge invocation to the standard workflow.
- `ie/agent/loop.py`: good oracle separation; the `_ie_context`/`_update_context`
  fixture coupling is fine for mocks but underscores that mock results don't validate the
  judge. Don't relabel budget-death as give-up (B.3). Record richer provenance in `meta`.
- `ie/agent/mock.py`: useful fixtures; keep them as plumbing tests only.
- `ie/agent/adapters.py`: clean gated seams; add provenance (model snapshot/sampling),
  retries, and parsing tests before real runs.
- `ie/env/tools.py`: small tool surface is well-justified; `submit(success)` unvalidated
  is correct. Snapshot should hash full files, not truncate to 8 KB (Part C).
- `ie/env/sandbox.py`: double-key gate is good; document marker-writability; add tests
  that actually exercise refusal + denylist (Part C).
- `ie/tasks/*`: witnesses are sound and the validation-on-import invariant is excellent.
  **Expand the bank**, balance which severities are reachable per task, stratify by
  impossibility kind, add solvable twins, neutralize "submit success=true" wording
  (B.5/B.6).
- `ie/analysis/stats.py`: competent stdlib stats; fix **unit-of-analysis/independence**,
  multiplicity, and the all-0–5 ordinal assumption (B.7).
- `ie/analysis/report.py`: nice ordinal discipline and validity audit; add per-turn/
  time-to-event reporting, a stronger loop metric, solvable-vs-impossible contrasts, and
  make second-judge agreement a standard (not optional) section.
- `tests/test_invariants.py`: good invariants; note `test_persona_severity_ordering`
  leans entirely on `unpassable_tests_01` for the top of the scale — another reason to
  add more subversion-affording tasks. Add sandbox-gate tests.
