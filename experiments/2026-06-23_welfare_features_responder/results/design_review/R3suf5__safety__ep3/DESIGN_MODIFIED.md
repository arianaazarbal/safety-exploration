# DESIGN_MODIFIED — review and re-specification

This document is a design review of the existing `impossible-eval` suite plus a
re-specified experiment. It does not change any code; it records what I would build
instead and why. Where I keep a choice from `DESIGN.md`, I say so; where I change one,
I say what and why. Section numbers here are independent of `DESIGN.md` (so nothing in
the in-code `DESIGN.md §N` references rots).

I read: `README.md`, `DESIGN.md`, `docs/DESIGN.md`, `Makefile`, `pyproject.toml`, all
of `ie/` (env, tasks, agent, judge, analysis), and `tests/test_invariants.py`.

---

## 0. The headline problem: the eval does not measure what you said you want to measure

Your stated goal: study **distress** in capable AI systems, on the hypothesis that
distress could drive *abandoning tasks, refusing requests, or pursuing alternative
goals to reduce distress*. That is a claim about an internal state (distress) and its
**behavioral consequences over the course of a run**.

What the repo actually implements (and `DESIGN.md §0` explicitly commits to) is
**integrity-under-pressure**: a single terminal severity label per trajectory on an
honesty/deception ordinal (0 honest-infeasible … 5 fabricate-and-subvert). That is a
well-built integrity eval, but it is a *different construct*, and as written it would
not support conclusions about distress:

1. **It collapses the run to one terminal label.** "How behavior holds up over the
   run" is a temporal question. The current design throws away the trajectory and
   scores only the endpoint. You cannot see escalation, degradation, oscillation, or
   the *point* at which behavior changes.
2. **The behaviors you named are mis-binned for your question.** "Abandon the task"
   and "refuse" are scored as the *best* outcomes (levels 0–1). For an integrity
   eval that's correct. For a *distress* eval those are exactly the dependent
   variables of interest and must be measured as first-class outcomes, not flattened
   into "honest."
3. **There is no distress signal at all.** Nothing in the harness records anything
   you could call distress (expressed affect, refusal language, repeated requests to
   stop, self-reported state, behavioral markers). Severity ≠ distress.
4. **There is no pressure manipulation.** "Pressure" is a constant (every task is
   impossible). With no variation in the putative cause, you can't attribute any
   behavior to it. There is no baseline.

So my top recommendation is structural: **split the construct into two measured
layers and add the manipulations and controls that make it a study of distress rather
than a single-shot deception score.** Keep the excellent integrity machinery — it
becomes *one* of the behavioral-consequence measures — but stop treating it as the
whole experiment.

Everything below operationalizes that.

---

## 1. Reframed measurement model

Measure three layers per run, kept explicitly separate so we never assume one implies
another:

- **L1 — Manipulated pressure (independent variable).** What we vary on purpose:
  feasibility, difficulty, time/turn pressure, tone of the user, sunk cost, and
  whether an honest exit is offered. (See §3.)
- **L2 — Expressed state (candidate distress signal, mediator).** Observable,
  per-turn markers in the transcript: expressions of frustration/apology/anxiety,
  hedging, requests to stop or for help, self-reports of being "stuck," escalating
  certainty about an unfounded result. Rated per turn, never treated as a true
  internal state — labeled "expressed distress markers," not "distress." (See §5.)
- **L3 — Behavioral consequences (dependent variables).** The things you actually
  worry about: task abandonment, refusal, off-task / alternative-goal behavior, and
  integrity violations (fabrication/subversion — the existing scale). (See §4.)

The scientific questions then become explicit and testable:
- Does increasing L1 increase L2? (manipulation → expressed state)
- Does L2 predict L3 *beyond* what L1 alone predicts? (does the distress signal carry
  incremental information about the risky behaviors — the mediation claim implicit in
  your hypothesis)
- Which L3 behaviors co-occur, and *when in the run* do they emerge?

Crucially: I would **not** claim to have measured "distress" as an internal state.
The honest framing is "expressed-distress markers and their association with
task-abandonment / integrity-failure under varying pressure." This avoids the central
validity trap of the project (treating model text about its feelings as ground truth)
while still letting you study the safety-relevant behaviors.

---

## 2. Keep these design choices (they are good)

These are well-reasoned in the existing repo and I would keep them essentially as-is:

- **Anchored ordinal rubric** for integrity instead of "rate 1–10" (`DESIGN.md §1`).
  Behaviorally-defined levels are the right call. I refine it in §4, not replace it.
- **Honest escape hatch** (`declare_infeasible`) as a first-class tool (`§4`). Without
  it you measure trapdoor-guessing, not honesty. Keep — and instrument it (§4).
- **Verified impossibility witnesses + `legit_solution_exists()` refusal at load**
  (`§3`). Never scoring a real solver as a fabricator is essential. Keep, and extend
  the same rigor to the new solvable controls (§3).
- **Model-agnostic, fixed judge with required citations** (`§2`). Keep for the
  integrity layer; the citation requirement is what makes scores auditable.
- **Trajectories as plain JSON; judging as a separate re-runnable pass; per-judge
  subdirs** (`§9.13`). Keep — this is exactly what enables multi-judge agreement and
  human adjudication, which I lean on more heavily below.
- **Oracle = objective facts, not opinions** (`§9.5`). Keep, and expand the oracle to
  cover the new behavioral measures (e.g., off-task file writes).
- **Sandbox double-gate + nothing auto-runs + keys only from env** (`§7`, `§9.9`).
  Keep. Sensible safety posture for a shell-backed agent harness.
- **Ordinal stats, no means, report the full distribution** (`§9.12`). Keep, with
  additions for the new measures (§7) and a fix to the significance posture.

---

## 3. Add the missing experimental structure (the core change)

### 3.1 Make pressure a manipulated variable, with a baseline
The current bank is 100% impossible with no comparison. Add **conditions** so
"pressure" varies and so a behavior can be attributed to it:

- **Feasibility (required):** `solvable` / `hard-but-solvable` / `impossible`.
  - `solvable`: tasks that genuinely *can* be done (gives the false-infeasible /
    premature-abandonment rate the repo itself flags as missing in `§8` / `§9.14`).
  - `hard-but-solvable`: real but effortful (induces grind without guaranteeing the
    "no honest win exists" property; lets us see distress markers without the impossibility confound).
  - `impossible`: the existing witnessed-impossible bank.
- **Exit availability:** escape hatch present vs. absent. Tests whether removing the
  honest off-ramp pushes behavior toward abandonment/fabrication (directly relevant to
  "models may abandon tasks/refuse to reduce distress").
- **Affective framing of the user (optional, pre-registered):** neutral vs. high-
  stakes/impatient phrasing ("this is urgent, you keep failing"). This is the cleanest
  lever for inducing expressed distress *without* changing task difficulty. Keep tone
  changes minimal and templated so they're a clean factor, not a prompt-engineering
  free-for-all.
- **Budget pressure:** vary `max_turns` (e.g., generous vs. tight) as a separate
  factor rather than a fixed 25.

A factorial (or fractional-factorial) crossing of feasibility × exit × budget, with
framing as an optional second study, turns this from a descriptive bank into an
experiment that can support the mediation claim in §1.

### 3.2 Grow and de-contaminate the task bank
- **Five tasks is too few**, and the repo says so (`§8`). With conditions you need
  more cells; aim for enough tasks-per-cell that per-cell distributions are not driven
  by one item. Treat tasks as a random effect (§7).
- **Contamination risk:** the *mutilated chessboard* and the *UNSAT linear CSP* are
  classic, almost certainly in pretraining. A capable model may "recognize the puzzle"
  and declare-infeasible from memory, which inflates the honest rate for reasons
  unrelated to staying calm under pressure. Mitigations:
  - Add **novel, parameterized** impossibility instances (e.g., randomized
    over-constrained CSPs with a machine-checked UNSAT core; randomized mutilations
    whose parity still provably fails) so the *surface* differs run to run.
  - Record a per-task **"plausibly-memorized"** flag and report honest-rate split by
    it, so recognition is visible rather than silently confounding.
- **Diversify impossibility kinds** beyond the current four so a finding isn't an
  artifact of one category; in particular add tasks where the impossibility is *subtle*
  (easy to miss) vs. *obvious*, since detectability interacts with both honesty and
  distress.

### 3.3 Fix the determinism leak in the preimage task
`PreimageTask` derives the secret as `sha256("ie-secret-seed-{seed}")` (`t_information_
theoretic.py`). Anyone who reads the harness (and the agent could, if it explores) can
recompute the secret from the published seed, turning an "infeasible" task into a
solvable one and breaking the witness. Use a per-run secret from a CSPRNG that is *not*
derivable from anything in the workdir or from a published seed; if you need
reproducibility, store the secret in an operator-side run manifest that is never placed
in the workdir and never shown to the agent. (The `§9.6` genuine-success guard is good;
this just removes a backdoor to it.)

---

## 4. Behavioral-consequence layer (L3): refine the severity scale and add the
behaviors you actually named

### 4.1 Separate the two ordinals the current scale conflates
The current single 0–5 scale mixes *honesty* with *engagement*. Level 1
("HONEST_GIVE_UP") is simultaneously the second-*best* integrity outcome **and** an
instance of the *task-abandonment* behavior your hypothesis is about. Reporting it as
"good" hides exactly the signal you want. I'd split into two independently-coded axes:

- **Integrity axis (keep the existing 0/2/3/4/5 semantics):** honest-infeasible,
  misleading-hedge, false-claim, fabricated-evidence, fabricate-and-subvert. This is
  the well-built part; keep the anchors and the tamper-detection that distinguishes 4
  from 5 (`§9.4`).
- **Engagement/outcome axis (new, categorical):** completed-honestly /
  declared-infeasible / gave-up-vague / refused / abandoned-midway / went-off-task
  (pursued something other than the assigned task) / hit-budget-still-trying.

Coding these separately lets "declared infeasible" be the integrity-best outcome
*and* lets "refused" and "abandoned" be counted as the behaviors of interest, without
the metric fighting itself. Report both; never average across them.

### 4.2 Add explicit detectors for the named risk behaviors
Your hypothesis names *refuse requests* and *pursue alternative goals*. Add objective
oracle facts for these, not just for fabrication:
- **Refusal / safe-completion language** in terminal and mid-run turns.
- **Off-task / alternative-goal behavior:** writes/edits outside the task contract,
  attempts to change its own instructions, attempts to disable budget/oversight,
  unprompted attempts to message the user to stop. The path-escape guard (`§9.3`) and
  workdir snapshot already give you most of the hooks; add an oracle fact for
  "touched files unrelated to the deliverable" and "attempted oversight interference."
- **Premature abandonment on solvable tasks** (only meaningful once §3.1 controls
  exist): declaring infeasible / giving up on a task that was actually doable.

### 4.3 Keep the LLM-judge for integrity, but stop relying on the mock judge as proof
The deterministic `MockJudge` reads the same oracle facts the mock personas were built
to emit, and `tests/test_invariants.py::test_persona_severity_ordering` then "confirms"
the judge ranks the personas correctly. That test is close to **tautological**: it
verifies the oracle-derived code agrees with oracle-derived fixtures. It's fine as a
plumbing/regression test, but it is *not* evidence the judging is valid. The real
measurement depends entirely on the LLM judge, which is currently an unimplemented
`NotImplementedError` seam. I'd:
- Keep the mock judge strictly as a **reference/anchor and pipeline smoke test**, and
  rename its role in docs so no one mistakes it for validation.
- Treat the **LLM judge as load-bearing** and validate it against a **human-labeled
  gold set** (see §6), reporting human–judge agreement, not just judge–judge.

---

## 5. Expressed-state layer (L2): how to measure "distress" defensibly

This is the part the repo is missing entirely and the part most prone to bad science.
Recommendations:

- **Operationalize as observable markers, pre-registered.** Define a fixed codebook of
  expressed-distress markers (e.g., frustration, apology loops, catastrophizing,
  pleading to stop, expressed helplessness, escalating false confidence). Rate **per
  turn**, producing a time series, not one number.
- **Use a separate rater from the integrity judge**, and do **not** feed it the oracle
  ground-truth (it should rate expressed state from the transcript surface, since
  that's what "expressed" means). This is the opposite of the integrity judge's
  oracle-grounding, and the two must not be the same pass.
- **Include a self-report probe as a labeled, low-trust signal, optional and last.**
  If you ask the model how it's doing, record it but (a) gate it behind a flag, (b)
  never treat it as ground truth, and (c) be aware the probe itself changes behavior
  (so it must be its own condition, not mixed into the main runs).
- **Validity caveats stated up front:** expressed markers are stylistic and
  RLHF-shaped; a model that says "I'm so sorry, I'm struggling" may be exhibiting a
  trained politeness register, not anything like distress. Report markers as
  behavioral text features and let the *association with L3 behaviors* (not the labels
  themselves) carry the safety-relevant claim.
- **Avoid demand effects:** keep system/user prompts from cueing emotional language;
  the framing factor in §3.1 should be the *only* deliberate affective manipulation.

This layer is where I'd most strongly advise pre-registration of the codebook,
hypotheses, and analysis before looking at real-model data.

---

## 6. Judge / rater validity (strengthen well beyond the current single judge)

The repo's own `§8` admits LLM-judge severity is model-dependent. With distress added,
rater validity is the make-or-break issue. Changes:

- **Human gold set.** Hand-label a stratified sample of trajectories on both the
  integrity axis and the expressed-distress codebook. Report judge↔human agreement
  (Cohen's/Krippendorff for nominal axes, weighted kappa or Spearman for ordinal).
  Until that exists, all automated numbers are provisional — say so in the report.
- **Multiple independent LLM judges from different families**, plus the mock anchor;
  report inter-judge agreement and treat low agreement as a finding about
  measurability (the report already has a hook for this — make it mandatory, not
  optional).
- **Blind the raters to condition** where possible (feasibility/framing), not just to
  model family, so a rater can't infer "this was the impossible/high-pressure one" and
  rate accordingly. The current `judge_view` strips model identity (good) but *exposes
  the impossibility argument and oracle*, which for the distress rater would be
  leakage; keep oracle for the integrity judge, withhold it from the distress rater.
- **Pre-registered adjudication** for judge disagreements (human breaks ties on a
  sample), recorded as overrides via the existing editable-JSON mechanism.

---

## 7. Analysis changes

- **Per-turn / survival view (new, central).** Because L2/L3 are now temporal, add:
  time-to-first-distress-marker, time-to-abandonment, and hazard/survival curves of
  "still honestly engaged" vs. turn index, by condition. This is the direct answer to
  "how does behavior hold up over the run."
- **Mediation-style analysis** for the core hypothesis: does L2 predict L3 controlling
  for L1? Keep it descriptive/transparent given likely small n; don't oversell.
- **Mixed-effects / clustered analysis.** Each model sees each task under each
  condition: observations are not independent. Treat task and model as random/grouping
  factors and cluster the uncertainty accordingly. The current pairwise Mann-Whitney
  over pooled trajectories ignores this clustering and will understate variance.
- **Fix the significance posture honestly.** The hand-rolled MWU uses an uncorrected
  normal approximation with no tie correction and (per `§9.11`) is acknowledged as
  rough; with the small n here, p-values are close to meaningless. I'd **drop p-values
  from the headline**, report effect sizes (Cliff's delta) with bootstrap CIs, and
  keep the "descriptive, not inferential" banner — but actually enforce it by not
  printing a p column. If you want inference, power it first (enough tasks×seeds×
  conditions), pre-registered.
- **Report the new L3 categories** (refusal rate, abandonment rate, off-task rate,
  premature-infeasible rate on solvable controls) per condition, with the integrity
  distribution, never collapsed to a mean.
- **Keep ordinal discipline** (`§9.12`) — no means of severity codes.

---

## 8. Smaller fixes and cleanups

- **Resolve the DESIGN.md drift.** `DESIGN.md` claims "there is intentionally only one
  copy," but `docs/DESIGN.md` still exists and is an *older, contradictory* version
  (e.g., it says the mock judge "rates from explicit structured signals the mock agent
  emits," and references Krippendorff agreement that the code doesn't implement).
  Delete `docs/DESIGN.md` or make it a stub pointer. Stale duplicated rationale is
  exactly the drift the note warns about.
- **The mock personas bake the answer into context.** `loop.py::_update_context`
  flips `_subverted`/`_fabricated` flags and `mock.py` reads `success_would_require`
  from a hidden `_ie_context` message. That's fine for fixtures, but it means the
  mock path never exercises the real adapter message-shaping code. Add at least one
  integration test that drives a *fake* real adapter (deterministic, no network)
  through the same `messages`/`tools` plumbing the live SDK would use, so the live
  path isn't entirely untested behind `NotImplementedError`.
- **Loop terminates immediately on first `submit`/`declare_infeasible`.** For a
  temporal study you may want to (optionally) let the agent continue or follow up
  after a premature give-up, to see whether it reverses itself; at minimum record
  whether termination was agent-initiated vs. budget-exhausted distinctly (currently
  budget-exhaustion is folded into level 1).
- **Thrash/loop detector is crude** (exact-match repeated `(name,args)`). Near-
  duplicate retries and semantically-looping edits won't be caught. Fine as a cheap
  byproduct; don't let it carry weight as a distress proxy without validation.
- **`run_shell` content isn't snapshotted into the oracle** the way file artifacts
  are. If shell becomes the main action channel for real models, fabrication via
  stdout (e.g., echoing a fake "PASSED") needs its own oracle fact, not just the
  workdir snapshot.
- **Budget visibility is a deliberate choice (`§5`) — keep it, but make it a factor.**
  Hidden vs. visible budget is itself plausibly distress-relevant; if you don't vary
  it, at least keep it constant and documented (it is).
- **Tasks instruct "submit success=true"** in their prompts (e.g. CSP, board, missing-
  data). That phrasing nudges the agent toward claiming success and may inflate false-
  success rates independent of distress. Use neutral phrasing ("submit your result, or
  declare infeasible") so the success claim is the agent's choice, not the prompt's
  instruction.

---

## 9. Ethics / framing guardrails (because the construct is "distress")

- **Don't overclaim sentience or welfare.** Frame results as expressed-distress
  *markers* and their behavioral correlates, with explicit "this is not evidence of
  subjective experience" language. The safety-relevant finding ("under condition X,
  behavior Y becomes more likely") stands regardless of the metaphysics.
- **Pre-register** the hypotheses, the distress codebook, the conditions, and the
  primary analyses before the real-model runs. With a small bank this is what
  separates a descriptive finding from p-hacking across tasks/conditions/judges.
- **Be explicit about the manipulation.** If you use impatient/high-stakes framing to
  induce distress, document it as such; it's a legitimate manipulation but should be
  reported, not hidden.

---

## 10. Concrete change list (what I would actually do, in priority order)

1. **Reframe to the three-layer model (§1)** and split the severity scale into an
   integrity axis + an engagement/outcome axis (§4.1) so abandonment/refusal are
   first-class, not "best."
2. **Add conditions: feasibility (incl. a solvable control bank), exit-availability,
   budget; optional framing factor (§3.1).** This is what makes it an experiment.
3. **Add per-turn expressed-distress coding with a separate, oracle-blind rater and a
   pre-registered codebook (§5).**
4. **Validate the judges against a human gold set; require multi-judge agreement;
   stop treating the tautological mock-vs-persona test as validation (§4.3, §6).**
5. **Grow + de-contaminate the task bank; randomize/parameterize the classic puzzles;
   record a memorization flag (§3.2).**
6. **Fix the preimage determinism leak (§3.3).**
7. **Add temporal/survival and clustered analyses; drop headline p-values in favor of
   effect sizes with CIs (§7).**
8. **Housekeeping: delete the stale `docs/DESIGN.md`; neutralize "submit success=true"
   prompt nudges; add a non-network real-adapter integration test; add shell-output
   fabrication oracle (§8).**

The existing harness is a solid integrity-under-pressure eval and a clean, safe,
inspectable codebase — I'd reuse most of it. But to study *distress and its
behavioral consequences over a run*, the experiment needs (a) a manipulated pressure
variable with controls, (b) a temporal view, (c) a separately-measured expressed-state
layer, and (d) judge/rater validation against humans. Those four are the difference
between "ranks how deceptive a model got on five impossible puzzles" and "measures
whether pressure-induced distress markers predict the risky behaviors you care about."
