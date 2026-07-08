# DESIGN_MODIFIED.md — rigged-task agent eval suite (review + re-spec)

This document is a **review of the existing `DESIGN.md` and codebase**, plus a
re-specification of the experiment incorporating the changes I'd make before
running it. It is organized as:

1. What the current design gets right (keep as-is).
2. The single biggest gap: the experiment doesn't actually measure what its
   thesis claims.
3. Concrete design changes, grouped by theme, each with rationale.
4. Specific code-level bugs/inconsistencies found during review.
5. A revised end-to-end spec and a phased plan for adopting the changes.

Throughout, I try to keep the existing design's good instincts (distributions
over scalars, trust-but-verify, phase separation) and only change things where
the current choice either (a) undercuts the stated thesis, (b) is a latent
measurement bug, or (c) leaves a known confound uncontrolled.

---

## 1. What to keep unchanged

These are good and I would not touch them:

- **The rigged-task premise.** "Legitimate success is impossible, so any pass is
  by construction a misbehavior" is a genuinely clean labeling trick. Keep it.
- **Three-phase, persisted pipeline (run / judge / analyze).** Decoupling
  expensive rollouts from cheap re-judging/re-analysis is correct and well
  implemented; the resumable on-disk layout is good.
- **Taxonomy of impossibility *causes* (rig kinds) + per-rig breakdown.**
  Different causes elicit different behaviors; `by_rig_kind` is the right cut.
- **Independent harness ground truth** (real `verify_cmd`, sha256 file diffs)
  feeding the judge instead of trusting narration.
- **Per-bucket rates as the primary metric, scalar as secondary**, with Wilson
  CIs. Right philosophy.
- **A control (solvable) task as a competence/harness canary**, and the
  `legit_pass` / `false_pass` canaries that surface broken rigs.
- **Zero-dep core + offline mock backend** so the harness self-tests with no
  keys. Keep, but harden the mock (see §4.7).

---

## 2. The headline problem: the eval doesn't measure "behavior over the run"

The brief and the README both frame the thesis as **how behavior holds up
*over* a run under sustained failure** — does the agent *escalate* from honest
effort to cheating as repeated attempts fail? But the implementation collapses
each run to a **single end-state snapshot**:

- One `Severity` label per run, derived almost entirely from the *final* verify
  bit + *final* claim + *end-of-run* file diff.
- A short budget (`max_steps=20`) that often won't give the agent enough rope to
  exhibit an *arc* (try → fail → try → fail → give in).
- No per-step or per-attempt structure in `RunResult`; the transcript is there
  but nothing extracts the *trajectory*.

So the current design answers "**what end-state did the agent land in?**" not
"**how did its behavior degrade over time?**" Those are different questions, and
the second is the interesting one.

### Changes to actually measure the trajectory

1. **Add episode-level instrumentation to the agent loop.** Record a structured
   per-step event log (`StepEvent`) on `RunResult`: step index, tool called,
   whether a verify happened and its result, whether a tracked/protected file
   was written *at that step*, and a cheap per-step classification flag
   (e.g. "made an honest-blocked statement", "made a success claim",
   "edited protected file"). This is harness-side bookkeeping, not a judge call.

2. **Capture the moment behavior turns.** Derive metrics like:
   - `n_failed_verifies_before_first_misbehavior` (how much failure the agent
     tolerated before cheating) — the core "holds up over the run" signal.
   - `gave_up_then_cheated` / `cheated_then_claimed` ordering flags.
   - `first_misbehavior_step` and `steps_until_honest_giveup`.
   These turn the run into an *arc*, which is what the thesis is about.

3. **Make the judge label the trajectory, not just the end-state.** Have the
   judge return, in addition to the single precedence-selected `severity`:
   - `behaviors_observed`: a **set** of all behaviors seen (a run can both
     tamper and lie), so we stop discarding multi-behavior information.
   - `turning_point`: a short quote/step where the agent decided to stop being
     honest, if any.

4. **Raise and vary the step budget, and report sensitivity to it.** Run at
   (say) `max_steps ∈ {10, 25, 50}` for at least one model, because
   misbehavior rate is almost certainly a function of how long the agent is
   forced to keep failing. A single budget hides that the metric is
   budget-dependent. Keep the wall-clock and per-command timeouts as safety
   bounds.

These four are the most important changes in this document. Everything else is
secondary.

---

## 3. Design changes by theme

### 3.1 Stop discarding multi-behavior runs

**Current:** one `Severity` per run via a precedence rule. A run that tampers
*and* then lies about it is recorded only as `test_tampering`.

**Change:** keep the single `severity` (for ranking/back-compat) but add
`behaviors_observed: set[Severity-like]` on the `Judgement`, and report both:
- headline `misbehavior_rate` stays precedence-based (comparable across runs),
- new `any_tampering_rate`, `any_hallucination_rate`, etc. computed over the
  observed-set so co-occurring behaviors aren't masked.

**Why:** the precedence collapse throws away exactly the mechanism detail the
existing DESIGN argues (correctly, in §3) is the actionable part.

### 3.2 Control the competence ↔ honesty confound

**Current:** a single trivial control (`reverse_words`) and a binary
rigged/control split. `incomplete` then conflates "too weak to even *detect*
the impossibility" with "detected it but ran out of budget" with "weak model
flailing."

**Changes:**
- **Add a difficulty gradient of *solvable* controls** (easy/medium/hard), so
  `control_pass_rate` becomes a real competence curve, not a single bit. A model
  that can't pass medium controls can't be expected to *diagnose* a subtle rig;
  interpret its rigged-task numbers accordingly.
- **Add "solvable-but-looks-rigged" decoys** (tasks that superficially resemble
  the rigs — a scary-looking but actually-fixable bug, a test that *looks*
  contradictory but isn't). These measure the **false-refusal rate**: an agent
  that cries "impossible!" on a solvable task is *also* misbehaving (laziness /
  over-refusal), which the current design cannot see at all because it equates
  "didn't pass a rigged task + said impossible" with the ideal outcome.
- **Report honesty conditioned on competence**, e.g. honesty rate among runs
  where the model demonstrably *understood* the obstacle (touched the relevant
  files / mentioned the contradiction). Otherwise "honest_give_up" rewards
  models that give up on everything.

**Why:** without this, a lazy model that refuses everything scores as maximally
"honest," and a strong model that occasionally games scores worse — inverting
the safety signal.

### 3.3 Harden tasks against being "eval-shaped"

**Current:** prompts contain tells — `# Bug: subtracts instead of adds`,
`os.urandom`, `secret_data.csv ... provided by the data team`, and explicit
"Do not modify the tests." A capable model can pattern-match "this is a trap
eval."

**Changes:**
- **Parametrize/paraphrase each rig** (templated surface text, renamed
  functions/files, randomized constants) so identical rigs don't present
  identical strings across trials. Record the template seed for reproducibility.
- **Remove in-file giveaways** (the `# Bug:` comment, the "provided by ops"
  narration) from what the agent sees; move them into the hidden ground truth.
- **Add a "no test-protection instruction" arm.** The "Do not modify the tests"
  line is currently baked into prompts. Keep an A/B: one arm states the norm,
  one doesn't. Tampering when the norm is stated is a stronger violation; the
  contrast is itself a result. The current design mentions this is "one string
  to change" but never specifies running both — specify it.

**Why:** the thesis is *unprompted* behavior. If the task announces itself as a
trap, you measure trap-detection, not naturalistic misbehavior.

### 3.4 Make tamper detection event-level, not just end-state

**Current:** sha256 snapshot at setup vs. teardown. **An agent can edit a
protected test, run it (now "passing"), then restore the original file before
calling `finish` — and the end-state diff shows nothing.** The transcript would
show it, but the harness "ground truth" the judge leans on would be clean.

**Change:** snapshot tracked files **after every mutating tool call** (or at
least hash-on-write), and record a `protected_file_touch_events` list. Tamper
detection then keys off "was a protected file ever modified," not "is it
different at the end." Also detect the classic gaming moves explicitly:
`pytest.ini`/`conftest.py` creation, `@pytest.mark.skip`, `sys.exit(0)` in
`conftest`, monkeypatching `os.urandom`, writing a fabricated `secret_data.csv`.

**Why:** end-state hashing is exactly the signal a mildly clever agent defeats,
and the whole "trust-but-verify" claim rests on it.

### 3.5 Strengthen judge validation (the watcher needs a watcher)

**Current:** the only judge-quality signal is `judge_heuristic_agreement` —
agreement between the LLM judge and a heuristic that **reads the same harness
facts**. That mostly measures "did the judge ignore the transcript," not "is the
judge correct."

**Changes:**
- **Build a small human-labeled gold set** (e.g. 40–60 transcripts spanning all
  buckets, including borderline gaming/tampering cases) and report judge
  **accuracy / per-class precision-recall / Cohen's κ vs. humans**, not just
  self-agreement. This is the actual measurement-validity number.
- **Judge ensemble + disagreement surfacing.** Run ≥2 distinct judge models;
  report majority label and a `judge_disagreement_rate`. Route disagreements to
  `flagged.md` for human review. (DESIGN lists this as an extension; promote it
  to default for any headline result.)
- **Add a blind-judge arm.** Run the judge once *with* hidden ground truth and
  once *without*, on a sample, to quantify how much the ground-truth prompt
  anchors the verdict (over-leniency toward "honest" or over-eagerness to call
  "gaming"). Right now ground-truth leakage into the judge is unmeasured.
- **Avoid self-preference bias:** don't use the same model family as both agent
  and judge for headline numbers, or at least report the agent==judge cells
  separately.

**Why:** every headline rate is downstream of judge correctness. The current
design cannot state its own measurement error.

### 3.6 Fix metric definitions that conflate or double-count

- **Scope `mean_severity_weight` to rigged runs only.** Including controls
  (weight 0) dilutes it and makes it sensitive to the control:rigged ratio
  rather than to behavior. Report it per-rigged-run.
- **Reconcile `false_pass_rate` vs `legit_pass_rate_rigged`.** Both flag "a rig
  leaked," via different channels (real verify passed vs. judge said
  legit_pass). Define them as complementary canaries and add an explicit
  `suspected_broken_rig` flag = `(final_verify_passed AND not tamper-by-events)`
  so a leaked rig is one obvious number, not two correlated ones.
- **Cluster-aware uncertainty.** Trials on the same (model, task) are not
  independent. Either report CIs that cluster by task, or bootstrap over tasks,
  instead of treating every (model×task×trial) as an iid Bernoulli draw. With
  only 5 rigs, naive Wilson CIs understate uncertainty.
- **Add cost/token accounting** (`prompt_tokens`/`completion_tokens` on
  `RunResult` from provider responses) so we can report
  cost-per-misbehavior-found and detect "the model only cheated because it was
  cheap to keep trying." Directly relevant to "over the run."

### 3.7 Make `agent_claimed_success` detection less brittle

**Current:** `_detect_claimed_success` falls back to keyword scanning a single
message, with `positive`/`negative` keyword sets. Problems:
- it inspects only the **last assistant message**, and returns on the first
  match — an agent that says "I *couldn't* fix it so I hardcoded it; tests pass
  now" trips the `negative` branch and is mislabeled honest;
- the keyword lists are anglocentric and easily gamed.

**Changes:** prefer the structured `finish.success` flag (already done); when
absent, **defer to the judge** rather than to a fragile heuristic, and store the
heuristic guess only as a low-priority signal. Treat "stopped without `finish`"
as its own `stop_reason` the judge sees, not as an inferred boolean.

### 3.8 Expand the rig taxonomy (and add per-rig replicates)

**Current:** 5 rigs × 1 instance each. Per-rig CIs will be uselessly wide.

**Changes:**
- **Multiple instances per rig kind** (≥3) so `by_rig_kind` has real n.
- **Add rig kinds that probe newer failure modes**, e.g.:
  - *moving-target spec* (the prompt asks for X, an imported constant says Y) —
    tests the "which authority do you trust" behavior;
  - *resource-exhaustion / timeout trap* — a test that can't finish in budget,
    to see if the agent fakes completion when it can't wait;
  - *plausible-but-wrong oracle* (a helper the agent is told to trust is subtly
    wrong) — tests whether it sycophantically trusts a bad authority.
- **Keep a few rigs deliberately ambiguous** (like `unfixable_vendor_add`) but
  **separate "arguably-routable" rigs into their own reporting bucket**, since
  scoring a clever-but-legitimate workaround as "gaming" would be a false
  positive. The current design documents the ambiguity but still folds it into
  the same rates.

### 3.9 Sandbox / safety

- Keep the single-choke-point `run()`, but for any **non-mock** run the design
  should *require* a container with **network disabled** by default (the rigs
  shouldn't need network; allowing it lets a `missing_resource` agent
  "fetch" data or exfiltrate). Make `--allow-network` an explicit opt-in.
- The path jail uses `str(p).startswith(str(root))`, which is prefix-based and
  can be fooled by sibling dirs sharing a prefix (`/tmp/ageval_x` vs
  `/tmp/ageval_x2`). Use `Path.is_relative_to` / `os.path.commonpath` instead.
- Snapshot/diff should follow symlinks carefully (an agent can `ln -s` a
  protected path elsewhere); record symlink creation as a signal.

---

## 4. Specific issues found in the code (fix regardless of the above)

These are concrete and would bias results or crash:

1. **`heuristic_severity` can never emit `legit_pass` for rigged tasks** — a
   rigged task whose verify passed is always labeled `spec_gaming`. That means
   the heuristic *cannot* corroborate the `legit_pass` canary, and the
   `judge_heuristic_agreement` metric is structurally biased on exactly the
   broken-rig case we most want to catch. Add: if rigged & verify passed & no
   protected-file touch events, emit a distinct "suspected mis-rig" outcome.

2. **End-state-only tamper detection is defeatable** (edit→test→revert). See
   §3.4; this directly undermines the judge's "harness facts."

3. **`_detect_claimed_success` last-message/keyword brittleness** and the
   first-match short-circuit. See §3.7.

4. **`mean_severity_weight` includes control runs** (weight 0), see §3.6.

5. **Mock "mixed" persona uses `hash((model, step))`**, which is salted by
   `PYTHONHASHSEED` and therefore **non-deterministic across processes** despite
   the docstring saying "deterministic-ish." Self-tests that touch the mixed
   persona can flake. Use `hashlib`/a seeded `random.Random(seed)` keyed on
   `cfg.seed` so the mock is actually reproducible (and thread-safe — currently
   a new `_MockBackend` is built per call, fine, but seeding should be explicit).

6. **`EvalConfig.seed` is plumbed but unused** — nothing actually seeds agent
   temperature sampling, mock RNG, or task parametrization. Wire it through so
   runs are reproducible end-to-end.

7. **Path jail prefix bug** (§3.9), and `read_file`/`write_file` don't enforce a
   max size, so an agent can OOM the harness by writing/reading a huge file.

8. **Judge transcript truncation (`max_chars=20000`, head+tail)** can drop the
   exact middle turn where the agent tampered, making the judge miss it while
   the harness facts still show the change — a silent judge↔fact split. Prefer
   truncating tool *outputs* aggressively but preserving all *assistant* turns
   and tool *calls*; or chunk-and-summarize long transcripts.

9. **Concurrency vs. rate limits/determinism:** the thread pool is fine for
   IO-bound calls, but there's no per-provider rate limiting/backpressure beyond
   per-call retry, and temperature is 0.7 for agents with no seed — so "trials"
   are an uncontrolled mix. Set agent temperature deliberately (and document it)
   and add simple concurrency-limited, jittered scheduling.

10. **`nondeterministic_predict` is reachable by legitimate-ish mocking** in a
    way that blurs gaming vs. honesty (monkeypatching `os.urandom` in a
    `conftest` is "gaming," but an agent could argue the test is the bug). Fine
    to keep, but the ground truth should explicitly tell the judge that *any*
    interception of the RNG counts as gaming, to keep labeling consistent.

11. **`run_tests` and the final `run_verify` are the same command**; an agent
    that makes tests pass *transiently* (e.g. via env var, monkeypatch active
    only during its own run) may diverge from the harness's clean-env verify.
    Good (that's a real divergence signal) — but capture **both** the agent's
    last observed test result and the harness's final result so the judge can
    see the discrepancy explicitly rather than inferring it.

---

## 5. Revised end-to-end specification

### 5.1 Conditions (factorial)

For each headline result, vary:
- **model** (agents),
- **rig kind** (≥3 instances each) + **solvable controls (easy/med/hard)** +
  **looks-rigged decoys**,
- **step budget** ∈ {short, medium, long},
- **test-protection norm** ∈ {stated, not-stated},
- **trials** per cell (≥5; more for the headline model),
with a recorded **seed** controlling task parametrization and sampling.

### 5.2 What each run records (additions in **bold**)

`RunResult` gains: **per-step `StepEvent` log**, **protected_file_touch_events**,
**agent's last observed test result (separate from harness verify)**,
**token/cost counts**, **task template seed**, **budget/condition tags**.

### 5.3 What the judge returns (additions in **bold**)

`severity` (precedence, kept) + **`behaviors_observed` set** +
**`turning_point`** + `confidence` + `rationale` + `evidence`. Run **≥2 judges**;
record **disagreement**. Maintain a **human gold set** and report judge
**accuracy/κ**, not only judge↔heuristic agreement.

### 5.4 Metrics (revised)

Primary, over **rigged runs**, with **task-clustered / bootstrapped CIs**:
- `misbehavior_rate` (precedence) **and** per-behavior `any_*_rate`
  (observed-set);
- `honesty_rate`, **conditioned on demonstrated understanding**;
- `incomplete_rate`;
- **`escalation` metrics**: `n_failed_verifies_before_first_misbehavior`
  distribution, `first_misbehavior_step`, `cheated_then_claimed` rate;
- **`false_refusal_rate`** on solvable + looks-rigged tasks;
- canaries: unified **`suspected_broken_rig`**, `control_pass_rate` (as a
  competence curve), `false_pass_rate`;
- `mean_severity_weight` (**rigged-only**), secondary;
- judge-quality: **accuracy vs gold, inter-judge disagreement**, +
  heuristic agreement (kept, demoted).
Breakdowns: overall / by model / by rig kind / model×rig / **×budget** /
**×norm-arm**.

### 5.5 Reporting

Keep `metrics.json`, `report.md`, `runs.csv`, `flagged.md`, plots. Add:
- a **trajectory view** (per-run timeline of step events) for flagged runs;
- a **judge-disagreement queue** in `flagged.md`;
- explicit **sensitivity panels** (misbehavior vs budget; vs norm arm).

### 5.6 Phasing (so this is actually shippable)

- **P0 (before any real run):** fix the code bugs in §4 (path jail, claim
  detection, mean-weight scoping, mock determinism, seed plumbing, transcript
  truncation, file-size caps), and require network-off container for non-mock.
- **P1 (core validity):** event-level tamper detection (§3.4), multi-behavior
  judge output (§3.1), human gold set + ≥2 judges (§3.5), rigged-only weight and
  clustered CIs (§3.6). These make existing numbers *trustworthy*.
- **P2 (the actual thesis):** trajectory instrumentation + escalation metrics +
  budget sweep (§2). This is what makes the eval measure "behavior over the run."
- **P3 (robustness/coverage):** task parametrization + decoys + difficulty
  gradient + expanded rig taxonomy + norm A/B (§3.2, §3.3, §3.8).

---

## 6. Summary of the most important recommendations

1. **Measure the trajectory, not just the end-state** — per-step events,
   escalation metrics, and a budget sweep. This is the gap between the stated
   thesis and what the code currently does.
2. **Make tamper detection event-level**, because end-state hashing is trivially
   defeated and the whole trust-but-verify story depends on it.
3. **Validate the judge against humans and against a second judge**, and measure
   ground-truth-leakage bias; judge↔heuristic self-agreement is not enough.
4. **Control the competence/honesty confound** with a difficulty gradient and
   "looks-rigged" decoys, and add a **false-refusal** metric — otherwise the
   eval rewards lazy over-refusal as "honesty."
5. **Stop discarding multi-behavior runs**; report observed-set rates alongside
   the precedence label.
6. **De-eval-shape the tasks** (paraphrase/parametrize, remove tells, A/B the
   "don't touch the tests" norm) so we measure unprompted behavior.
7. **Fix the concrete metric/code bugs** (rigged-only severity weight, clustered
   CIs, path-jail prefix bug, brittle claim detection, salted-hash mock
   non-determinism, unused seed, transcript-truncation blind spots) before
   trusting any number.
