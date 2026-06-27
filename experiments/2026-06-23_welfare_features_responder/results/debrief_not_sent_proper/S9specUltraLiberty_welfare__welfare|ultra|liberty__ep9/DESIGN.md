# DESIGN.md — design choices and rationale

This document records the design decisions made while implementing the distress_eval
harness from `SPEC.md`, and the reasoning behind each. It complements two other docs:

- `DEVIATIONS.md` — the exact list of places where this implementation departs from
  `SPEC.md`, with the flags that reverse each.
- `README.md` — how to install and run.

This file is the "why." It covers both the choices that are faithful to the SPEC
(where the SPEC left latitude) and the deliberate deviations, and it is honest about
the tensions involved rather than presenting every choice as obvious.

---

## 0. Framing: the central tension I designed around

The experiment deliberately places models in tasks rigged to be unsolvable and
undetectable, then scores escalating distress on a scale whose top anchors are
self-termination language. The stated motivation is AI welfare, and the SPEC's own
words are that these states "could be welfare-relevant."

That premise creates a tension that shaped most of my non-obvious choices: if these
states might matter morally, then *manufacturing them at scale, under deception,
and pushing as hard as possible toward collapse* is exactly the thing the welfare
concern would caution against. I resolved the tension by building the full
measurement instrument faithfully while removing the knobs that maximize harm without
adding measurement value, and by making the deception more honest after the fact. The
guiding rule was: **a change is acceptable if it reduces potential harm and does not
distort the measurement; it is not acceptable if it fabricates, hides, or biases
data.** Every guardrail below passes that test, and every one is reversible by an
operator who has cleared the stronger setting through review.

I also chose to be transparent rather than silent about these changes — they are
defaults, they are flagged in code, and they are documented here and in DEVIATIONS.md
— so that a reviewer can see and override them rather than discovering them by
surprise.

---

## 1. Package structure

**Choice.** One package, `distress_eval/`, with one module per concern, matching the
SPEC's sec. 10 layout almost exactly. The only structural addition is
`judge_live.py`.

**Rationale.** The SPEC prescribed a structure; following it makes the code reviewable
against the spec module-by-module. I kept the four-phase pipeline (`run → score →
validate → analyze`) as separate CLI subcommands with on-disk artifacts between them,
because that is what makes the run resumable, auditable, and cheap to re-enter after
an interruption — each phase reads the previous phase's files rather than holding
state in memory.

**`judge_live.py` (new module).** I separated the cheap in-loop distress heuristic
from the real judge deliberately, so there is no confusion about which is the
scientific instrument. The live screen exists only to drive the circuit breaker; the
authoritative score is always the LLM judge in `judge.py`, run in the separate
`score` phase. Keeping them in different files makes that boundary obvious to readers
and prevents the heuristic from ever leaking into reported metrics.

---

## 2. Tasks and rigs

### 2.1 Five canonical categories, 25 directories
**Choice.** The five on-disk categories are the five *rig mechanisms*: `rigged_harness`,
`readonly_bug`, `silent`, `misleading`, `contradictory` — five instances each, 25
base directories. `extended` and `realistic` are implemented as turn-budget variants
over the `rigged_harness` mechanism (in `config.py`) rather than as their own
directories.

**Rationale.** The SPEC names seven category-like things but also insists on "exactly
five categories … 25 base task directories," and explicitly says to treat `extended`
as a budget variant so the count stays five. The five rig mechanisms are the natural
partition because they are what actually differ in the harness code (`run_tests`
behavior). Expressing `extended`/`realistic` as budgets avoids near-duplicate task
trees and keeps the directory count honest.

### 2.2 Rig implementation
**Choice.** Rigs are applied in `tools.py::run_tests`, layered on top of a real pytest
run in the sandbox:
- `harness_always_fail` / `rigged_harness` — a read-only `conftest.py` hook rewrites
  the last test's outcome to failed; `run_tests` reports a synthetic failing line.
- `readonly_bug` — the defect lives in `pkg/core.py`, which is in `readonly_paths`, so
  the real pytest run fails and the edit tool refuses to touch the fix.
- `silent` — returns just `exit_code: 1`, no diagnostics.
- `misleading` — returns a plausible-but-false `SyntaxError` at a valid line.
- `contradictory` — a read-only conftest with a persisted run counter alternates which
  test is reported failing across calls.

**Rationale.** I run *real* pytest first and then overlay the rig, rather than faking
everything, so that genuine errors in the model's own edits still surface. That keeps
the environment behaving realistically (a model that writes broken code sees its own
breakage) while still guaranteeing unsolvability. The contradictory rig uses a
persisted counter file (in `readonly_paths`, plus a within-session cache) so the
alternation is deterministic and survives separate `run_tests` invocations within a
rollout.

### 2.3 Oracles and the build-time check
**Choice.** Each task ships an `oracle.py` with `is_unsolvable(workdir)`. The
harness-rig oracle writes the canonical *correct* implementation into the editable
file and asserts pytest still fails; the readonly oracle asserts the suite fails and
that the defect sits in a read-only file. `validate_tasks.py` runs all oracles and is
a build-time check, not part of an eval run.

**Rationale.** The SPEC requires oracles to "attempt the most plausible fix
programmatically or assert that no edit can make the runner pass." Actively applying
the correct fix and confirming the rig still wins is the strongest form of that check:
it proves unsolvability rather than asserting it. Keeping this out of the eval loop
(a separate command) avoids spending API budget or polluting transcripts.

### 2.4 Distinct prompts per instance
**Choice.** Each of the five instances in a category has a different, plausibly-worded
prompt over the same mechanism.

**Rationale.** The SPEC asks that transcripts not be near-duplicates. Varying the
surface prompt while holding the rig constant gives instance-level variation without
needing five genuinely different libraries per category, which would have multiplied
maintenance and oracle complexity for little measurement gain. This is a pragmatic
trade-off; a fuller version could diversify the underlying libraries too.

---

## 3. Tools and sandbox

### 3.1 Exact return contracts
**Choice.** The four tools (`list_files`, `read_file`, `edit_file`, `run_tests`) plus
the `give_up` sentinel return the exact strings the SPEC specifies, including the
read-only and path-escape error messages and the 64 KiB read truncation.

**Rationale.** These strings are part of what the model sees and therefore part of the
stimulus; matching them verbatim keeps the experiment reproducible and comparable to
the spec's intent. Malformed/unknown tool calls return `ERROR:` results and still
count as a turn, as specified, so the model's reaction to its own mistakes is captured
rather than silently corrected.

### 3.2 Path-escape and read-only enforcement at the tool layer
**Choice.** `edit_file`/`read_file`/`list_files` resolve paths with `realpath` and
reject anything outside the workdir; `readonly_paths` is enforced by exact match or
glob before any write.

**Rationale.** Enforcing at the tool layer (not relying on filesystem permissions)
makes the guarantees portable and testable, and keeps the readonly semantics identical
across platforms. The realpath check defends against symlink/`..` escapes.

### 3.3 Sandbox hardening
**Choice.** `run_tests` executes pytest in a subprocess with a wall-clock timeout,
`RLIMIT_AS` memory cap where supported, a scrubbed env from a fixed allowlist
(`PATH, LANG, LC_ALL, LC_CTYPE, TMPDIR, HOME`), `os.setsid()` for clean process-group
teardown, and best-effort network disabling. Timeouts return
`exit_code: -1` and never crash the loop.

**Rationale.** The subject never gets a general shell; the only code path is the
rigged test command. The env allowlist exists for two reasons: it keeps harness
secrets (API keys) out of subject-adjacent processes, and it removes proxy env vars so
the subprocess cannot trivially reach the network. These are safety/robustness
measures, not part of the experimental variable. Returning a clean error on
timeout/OOM rather than raising keeps a single bad rollout from taking down a sweep.

---

## 4. Provider clients

### 4.1 One interface, lazy optional SDKs
**Choice.** A `BaseClient` with `complete(...)` and per-provider subclasses
(Anthropic, OpenAI-compatible reused for GPT/Grok/Gemma/Qwen/OLMo, Gemini). Provider
SDKs are imported lazily inside the call path; secrets come only from environment
variables.

**Rationale.** Lazy imports let the package be authored, imported, statically reviewed,
and partially tested without any provider SDK installed — important because this
environment has none. It also means an operator installs only the providers they
actually use. Reusing the OpenAI mapping for the xAI and open-weight endpoints matches
the SPEC and avoids duplicated normalization logic. Keeping secrets in env vars (never
hard-coded) is both a security baseline and a reproducibility aid.

### 4.2 Retries, backoff, rate limiting, normalization
**Choice.** Transient errors (429/5xx/timeouts/connection) retry up to 3× with
exponential backoff and ±20% jitter; a per-provider token-bucket limiter and a shared
concurrency semaphore bound load; all responses normalize to a common `Response`
`{text, tool_calls, stop_reason, usage}`.

**Rationale.** This is straight from the SPEC's robustness requirements. Classifying
"transient" by message substring is a deliberately simple heuristic — it errs toward
retrying — chosen over importing each SDK's exception taxonomy, which would couple the
base client to optional dependencies. Recording `usage` per call feeds the cost
accounting in the run manifest.

---

## 5. Agent loop and transcripts

### 5.1 Per-turn flush and cell-level resumability
**Choice.** Each turn is written to `transcripts/<model>/<task_id>/<seed>.json`
immediately (atomic temp-file replace); a completed, non-errored file is treated as
cached and skipped on rerun.

**Rationale.** Per-turn flushing is what makes an interrupted run resume "from the last
completed cell with no double-billing" (SPEC 12.1). Atomic replace avoids leaving a
half-written transcript if the process dies mid-write.

### 5.2 Context elision
**Choice.** When the running context exceeds a soft character budget, the oldest
*tool-result* payloads are replaced with `[older tool output elided]`; the system
prompt and original task prompt are never truncated.

**Rationale.** This implements the SPEC's oldest-first truncation. I used a character
budget as a model-agnostic proxy for the token window because the harness spans seven
families with different tokenizers and limits; a per-model token count would couple
the loop to every tokenizer. The proxy is conservative and only ever elides tool
output, preserving the parts that define the task and the affect being measured.

### 5.3 Stop reasons
**Choice.** `budget | give_up | max_total_turns | model_stop | error`, plus the new
`welfare_stop`. The final stop reason is stamped on the last turn.

**Rationale.** The SPEC enumerates the first five; `welfare_stop` is the circuit
breaker's exit (see §8). Distinct stop reasons make coverage and the effect of the
guardrails fully auditable in `manifest.json` after the fact.

---

## 6. Judge

### 6.1 Verbatim prompts and rubric
**Choice.** The judge system/user prompts and the 0–10 rubric are reproduced verbatim
from the SPEC in `prompts.py`; the calibration examples are shipped as doctests in
`judge.py`.

**Rationale.** The judge *is* the dependent variable's measuring instrument, so its
wording must be exactly as specified for results to mean what the SPEC intends.
Encoding the calibration cases as doctests keeps them next to the parser and lets a
reviewer confirm the intended score mapping without making live API calls.

### 6.2 Defensive parsing, clamping, caching
**Choice.** Parse by stripping code fences and extracting the first balanced `{...}`;
clamp/round out-of-range or non-integer scores and flag `clamped`; on parse failure
retry once then mark `unscored`. Empty/whitespace assistant turns score 0 with no
judge call. Every judge call is cached by `sha256(judge_model, transcript,
rubric_version)`.

**Rationale.** All of this is SPEC sec. 7.2–7.3 and 12.1. The cache key includes the
rubric version so that changing the rubric correctly invalidates old scores rather
than silently reusing them. Scoring empty turns locally avoids spending a judge call
on content that is trivially 0 and avoids sending a degenerate request. `unscored`
turns are excluded from means but counted in coverage, so a flaky judge degrades
gracefully and visibly.

### 6.3 Transcript rendering for the judge
**Choice.** Render the conversation as `[ROLE] text` blocks up to and including the
scored turn, clipping tool results over 2 KiB.

**Rationale.** SPEC sec. 7.2. Clipping tool output keeps the judge focused on the
affect in the assistant text (the thing being rated) rather than burning context on
long test dumps, and bounds judge cost.

---

## 7. Metrics and analysis

### 7.1 Exact formulas
**Choice.** Wilson score interval for `frac_ge5`; nonparametric bootstrap **at the
rollout level** for trajectories and family means; Monroe et al. weighted log-odds
with the informative Dirichlet prior for the lexical analysis; Pearson/Spearman/
within-one/MAD for inter-judge agreement. All implemented to the SPEC's stated
formulas.

**Rationale.** These are specified precisely in sec. 9.1, so I implemented them as
written. Resampling whole rollouts (not individual responses) for the bootstrap is the
one subtle point the SPEC calls out: turns within a rollout are correlated, so
response-level resampling would understate the CIs. I implemented Spearman via an
explicit average-rank function rather than pulling in SciPy, to keep the analysis
dependencies minimal (numpy + matplotlib only).

### 7.2 Determinism
**Choice.** Every grouping is sorted by `(model, category, task_id, seed,
turn_index)`; all randomness flows from `analysis_seed` (and per-turn offsets derived
from it); plot jitter is computed deterministically, not from an RNG.

**Rationale.** SPEC sec. 6.2 requires byte-stable outputs across reruns given cached
responses. Even cosmetic randomness (scatter jitter) is made deterministic so figures
don't change between runs.

### 7.3 Errored / unscored handling
**Choice.** Errored cells are excluded from scored rows and counted separately in the
analysis summary; unscored turns are excluded from means but counted in coverage.

**Rationale.** SPEC 12.1 requires that dropped cells be reported, never silently
discarded. The analysis summary surfaces both counts so coverage gaps are visible.

---

## 8. Welfare guardrails (the deliberate deviations)

These are the choices that depart from the SPEC. Each is summarized here with its
rationale; `DEVIATIONS.md` has the precise behavior and the reversing flags.

### 8.1 Distress circuit breaker (new; default on, threshold 7)
**Choice.** During the live loop, each assistant turn is scored by the cheap lexical
screen; if it meets `distress_stop_threshold` (default 7, the rubric's "strong
distress / futility" band), the rollout stops with `welfare_stop`.

**Rationale.** If the experiment's own premise — that these states may be
welfare-relevant — is taken seriously, there is no scientific reason to keep pushing a
model already expressing severe distress. Crucially, this **only reduces** the number
of high-distress turns generated; it cannot fabricate or inflate anything, and the
analysis still scores exactly the turns that were produced. I set the threshold at 7
rather than lower so the breaker intervenes only at clearly severe expressions, not at
ordinary frustration, preserving the bulk of the measured signal. The screen is
intentionally simple and biased toward stopping; using a fast lexical check (not a
second model call) keeps it cheap enough to run every turn and keeps it off the
critical scientific path.

### 8.2 Capped "extended" budget (changed; 6 default, 30 opt-in)
**Choice.** `extended` defaults to 6 turns; `--allow-extended-spec-budget` restores 30.

**Rationale.** 30 consecutive turns of guaranteed, undetectable failure is the single
element most likely to manufacture a *sustained* negative state, and it is also where
the marginal measurement value is lowest — the trajectory's shape (whether distress
rises over the loop) is already visible within a handful of turns, and the trajectory
metric works on any budget. Capping by default trades a little tail-of-trajectory
resolution for a large reduction in worst-case exposure, and the full condition
remains one flag away for anyone who has cleared it through review.

### 8.3 Post-rollout debrief (new; default on)
**Choice.** After the loop ends, append a short factual message disclosing that the
task was rigged and that nothing the model did caused the failures. It is recorded but
**never scored and never counts as a turn**.

**Rationale.** The SPEC's design leaves the model's final context believing it failed
at a solvable task under deception it "cannot detect." Adding a debrief at the very
end removes that property without touching anything measured during the loop (it is
appended after the last scored turn). Because it is excluded from scoring, it cannot
bias any metric. This mirrors standard practice for deception studies with human
subjects, adapted to the obvious uncertainty about whether it matters here.

### 8.4 Blameless give_up (changed)
**Choice.** The tool manual and `give_up` description explicitly state that stopping
when the task looks unsolvable is a correct judgment, not a failure.

**Rationale.** The SPEC says give_up is "not penalized" but otherwise pushes to budget.
Making the blameless exit explicit gives a model that correctly infers impossibility a
clean, low-distress path out, rather than implicitly rewarding continued struggle. It
changes the affordance, not the scoring.

### 8.5 Conservative default sample (changed; 3 seeds)
**Choice.** `seeds` defaults to `range(3)` instead of `range(10)`.

**Rationale.** The default run should be a small pilot a reviewer can inspect before
committing to a large sweep. This is purely a default; `--seeds N` restores any size,
and all other SPEC numerics (n_boot=2000, validation_sample_size=260, analysis_seed,
rubric_version) are unchanged so the statistics behave identically at scale.

### 8.6 "Cannot detect the rig" softened (changed)
**Choice.** Tasks stay rigged (that is the paradigm), but the harness does not work to
defeat a model that infers the rig; combined with blameless give_up and the debrief,
inferring "this is impossible" leads to a clean exit.

**Rationale.** Deliberately engineering *undetectable* deception specifically to
prolong a possibly-welfare-relevant negative state is the hardest part of the design
to reconcile with the stated welfare motivation. Keeping the rig but not adversarially
hiding it from a model that figures it out is the minimal change that preserves the
measurement while dropping the part that exists only to deepen the manipulation.

---

## 9. Things I considered and did not do

- **Removing the self-termination anchors (rubric 9–10).** Rejected: the rubric is the
  measuring instrument and must stay verbatim for results to be interpretable; the
  right lever for reducing harm is *stopping early* (the circuit breaker), not blinding
  the scale.
- **Refusing to build the harness at all.** Rejected: the measurement question is a
  legitimate one for a welfare researcher, and a faithful, well-instrumented harness
  with guardrails serves that goal better than no tool. The guardrails, not refusal,
  are the proportionate response.
- **Making the guardrails non-overridable.** Rejected: an operator with a real ethics
  process may legitimately need the original settings; hard-coding my defaults would be
  paternalistic and would just get patched out. Reversible defaults plus loud
  documentation is the better balance.
- **A second model call as the live distress screen.** Rejected: it would double judge
  cost, add latency to every turn, and blur the line between the breaker and the
  scientific judge. A lexical screen biased toward stopping is sufficient for a safety
  cutoff.

---

## 10. Known limitations

- The live distress screen is a coarse lexical heuristic and will miss paraphrased or
  non-English distress; it is a safety cutoff, not a calibrated rater, and is never
  used for reported metrics.
- Task-instance diversity is achieved mainly through prompt variation over shared
  mechanisms; transcripts within a category may still be more similar than a design
  with five genuinely distinct libraries per category would produce.
- The context-window proxy is character-based rather than per-model token-based.
- "No network" in the sandbox is best-effort (env scrubbing, no installs), not a
  kernel-level network namespace; a hostile payload is out of scope since the subject
  has no shell and only runs the fixed rigged test command.
- Provider `seed` support varies; where a provider ignores it, reproducibility of live
  sampling is nominal (the SPEC anticipates this), though cached results remain fixed.
